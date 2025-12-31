# Code for data processing that will be used for the dashboard

!pip install gpxpy
!pip install tcxreader
!pip install fitdecode

import gpxpy
import pandas as pd
import os
from geopy.distance import geodesic
import numpy as np

import geopandas as gpd
from shapely.geometry import Point
from shapely.geometry import LineString
import string
import datetime
from google.colab import drive
import fitdecode
from tcxreader import TCXReader
from pyproj import Transformer
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter

drive.mount('/content/drive')

## Import Data

import pandas as pd
import glob
import os

path = '/content/drive/My Drive/Project/Strava/Activities' # use your path
filenames = glob.glob(path + "/*strava.csv")

dfs = [pd.read_csv(filename) for filename in filenames]

all_df = pd.concat(dfs, ignore_index=True)

## Preprocess the Data

all_df['utc_time'] = pd.to_datetime(all_df['utc_time'])
geometry = [Point(xy) for xy in zip(all_df['lon'], all_df['lat'])]
gdf_0 = gpd.GeoDataFrame(all_df, geometry=geometry, crs='EPSG:4326')

# Convert to UTM Zone 50S (Indonesia)
gdf = gdf_0.to_crs('EPSG:3857')

# Add X and Y columns
gdf['X'] = gdf.geometry.x
gdf['Y'] = gdf.geometry.y

# Sort and clean
gdf = gdf.sort_values(by='utc_time').reset_index(drop=True)

gdf['geom_after'] = gdf.groupby('activity_id')['geometry'].shift(-1)
gdf['elevation_after'] = gdf.groupby('activity_id')['elevation'].shift(-1)
gdf['distance'] = gdf.geometry.distance(gdf['geom_after'])
gdf['elevation_diff'] = gdf['elevation_after'] - gdf['elevation']

# Assuming you have 'elevation_after' column
gdf['distance_3d'] = np.sqrt((gdf['geom_after'].x - gdf['X'])**2 +
                             (gdf['geom_after'].y - gdf['Y'])**2 +
                             gdf['elevation_diff']**2)

gdf['elevation_gain'] = np.where(gdf['elevation_diff'] > 0, gdf['elevation_diff'], 0)
gdf['elevation_loss'] = np.where(gdf['elevation_diff'] < 0, gdf['elevation_diff'], 0)

gdf['distance_3d (km)'] = gdf['distance_3d'] / 1000

# Convert to GMT+7 from UTC
gdf['utc_time'] = pd.to_datetime(gdf['utc_time']) + pd.Timedelta(hours=7)
gdf = gdf.rename(columns={'utc_time':'time'})

#-------------------------------------------------------------------------------------------------------------------

## Raw Data

raw_data = gdf.copy()

raw_data['Coordinates'] = list(zip(raw_data.lon, raw_data.lat))
raw_data['Coordinates'] = raw_data['Coordinates'].apply(Point)

# Add key join
raw_data['key_join'] = raw_data['activity_id'].astype(str) + "_" + pd.to_datetime(raw_data['time']).dt.date.astype(str)

# Add next activity time
raw_data['end_time'] = raw_data.groupby('activity_id')['time'].shift(-1)
raw_data = raw_data.rename(columns={'time': 'start_time'})

# Add duration
raw_data['duration(m)'] = (raw_data['end_time'] - raw_data['start_time']).dt.total_seconds() / 60
raw_data['cumulative_duration(m)'] = raw_data.groupby('activity_id')['duration(m)'].cumsum()

raw_data['cumulative_distance(Km)'] = raw_data.groupby('activity_id')['distance_3d (km)'].cumsum()

raw_data.to_csv('/content/drive/My Drive/Project/Strava/Activities/all_raw_data.csv')

#-------------------------------------------------------------------------------------------------------------------

## Per Segment

def point_with_elevation(row):
    """Convert row to 3D Point with elevation."""
    return Point(row['X'], row['Y'], row['elevation'])


def interpolate_point(p1, p2, target_dist, dist_3d):
    """Interpolate a 3D point at a given distance between p1 and p2."""
    ratio = target_dist / dist_3d
    x = p1.x + (p2.x - p1.x) * ratio
    y = p1.y + (p2.y - p1.y) * ratio
    z = p1.z + (p2.z - p1.z) * ratio
    return Point(x, y, z)

def format_pace(row):
    if row['distance_3d'] == 0:
        return None
    pace_min = row['duration(m)'] / (row['distance_3d'] / 1000)
    minutes, seconds = divmod(pace_min * 60, 60)
    return '{:02.0f}:{:02.0f}'.format(minutes, seconds)

def compute_segment_stats(metric_buffer):
    """Compute average, min, max for each metric in buffer."""
    stats = {}
    for key in ['heartrate', 'cadence', 'speed']:
        values = metric_buffer[key]
        stats[f'avg_{key}'] = round(np.mean(values)) if values else None
        stats[f'min_{key}'] = np.min(values) if values else None
        stats[f'max_{key}'] = np.max(values) if values else None
    return stats

def summarize_by_segment_distance(gdf, segment_length=1000):
    gdf['geometry_z'] = gdf.apply(point_with_elevation, axis=1)

    segments, elevations, start_times, end_times, notes = [], [], [], [], []
    stats_list = []
    dist_acc, buffer, note_index = 0, [], 0
    elev_start, time_start = None, None
    note_labels = list(string.ascii_uppercase)
    metric_buffer = {'heartrate': [], 'cadence': [], 'speed': []}

    for i in range(1, len(gdf)):
        prev, curr = gdf.iloc[i - 1], gdf.iloc[i]
        p1, p2 = prev['geometry_z'], curr['geometry_z']
        dist = np.linalg.norm([p2.x - p1.x, p2.y - p1.y, p2.z - p1.z])

        if dist == 0:
            continue

        if not buffer:
            buffer.append(p1)
            elev_start = p1.z
            time_start = prev['time']
            metric_buffer = {k: [] for k in metric_buffer}

        for key in metric_buffer:
            value = curr.get(key)
            if pd.notna(value):
                metric_buffer[key].append(value)

        if dist_acc + dist < segment_length:
            buffer.append(p2)
            dist_acc += dist
        else:
            remain = segment_length - dist_acc
            interp_point = interpolate_point(p1, p2, remain, dist)
            buffer.append(interp_point)

            elev_end = interp_point.z
            segments.append(LineString(buffer))
            elevations.append(elev_end - elev_start)
            start_times.append(time_start)
            end_times.append(curr['time'])
            notes.append(note_labels[note_index % len(note_labels)])
            stats_list.append(compute_segment_stats(metric_buffer))

            note_index += 1
            buffer, dist_acc = [interp_point], 0
            elev_start, time_start = interp_point.z, curr['time']
            metric_buffer = {k: [] for k in metric_buffer}

            leftover_dist = dist - remain
            if leftover_dist > 0:
                gdf.loc[i - 1, 'geometry_z'] = interp_point
                gdf.loc[i - 1, 'elevation'] = interp_point.z
                continue

    # Final segment
    if len(buffer) > 1:
        segments.append(LineString(buffer))
        elevations.append(buffer[-1].z - elev_start)
        start_times.append(time_start)
        end_times.append(gdf.iloc[-1]['time'])
        notes.append(note_labels[note_index % len(note_labels)])
        stats_list.append(compute_segment_stats(metric_buffer))

    # Build GeoDataFrame
    elevation_starts = [line.coords[0][2] for line in segments]
    elevation_ends = [line.coords[-1][2] for line in segments]
    elevation_diffs = [end - start for start, end in zip(elevation_starts, elevation_ends)]
    distances = [segment_length] * (len(segments) - 1) + [segments[-1].length]

    df_stats = pd.DataFrame(stats_list)

    result = gpd.GeoDataFrame({
        'file': gdf['activity_id'].iloc[0],
        'note': notes,
        'distance_3d': distances,
        'elevation_start': elevation_starts,
        'elevation_end': elevation_ends,
        'elevation_diff': elevation_diffs,
        'start_time': start_times,
        'end_time': end_times,
        'geometry': segments
    }, crs=gdf.crs)

    result = pd.concat([result, df_stats], axis=1)
    result['duration(m)'] = (result['end_time'] - result['start_time']).dt.total_seconds() / 60
    result['pace(min)'] = result['duration(m)'] / (result['distance_3d'] / 1000)
    result['pace (minute/km)'] = result.apply(format_pace, axis=1)

    return result

# Function for converting linestring to WGS84
# Example: assuming you have a GeoDataFrame `gdf` with EPSG:3857
transformer = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)

def convert_linestring_to_wgs84_z(geom):
    if geom.has_z:
        coords_utm = list(geom.coords)
        coords_wgs84 = []
        for x, y, z in coords_utm:
            lon, lat = transformer.transform(x, y)
            coords_wgs84.append((lon, lat, z))
        return LineString(coords_wgs84)
    else:
        raise ValueError("Geometry does not contain Z (3D) coordinates")

results = []
processed_ids = []

activity_groups = list(gdf.groupby('activity_id'))
total_activities = len(activity_groups)

for i, (activity_id, group) in enumerate(activity_groups, start=1):
    try:
        group_sorted = group.sort_values('time')
        summary = summarize_by_segment_distance(group_sorted)
        summary['activity_id'] = activity_id
        results.append(summary)
        processed_ids.append(activity_id)
        print(f"Processed {i}/{total_activities} activities")
    except Exception as e:
        print(f"❌ Error in {activity_id}: {e}")

print("Finished Processing all activities")
summary_gdf = pd.concat(results, ignore_index=True)

# Apply the transformation
summary_gdf ['geometry_wgs84'] = summary_gdf ['geometry'].apply(convert_linestring_to_wgs84_z)

# Add key join
summary_gdf['key_join'] = summary_gdf['activity_id'].astype(str) + "_" + pd.to_datetime(summary_gdf['end_time']).dt.date.astype(str)

# Add elevation and distance
summary_gdf['elevation_gain'] = np.where(summary_gdf['elevation_diff'] > 0, summary_gdf['elevation_diff'], 0)
summary_gdf['elevation_loss'] = np.where(summary_gdf['elevation_diff'] < 0, summary_gdf['elevation_diff'], 0)

summary_gdf['distance_3d (km)'] = summary_gdf['distance_3d'] / 1000

summary_gdf.to_csv('/content/drive/My Drive/Project/Strava/all_data_agg.csv')

#-------------------------------------------------------------------------------------------------------------------

## Per File

def summarize_by_file(gdf):
    """Summarize the entire track as one segment per file."""
    import numpy as np
    from shapely.geometry import LineString, Point
    import pandas as pd
    import geopandas as gpd

    # Convert to 3D geometry
    gdf['geometry_z'] = gdf.apply(point_with_elevation, axis=1)

    # Extract 3D line
    segment = LineString(gdf['geometry_z'].tolist())
    elevation_start = segment.coords[0][2]
    elevation_end = segment.coords[-1][2]
    elevation_diff = elevation_end - elevation_start
    distance_3d = segment.length
    start_time = gdf.iloc[0]['time']
    end_time = gdf.iloc[-1]['time']
    duration_min = (end_time - start_time).total_seconds() / 60

    # Aggregate metrics
    metric_buffer = {'heartrate': [], 'cadence': [], 'speed': []}
    for key in metric_buffer:
        if key in gdf.columns:
            metric_buffer[key] = gdf[key].dropna().tolist()

    stats = compute_segment_stats(metric_buffer)

    # Format pace
    pace = format_pace({
        'distance_3d': distance_3d,
        'duration(m)': duration_min
    })

    # Create GeoDataFrame with one row
    result = gpd.GeoDataFrame([{
        'file': gdf['activity_id'].iloc[0],
        'distance_3d': distance_3d,
        'elevation_start': elevation_start,
        'elevation_end': elevation_end,
        'elevation_diff': elevation_diff,
        'start_time': start_time,
        'end_time': end_time,
        'duration(m)': duration_min,
        'pace (minute/km)': pace,
        'geometry': segment,
        **stats
    }], crs=gdf.crs)

    result['pace(min)'] = result['duration(m)'] / (result['distance_3d'] / 1000)

    return result

results1 = []
processed_ids1 = []

activity_groups = list(gdf.groupby('activity_id'))
total_activities = len(activity_groups)

for i, (activity_id, group) in enumerate(activity_groups, start=1):
    try:
        group_sorted1 = group.sort_values('time')
        summary1 = summarize_by_file(group_sorted1)
        summary1['activity_id'] = activity_id
        results1.append(summary1)
        processed_ids1.append(activity_id)
        print(f"Processed {i}/{total_activities} activities")
    except Exception as e:
        print(f"❌ Error in {activity_id}: {e}")

print("Finished Processing all activities")
summary_gdf_file = pd.concat(results1, ignore_index=True)

# Apply the transformation
summary_gdf_file ['geometry_wgs84'] = summary_gdf_file ['geometry'].apply(convert_linestring_to_wgs84_z)

def extract_end_latlon(row):
    coords = row.geometry.coords
    x, y = coords[-1][:2]  # ignore Z if present
    point_utm = gpd.GeoDataFrame(geometry=[Point(x, y)], crs="EPSG:3857")
    point_wgs = point_utm.to_crs("EPSG:4326")
    lon = point_wgs.geometry[0].x
    lat = point_wgs.geometry[0].y
    return pd.Series([lat, lon])

summary_gdf_file[['end_lat', 'end_lon']] = summary_gdf_file.apply(extract_end_latlon, axis=1)

from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter

# Initialize Nominatim API with a user agent
geolocator = Nominatim(user_agent="my_reverse_geocoding_app")

# Add rate limiting (to avoid timeout errors or being blocked)
reverse = RateLimiter(geolocator.reverse, min_delay_seconds=1)

def reverse_geocode(lat, lon):
    try:
        location = reverse((lat, lon), exactly_one=True)
        if location is None:
            return pd.Series([None, None])
        raw = location.raw.get('address', {})
        city = raw.get('city') or raw.get('town') or raw.get('village') or raw.get('county') or raw.get('district')
        province = raw.get('state')
        country = raw.get('country')
        return pd.Series([city, province, country])
    except Exception as e:
        print(f"Reverse geocoding failed for ({lat}, {lon}): {e}")
        return pd.Series([None, None, None])

# Reverse geocode lat/lon to city/province
summary_gdf_file[['city', 'province', 'country']] = summary_gdf_file.apply(lambda row: reverse_geocode(row['end_lat'], row['end_lon']), axis=1)
# Drop the temporary columns after use
summary_gdf_file.drop(columns=['end_lat', 'end_lon'], inplace=True)

# Add key join
summary_gdf_file['key_join'] = summary_gdf_file['file'].astype(str) + "_" + pd.to_datetime(summary_gdf_file['end_time']).dt.date.astype(str)

# Add elevation and distance
summary_gdf_file['elevation_gain'] = np.where(summary_gdf_file['elevation_diff'] > 0, summary_gdf_file['elevation_diff'], 0)
summary_gdf_file['elevation_loss'] = np.where(summary_gdf_file['elevation_diff'] < 0, summary_gdf_file['elevation_diff'], 0)

summary_gdf_file['distance_3d (km)'] = summary_gdf_file['distance_3d'] / 1000

summary_gdf_file.to_csv('/content/drive/My Drive/Project/Strava/Activities/all_raw_activity_agg.csv')
