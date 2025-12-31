"""
# API Strava

http://www.strava.com/oauth/authorize?client_id={YOUR_ID}&response_type=code&redirect_uri=http://www.strava.com/exchange_token&approval_prompt=force&scope=activity:read_all
"""

# Run this after click the link
import requests

res = requests.post("https://www.strava.com/oauth/token", data={
    'client_id': {YOUR_ID}, # it can be found in your API application
    'client_secret': {YOUR_CLIENT_SECRET}, # it can be found in your API application
    'code': {YOUR_CODE},  # replace with the real code written in the web after clicking the above link
    'grant_type': 'authorization_code'
})

print(res.json())

#When you run above code, look for "refresh_token"

import os
import json
import time
import pandas as pd
import requests
from requests.exceptions import ConnectionError, Timeout

# =================== CONFIG =================== #
CLIENT_ID = {YOUR_ID}
CLIENT_SECRET = {YOUR_CLIENT_SECRET}
REFRESH_TOKEN = {YOUR_REFRESH_TOKEN}

DATA_FILE = "all_activities_data.csv" #It will be used to save Strava Data
PROGRESS_FILE = "progress.json" #It will be used to save the activity_id that are successfully processed from the API

#The process depends on your Strava Activities, if there are hundreds or even thousands of activities, then the process will take a lot of time
#The process also made to enable the next extraction in the next time(eg. 1 month after)

STREAM_KEYS = [
    "latlng", "time", "heartrate", "cadence", "altitude", "distance",
    "grade_smooth", "power", "vertical_ratio", "vertical_oscillation", "speed"  # ✅ MODIFIED
]

# =================== TOKEN =================== #
def get_access_token(retries=3, delay=5):
    for attempt in range(retries):
        try:
            response = requests.post("https://www.strava.com/oauth/token", data={
                'client_id': CLIENT_ID,
                'client_secret': CLIENT_SECRET,
                'refresh_token': REFRESH_TOKEN,
                'grant_type': 'refresh_token'
            }, timeout=10)

            if response.status_code == 200:
                return response.json()['access_token']
            else:
                print(f"⚠️ Error {response.status_code}: {response.text}")
                return None
        except (ConnectionError, Timeout) as e:
            print(f"⏳ Retry {attempt + 1}/{retries} failed. Reason: {e}")
            time.sleep(delay)
    print("❌ Failed to get access token after retries.")
    return None

# =================== GET ACTIVITIES =================== #
def get_all_activities(token):
    activities = []
    page = 1
    while True:
        url = f"https://www.strava.com/api/v3/athlete/activities?page={page}&per_page=200"
        response = requests.get(url, headers={"Authorization": f"Bearer {token}"})
        if response.status_code != 200:
            print("⚠️ Failed to fetch activities:", response.text)
            break
        data = response.json()
        if not data:
            break
        activities.extend(data)
        page += 1
    return activities

# =================== GET STREAMS =================== #
def get_activity_streams(activity_id, token):
    url = f"https://www.strava.com/api/v3/activities/{activity_id}/streams"
    params = {"keys": ",".join(STREAM_KEYS), "key_by_type": True}
    response = requests.get(url, headers={"Authorization": f"Bearer {token}"}, params=params)
    if response.status_code != 200:
        return None
    return response.json()

# =================== MAIN =================== #
def main():
    access_token = get_access_token()
    if not access_token:
        print("❌ Exiting: No access token.")
        return

    processed_ids = set()
    daily_count = 0
    all_data = pd.DataFrame()

    # Load progress if exists
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, "r") as f:
            progress = json.load(f)
            processed_ids = set(progress.get("processed_ids", []))
            daily_count = progress.get("daily_count", 0)

    try:
        activities = get_all_activities(access_token)
        for i, act in enumerate(activities):
            act_id = act['id']
            name = act.get("name", "")
            type_ = act.get("type", "")

            if act_id in processed_ids:
                continue

            print(f"[{i+1}/{len(activities)}] Fetching: {name} (ID: {act_id})")

            streams = get_activity_streams(act_id, access_token)
            if not streams or 'time' not in streams or 'latlng' not in streams:
                print(f"❌ Skipping activity {act_id}: Missing essential stream data.")
                continue

            num_points = len(streams['time']['data'])
            df = pd.DataFrame({
                "timestamp": streams['time']['data'],
                "lat": [pt[0] for pt in streams['latlng']['data']],
                "lon": [pt[1] for pt in streams['latlng']['data']],
                "heartrate": streams.get("heartrate", {}).get("data", [None]*num_points),
                "cadence": streams.get("cadence", {}).get("data", [None]*num_points),
                "elevation": streams.get("altitude", {}).get("data", [None]*num_points),
                "distance": streams.get("distance", {}).get("data", [None]*num_points),
                "grade_smooth": streams.get("grade_smooth", {}).get("data", [None]*num_points),
                "power": streams.get("power", {}).get("data", [None]*num_points),  # ✅ MODIFIED
                "vertical_ratio": streams.get("vertical_ratio", {}).get("data", [None]*num_points),  # ✅
                "vertical_oscillation": streams.get("vertical_oscillation", {}).get("data", [None]*num_points),  # ✅
                "speed": streams.get("speed", {}).get("data", [None]*num_points),  # ✅
                "activity_id": act_id,
                "activity_name": name,
                "activity_type": type_,
                "activity_start": act.get("start_date"),
            })

            df["utc_time"] = pd.to_datetime(df["activity_start"]) + pd.to_timedelta(df["timestamp"], unit="s")
            all_data = pd.concat([all_data, df], ignore_index=True)

            processed_ids.add(act_id)
            daily_count += 1

            # Save after each activity
            all_data.to_csv(DATA_FILE, index=False)
            with open(PROGRESS_FILE, "w") as f:
                json.dump({
                    "processed_ids": list(processed_ids),
                    "daily_count": daily_count,
                }, f)

    except Exception as e:
        print("❌ Exception occurred:", e)
    finally:
        print("✅ Final save...")
        all_data.to_csv(DATA_FILE, index=False)
        with open(PROGRESS_FILE, "w") as f:
            json.dump({
                "processed_ids": list(processed_ids),
                "daily_count": daily_count,
            }, f)
        print("✅ Done.")

main()