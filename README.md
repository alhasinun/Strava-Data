# Strava-Data
A project to get Strava Data, process it, and then display it in dashboard

There are 2 python files for this project to be consumed by the dashboard:
1. STRAVA_API:
   It contains the code for extracting data through Strava API, you need to have a Strava Account (with at least 1 activity) and a Strava API Application.
   You can follow the tutorial here:
   https://towardsdatascience.com/using-the-strava-api-and-pandas-to-explore-your-activity-data-d94901d9bfde/

2. STRAVA_DATA_PROCESS:
   It contains the code to process the data extracted from the API.
   In general, the process is done to separate the data into 3 datasets:
   a. Raw data
   b. Data record per KM
   c. Data record per activity

If you have any questions, feel free to message me!
Thank you.
