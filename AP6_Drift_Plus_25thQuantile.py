import pandas as pd
import numpy as np
import json
from datetime import timedelta
from evidently.report import Report
from evidently.metrics import ColumnDriftMetric, DatasetDriftMetric, DatasetMissingValuesMetric
from evidently.metrics import ColumnQuantileMetric, ColumnCorrelationsMetric

# Assume df is the DataFrame with your dataset

df = pd.read_json("processtimes.json")

current_time = pd.to_datetime("2024-10-23T18:11:37.111000+02:00")
reference_window = current_time - timedelta(minutes=10)
reference_data = df[df['TimeStamp'] <= reference_window]
current_data = df[df['TimeStamp'] > reference_window]

# Evidently Data Drift Report for each station
station_names = df['StationName'].unique()
reports = {}

# Function for 25th percentile check
def is_below_25th_percentile(value, reference_series):
    threshold_25th = np.percentile(reference_series, 25)
    return value < threshold_25th

for station in station_names:
    # Filter data for each station
    station_reference_data = reference_data[reference_data['StationName'] == station]
    station_current_data = current_data[current_data['StationName'] == station]
    
    # Ensure we have data in both reference and current data windows
    if not station_reference_data.empty and not station_current_data.empty:
        # Initialize drift report with all relevant metrics
        data_drift_report = Report(metrics=[
            ColumnDriftMetric(column_name='Value'),
            DatasetDriftMetric(),
            DatasetMissingValuesMetric(),
            ColumnQuantileMetric(column_name='Value', quantile=0.25),
            ColumnCorrelationsMetric(column_name='Value')
        ])
        
        # Run report
        data_drift_report.run(reference_data=station_reference_data[['Value']], current_data=station_current_data[['Value']])
        
        # Generate drift report and save as JSON
        drift_result = data_drift_report.as_dict()
        reports[station] = drift_result

        # Perform the 25th percentile check for each row in the current data
        station_current_data['Below_25th_Percentile'] = station_current_data['Value'].apply(
            lambda x: is_below_25th_percentile(x, station_reference_data['Value'])
        )
        # Add the percentile check results to the report
        reports[station]['Below_25th_Percentile'] = station_current_data['Below_25th_Percentile'].tolist()

# Save the full reports for each station as a JSON file
with open("drift_reports.json", "w") as f:
    json.dump(reports, f)

print("Drift reports created and saved for each station.")
