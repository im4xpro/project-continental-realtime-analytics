import json
import pandas as pd
from datetime import timedelta
from evidently import ColumnMapping
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
from evidently.metrics import ColumnDriftMetric, DatasetDriftMetric, DatasetMissingValuesMetric
from evidently.metrics import ColumnQuantileMetric, ColumnCorrelationsMetric

df = pd.read_json('processtimes.json')

current_time = pd.to_datetime("2024-10-23T18:11:37.111000+02:00")
reference_window = current_time - timedelta(minutes=10)
reference_data = df[df['TimeStamp'] <= reference_window]
current_data = df[df['TimeStamp'] > reference_window]

unique_station_names = df['StationName'].unique()

# Function to perform outlier detection on 'Value' for each unique StationName
def analyze_outliers(df, station_name):
    station_reference_data = reference_data[reference_data['StationName'] == station_name]
    station_current_data = current_data[current_data['StationName'] == station_name]

    # Skip if 'Value' column is empty in either the reference or current data
    if station_reference_data['Value'].empty or station_current_data['Value'].empty:
        print(f"Skipping drift analysis for {station_name} due to insufficient data in 'Value'.")
        return
    
    # Initialize an Evidently Report with Data Drift preset
    report = Report(metrics=[
        ColumnDriftMetric(column_name='Value'),
        DatasetDriftMetric(),
        DatasetMissingValuesMetric(),
        ColumnQuantileMetric(column_name='Value', quantile=0.25),
        ColumnCorrelationsMetric(column_name='Value')
    ])

    # Run the report to detect data drift
    report.run(current_data=station_current_data[['Value']], reference_data=station_reference_data[['Value']])
    
    # Extract the JSON result to check for drift
    report_json = report.as_dict()
    drift_detected = report_json["metrics"][0]["result"]["drift_detected"]

    # Save the report only if drift is detected
    if drift_detected:
        report.save_html(f"reports/{station_name}_outlier_report.html")
        print(f"Drift detected in 'Value' for {station_name}. Report saved as '{station_name}_outlier_report.html'")
    else:
        print(f"No drift detected in 'Value' for {station_name}. Report not saved.")

# Loop through each StationName with only one StationGroup and check for drift in 'Value'
for name in unique_station_names:
    analyze_outliers(df, name)
