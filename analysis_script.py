import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from evidently.report import Report
from evidently.metrics import ColumnDriftMetric, DatasetDriftMetric, DatasetMissingValuesMetric
from evidently.metrics import ColumnQuantileMetric, ColumnCorrelationsMetric
import json
from datetime import datetime, timedelta

# Load the data from JSON
data = pd.read_json("production_data_mock.json")

# Create a dictionary to store unique stations per serial number as a single string
stations_per_item = {}

for serial_number in data["SerialNumber"].unique():
    # Get unique stations for the current serial number and join them into a single string
    unique_stations = ", ".join(data[data["SerialNumber"] == serial_number]["StationName"].unique())
    stations_per_item[serial_number] = unique_stations

# Convert the dictionary to a DataFrame for easy comparison
stations_per_item_df = pd.DataFrame(list(stations_per_item.items()), columns=["SerialNumber", "Stations"])

# Identify unique station groups and assign a group ID for each unique set of stations
stations_per_item_df["Group_ID"] = stations_per_item_df.groupby("Stations").ngroup()

stations_per_item_df = stations_per_item_df.sort_values(by="Group_ID").reset_index(drop=True)


# Enrich groups with timestamp and compute total process time
first_occurrences = data.loc[data.groupby("SerialNumber")["TimeStamp"].idxmin()]

first_occurrences = first_occurrences[['SerialNumber', 'TimeStamp']]

processtime_per_station = data.groupby("SerialNumber")["Value"].sum()

stations_per_item_df = pd.merge(stations_per_item_df, processtime_per_station, on="SerialNumber", how="inner")

stations_per_item_df = stations_per_item_df.rename(columns={"Value":"ProcessTime"})
stations_per_item_df = pd.merge(stations_per_item_df, first_occurrences, on='SerialNumber', how='inner')

stations_per_item_df = stations_per_item_df.rename(columns={'TimeStamp':'FirstCheckIn'})

# Compute quantiles

# Ensure ProcessTime is numeric and valid
stations_per_item_df["ProcessTime"] = pd.to_numeric(stations_per_item_df["ProcessTime"], errors="coerce")

# Calculate the 25th percentile for each Group_ID
q25_values = (
    stations_per_item_df.groupby("Group_ID")["ProcessTime"]
    .quantile(0.25)
    .reset_index()
    .rename(columns={"ProcessTime": "Q25"})
)

q75_values = (
    stations_per_item_df.groupby("Group_ID")["ProcessTime"]
    .quantile(0.75)
    .reset_index()
    .rename(columns={"ProcessTime": "Q75"})
)

q90_values = (
    stations_per_item_df.groupby("Group_ID")["ProcessTime"]
    .quantile(0.90)
    .reset_index()
    .rename(columns={"ProcessTime": "Q90"})
)

median_values = (
    stations_per_item_df.groupby("Group_ID")["ProcessTime"]
    .median()
    .reset_index()
    .rename(columns={"ProcessTime": "Median"})
)

# Merge statistical values into dataset
stations_per_item_df = pd.merge(stations_per_item_df, q25_values, on="Group_ID", how="left")
stations_per_item_df = pd.merge(stations_per_item_df, q90_values, on="Group_ID", how="left")
stations_per_item_df = pd.merge(stations_per_item_df, q75_values, on="Group_ID", how="left")
stations_per_item_df = pd.merge(stations_per_item_df, median_values, on="Group_ID", how="left")

# Combine Items in a group into Time groups

stations_per_item_df_subgrouped = stations_per_item_df.sort_values(by=["Group_ID", "FirstCheckIn"])

stations_per_item_df_subgrouped["TimeGroup"] = stations_per_item_df_subgrouped.groupby("Group_ID").cumcount() // 10 + 1

# Calculate Exponentially Weighted Moving Average for each time group

alpha = 0.3  # EWMA smoothing factor
drift_threshold = 0.5  # Threshold for Drift Indicator

# Group by Group_ID and TimeGroup
grouped = stations_per_item_df_subgrouped.groupby(["Group_ID", "TimeGroup"])

# Apply calculations within each group
results = []
for (group_id, time_group), group_data in grouped:
    group_data = group_data.copy()
    group_data["EWMA"] = group_data["ProcessTime"].ewm(alpha=alpha, adjust=False).mean()
    group_data["EWMA_DriftIndicator"] = group_data["ProcessTime"] - group_data["EWMA"]
    group_data["EWMA_Alert"] = np.abs(group_data["EWMA_DriftIndicator"]) > drift_threshold
    results.append(group_data)

# Combine processed groups into a single DataFrame
processed_df = pd.concat(results).reset_index(drop=True)

# Perform Linear Regression with sliding windows

# Define sliding window size and slope threshold
WINDOW_SIZE = 10
SLOPE_THRESHOLD = 0.3

# Initialize a list to store results for all groups
results = []

# Loop over each unique Group_ID
for group_id in processed_df["Group_ID"].unique():
    # Extract and sort the data for the current group
    group_data = processed_df[processed_df["Group_ID"] == group_id]
    group_data = group_data.sort_values("FirstCheckIn")

    # Initialize lists for sliding window results for the current group
    sliding_means = []
    sliding_std_devs = []
    sliding_slopes = []
    timestamps = []
    window_indices = []  # To store the window indicator
    serial_numbers = []  # To store serial numbers of the sliding window

    # Perform sliding window analysis
    for i in range(len(group_data) - WINDOW_SIZE + 1):
        window = group_data.iloc[i:i + WINDOW_SIZE]
        timestamps.append(window["FirstCheckIn"].iloc[-1])  # Take the last timestamp of the window
        window_indices.append(f"{i}-{i+WINDOW_SIZE-1}")  # Save the window range as an indicator

        # Concatenate serial numbers in the current window
        serial_numbers.append(", ".join(window["SerialNumber"].astype(str)))

        # Calculate mean and standard deviation for the window
        sliding_means.append(window["ProcessTime"].mean())
        sliding_std_devs.append(window["ProcessTime"].std())

        # Fit a linear regression to the window
        X = np.arange(WINDOW_SIZE).reshape(-1, 1)
        y = window["ProcessTime"].values
        reg = LinearRegression().fit(X, y)
        sliding_slopes.append(reg.coef_[0])  # Extract the slope

    # Store results for the current group
    group_results = pd.DataFrame({
        "Group_ID": group_id,
        "Timestamp": timestamps,
        "Window": window_indices,  # Add window indicator
        "SerialNumbers": serial_numbers,  # Add serial numbers
        "MeanProcessTime": sliding_means,
        "StdDevProcessTime": sliding_std_devs,
        "Slope": sliding_slopes
    })

    # Detect upward trends based on slope threshold
    group_results["UpwardTrendDetected"] = group_results["Slope"] > SLOPE_THRESHOLD

    # Append group results to the overall list
    results.append(group_results)

# Combine results for all groups into a single DataFrame
final_results = pd.concat(results, ignore_index=True)

trend_mapping = (
    group_results
    .explode("SerialNumbers")  # Split concatenated serial numbers into individual rows
    .assign(SerialNumbers=lambda df: df["SerialNumbers"].str.split(", "))  # Split serial numbers
    .explode("SerialNumbers")  # Flatten the lists into rows
    .drop_duplicates(subset=["SerialNumbers", "UpwardTrendDetected"])  # Remove duplicates
    .rename(columns={"SerialNumbers": "SerialNumber"})  # Rename for merging
    [["SerialNumber", "UpwardTrendDetected"]]  # Select relevant columns
)

trend_mapping["SerialNumber"] = trend_mapping["SerialNumber"].astype(int)
# Step 2: Merge with the new dataset
result_report = pd.merge(trend_mapping,processed_df, on="SerialNumber", how="right")

trend_column = result_report.pop("UpwardTrendDetected")
result_report["UpwardTrendDetected"] = trend_column

# Add the EvidentlyAI analysis

# Initialize an empty list to hold results
updated_result_report = []

# Group by Group_ID and process each group
for group_id, group_data in result_report.groupby("Group_ID"):
    # Find the maximum TimeGroup for the current group
    max_timegroup = group_data["TimeGroup"].max()
    
    # Split into Current and Reference
    current = group_data[group_data["TimeGroup"] == max_timegroup]
    reference = group_data[group_data["TimeGroup"] != max_timegroup]
    
    # Create the data drift report
    data_drift_report = Report(metrics=[
        ColumnDriftMetric(column_name='ProcessTime'),
        DatasetDriftMetric(),
        DatasetMissingValuesMetric(),
        ColumnQuantileMetric(column_name='ProcessTime', quantile=0.5),
        ColumnCorrelationsMetric(column_name='ProcessTime')
    ])
    
    # Run the data drift report
    data_drift_report.run(reference_data=reference[['ProcessTime']], current_data=current[['ProcessTime']])
    
    # Get the drift result in dictionary format
    drift_result = data_drift_report.as_dict()
    print(drift_result)
    
    # Extract the drift detection status
    drift_detected = drift_result['metrics'][0]['result']['drift_detected']
    print("Drift Detected:", drift_detected)
    
    # Add the drift_detected value only to the current data window
    current['Evidently_Drift_Detect'] = drift_detected
    
    # Merge the drift result back into the original group (result_report)
    updated_group_data = group_data.merge(current[['Group_ID', 'TimeGroup', 'Evidently_Drift_Detect']], 
                                          on=['Group_ID', 'TimeGroup'], 
                                          how='left')
    
    # Append the updated group to the results list
    updated_result_report.append(updated_group_data)

# Combine all the groups back into a single DataFrame
result_report_with_drift = pd.concat(updated_result_report, ignore_index=True)

# Print or return the result
#result_report_with_drift = result_report_with_drift.drop(columns={"Evidently_Drift_Detect_x", "Evidently_Drift_Detect_y"})
result_report_with_drift = result_report_with_drift.drop_duplicates()

print(result_report_with_drift)

result_report_with_drift.to_json("result_report.json")