import json
from datetime import datetime, timedelta
import random
import pytz

# Function to generate timestamps with timezone
def generate_timestamps(start_time, num_entries):
    return [start_time + timedelta(seconds=i * 60) for i in range(num_entries)]

# Function to generate mock data for a station
def generate_station_data(station_name, serial_number, drift_type, num_entries=100):
    start_time = datetime(2024, 10, 23, 17, 19, 37, 111000, tzinfo=pytz.timezone('Europe/Berlin'))
    timestamps = generate_timestamps(start_time, num_entries)
    
    data = []
    
    for i, timestamp in enumerate(timestamps):
        if drift_type == 'no_drift':
            value = 1.0  # No drift
        elif drift_type == 'small_drift':
            value = 1.0 + random.uniform(0, 0.1) * (i // (num_entries // 10))  # Small drift
        elif drift_type == 'large_drift':
            value = 1.0 + random.uniform(0, 1) * (i // (num_entries // 10))  # Large drift
        
        data.append({
            "StationGroup": "Group1",
            "StationName": station_name,
            "SerialNumber": serial_number,
            "Value": round(value, 2),  # Round to 2 decimal places
            "TimeStamp": timestamp.isoformat()  # Include timezone in the ISO format
        })
    
    return data

# Generate datasets for all stations
all_data = []
all_data.extend(generate_station_data("Station1", "1", "no_drift"))
all_data.extend(generate_station_data("Station2", "2", "small_drift"))
all_data.extend(generate_station_data("Station3", "3", "large_drift"))

# Save the generated data to a JSON file
with open('mock_station_data.json', 'w') as json_file:
    json.dump(all_data, json_file, indent=4)

print("Mock station data generated and saved to 'mock_station_data.json'.")
