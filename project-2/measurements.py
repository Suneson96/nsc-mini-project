import pandas as pd
import os

MEASUREMENTS_FILE = "csv/measurements.csv"

def load_measurements(file_path: str = MEASUREMENTS_FILE) -> pd.DataFrame:
    if not os.path.exists(file_path):

        # If the file doesn't exist, return an empty DataFrame with approach and time_seconds columns.
        return pd.DataFrame(columns=["approach", "time_seconds"])
    return pd.read_csv(file_path)

def save_measurements(approach: str, time_seconds: float, file_path: str = MEASUREMENTS_FILE) -> None:

    # Ensure the directory exists before saving the file.
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    # Add measurements to CSV (overwriting any existing data for the same approach).
    measurements = load_measurements(file_path)
    measurements = measurements[measurements["approach"] != approach]  # Remove existing entry for the approach if it exists.
    new_measurement = pd.DataFrame({
        "approach": [approach],
        "time_seconds": [time_seconds]
    })
    measurements = pd.concat([measurements, new_measurement], ignore_index=True)
    measurements.to_csv(file_path, index=False)