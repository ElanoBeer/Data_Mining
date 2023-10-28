# ----------------------------------------------------------------------------------------------------------
# Imports
# ----------------------------------------------------------------------------------------------------------

import pandas as pd
import datetime

# ----------------------------------------------------------------------------------------------------------
# Combine data
# ----------------------------------------------------------------------------------------------------------

hr_df = pd.read_csv("tracker_data\HEARTRATE_AUTO\HEARTRATE_AUTO_1696925366934.csv")
hr_df.rename(columns={"heartRate": "bpm"}, inplace=True)

sleep_df = pd.read_csv("tracker_data\SLEEP\SLEEP_1696925366859.csv")
sleep_df["duration"] = pd.to_datetime(sleep_df["stop"]) - pd.to_datetime(
    sleep_df["start"]
)
sleep_df["duration"] = sleep_df["duration"].dt.total_seconds()

sleep_df.drop(columns=["naps", "start", "stop"], inplace=True)

merged_df = pd.merge(hr_df, sleep_df, on="date", how="inner")

df = merged_df.copy()
df.info()

# Convert the time columns to a datatime object
df["datetime"] = pd.to_datetime(df["date"] + " " + df["time"])

# Add a new column 'label' based on the date
df["label"] = df["datetime"].apply(lambda x: 0 if x < pd.Timestamp("2023-10-02") else 1)

# ----------------------------------------------------------------------------------------------------------
# Extract datetime features
# ----------------------------------------------------------------------------------------------------------

df["day"] = df["datetime"].dt.day
df["month"] = df["datetime"].dt.month
df["hour"] = df["datetime"].dt.hour
df["week"] = df["datetime"].dt.isocalendar().week
df["weekday"] = df["datetime"].dt.weekday

# ----------------------------------------------------------------------------------------------------------
# Convert to timeseries dataframes
# ----------------------------------------------------------------------------------------------------------

df.index = df["datetime"]
df.drop(columns=["datetime", "date", "time"], inplace=True)

# ----------------------------------------------------------------------------------------------------------
# Export dataset
# ----------------------------------------------------------------------------------------------------------

df.to_pickle("data/sleep_data.pkl")
