# ----------------------------------------------------------------------------------------------------------
# Imports
# ----------------------------------------------------------------------------------------------------------

import pandas as pd
import datetime

# ----------------------------------------------------------------------------------------------------------
# Combine data
# ----------------------------------------------------------------------------------------------------------

hr_df = pd.read_csv("tracker_data\HEARTRATE_AUTO\HEARTRATE_AUTO_1698575169318.csv")
hr_df.rename(columns={"heartRate": "bpm"}, inplace=True)

sleep_df = pd.read_csv("tracker_data\SLEEP\SLEEP_1698575169193.csv")

sleep_df["start"] = pd.to_datetime(sleep_df["start"])
sleep_df["stop"] = pd.to_datetime(sleep_df["stop"])
sleep_df["duration"] = sleep_df["stop"] - sleep_df["start"]

sleep_df["naps"] = pd.to_datetime(sleep_df["naps_end"]) - pd.to_datetime(
    sleep_df["naps_start"]
)
sleep_df["duration"] = sleep_df["duration"].dt.total_seconds() / 60
sleep_df["naps"] = sleep_df["naps"].dt.total_seconds().fillna(0)

sleep_df.drop(columns=["naps_start", "naps_end"], inplace=True)

merged_df = pd.merge(hr_df, sleep_df, on="date", how="inner")

df = merged_df.copy()
df.info()

# Convert the time columns to a datatime object
df["datetime"] = pd.to_datetime(df["date"] + " " + df["time"])

# Add a new column 'label' based on the date
df["label"] = df["datetime"].apply(
    lambda x: 0
    if (x < pd.Timestamp("2023-10-02")) or (x >= pd.Timestamp("2023-10-16"))
    else 1
)

# ----------------------------------------------------------------------------------------------------------
# Extract datetime features
# ----------------------------------------------------------------------------------------------------------

df["day"] = df["datetime"].dt.day
df["month"] = df["datetime"].dt.month
df["hour"] = df["datetime"].dt.hour
df["week"] = df["datetime"].dt.isocalendar().week
df["weekday"] = df["datetime"].dt.weekday
df["time"] = df["datetime"].dt.time

# ----------------------------------------------------------------------------------------------------------
# Convert to timeseries dataframes
# ----------------------------------------------------------------------------------------------------------

df.index = df["datetime"]
df = df[df["day"] != 23]
df.drop(columns=["datetime", "date", "time"], inplace=True)

# ----------------------------------------------------------------------------------------------------------
# Export dataset
# ----------------------------------------------------------------------------------------------------------

df.to_pickle("data/sleep_data.pkl")
