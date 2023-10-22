# ----------------------------------------------------------------------------------------------------------
# Imports
# ----------------------------------------------------------------------------------------------------------

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns

# ----------------------------------------------------------------------------------------------------------
# Load data
# ----------------------------------------------------------------------------------------------------------

df = pd.read_pickle("data/sleep_data.pkl")
df.info()

# ----------------------------------------------------------------------------------------------------------
# Adjust plot settings
# ----------------------------------------------------------------------------------------------------------

mpl.style.use("seaborn-v0_8-deep")
mpl.rcParams["figure.dpi"] = 100

# ----------------------------------------------------------------------------------------------------------
# Plot single columns
# ----------------------------------------------------------------------------------------------------------

day_df = df[df["day"] == 6]
plt.plot(day_df["bpm"])

# ----------------------------------------------------------------------------------------------------------
# Plot multiple columns
# ----------------------------------------------------------------------------------------------------------

# Sleep stages
df.drop(columns=["week", "month", "day", "label", "duration"]).groupby(
    ["weekday"]
).mean().plot(kind="bar")
df.groupby("week")["bpm"].plot(figsize=(20, 5))

# Duration
fig, ax = plt.subplots(nrows=1, ncols=2, sharex=False)
df.groupby(["week"])["duration"].mean().plot(kind="bar", ax=ax[0])
df.groupby(["weekday"])["duration"].mean().plot(kind="bar", ax=ax[1])
plt.tight_layout()

# Correlation matrix
fig, ax = plt.subplots()
correlation_matrix = df.drop(columns=["day", "month", "week"]).corr()
sns.heatmap(correlation_matrix, annot=True)
