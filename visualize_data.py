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

plt.style.use("fivethirtyeight")
mpl.rcParams["figure.dpi"] = 100
plt.rcParams["figure.figsize"] = (20, 5)

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

# ----------------------------------------------------------------------------------------------------------
# Plot BPM - pairs
# ----------------------------------------------------------------------------------------------------------

df = pd.read_pickle("data/all_features_df.pkl")
df.info()

control_df = df[df["label"] == 0].reset_index()
treatment_df = df[df["label"] == 1].reset_index()

control_df["time"] = control_df["datetime"].dt.time.astype(str)
treatment_df["time"] = treatment_df["datetime"].dt.time.astype(str)

control_lst = [25, 26, 27, 28, 29, 30, 1, 16, 17, 18, 19, 20, 21, 22]
treatment_lst = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]

for i in range(0, 14):
    fig, ax = plt.subplots()
    control_df[control_df["day"] == control_lst[i]][["bpm", "time"]].plot(
        x="time", ax=ax
    )
    treatment_df[treatment_df["day"] == treatment_lst[i]][["bpm", "time"]].plot(
        x="time", ax=ax
    )
    ax.set_title(f"Day {i}")

# ----------------------------------------------------------------------------------------------------------
# Plot sleep
# ----------------------------------------------------------------------------------------------------------

plot_df = df.copy().reset_index()
plot_df["time"] = plot_df["datetime"].dt.time.astype(str)
start_lst = sorted(list(set(plot_df["start"])))
duration_lst = list(plot_df["duration"].unique())
idx_lst = []

for i in range(len(start_lst)):
    idx_lst.append(plot_df.loc[plot_df["start"] == start_lst[i]].index[0])

for i in range(len(start_lst) - 1):
    plt.subplots()
    sleep_df = plot_df.loc[idx_lst[i] : idx_lst[i] + duration_lst[i]]
    if sleep_df["label"].mean() == 0:
        sleep_df["bpm"].plot()
    else:
        sleep_df["bpm"].plot(cmap="Set1")

# ----------------------------------------------------------------------------------------------------------
# Plot sleep - pairs
# ----------------------------------------------------------------------------------------------------------

insert_index = control_lst.index(1) + 1
control_lst[insert_index:insert_index] = treatment_lst
day_pairs = list(zip(control_lst, treatment_lst))
duration_lst.append(None)

start_dct = {
    key: [idx_lst[idx], duration_lst[idx]] for idx, key in enumerate(control_lst)
}

for i, j in day_pairs:
    fig, ax = plt.subplots()
    control_sleep_df = plot_df[plot_df["day"] == i]
    treatment_sleep_df = plot_df[plot_df["day"] == j]
    control_sleep_df = control_sleep_df.loc[
        start_dct[i][0] : start_dct[i][0] + int(start_dct[i][1])
    ]
    treatment_sleep_df = treatment_sleep_df.loc[
        start_dct[j][0] : start_dct[j][0] + int(start_dct[j][1])
    ]
    control_sleep_df[["bpm", "time"]].plot(x="time", ax=ax)
    treatment_sleep_df[["bpm", "time"]].plot(x="time", ax=ax)
