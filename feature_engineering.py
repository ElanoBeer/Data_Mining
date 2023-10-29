# ----------------------------------------------------------------------------------------------------------
# Imports
# ----------------------------------------------------------------------------------------------------------

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from remote import DataTransformation, TemporalAbstraction, FrequencyAbstraction
from tslearn.clustering import TimeSeriesKMeans
from sklearn.cluster import HDBSCAN
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from scipy.interpolate import interp1d

# ----------------------------------------------------------------------------------------------------------
# Load data and functions
# ----------------------------------------------------------------------------------------------------------

df = pd.read_pickle("data/sleep_data.pkl")
df.info()
plt.style.use("fivethirtyeight")
plt.rcParams["figure.figsize"] = (20, 5)
plt.rcParams["figure.dpi"] = 100

# ----------------------------------------------------------------------------------------------------------
# Butterworth LowPassFilter
# ----------------------------------------------------------------------------------------------------------

lowpass_df = df.copy()
LowPass = DataTransformation.LowPassFilter()

fs = 1 / 60
cutoff = fs / 12

lowpass_df = LowPass.low_pass_filter(lowpass_df, "bpm", fs, cutoff, order=5)

fig, ax = plt.subplots(nrows=2, sharex=True, figsize=(20, 10))
ax[0].plot(lowpass_df["bpm"].reset_index(drop=True), label="raw data")
ax[1].plot(lowpass_df["bpm_lowpass"].reset_index(drop=True), label="butterworth filter")
ax[0].legend(), ax[1].legend()

subset = lowpass_df[lowpass_df["day"] == 5]
fig, ax = plt.subplots(nrows=2, sharex=True, figsize=(20, 10))
ax[0].plot(subset["bpm"].reset_index(drop=True), label="raw data")
ax[1].plot(subset["bpm_lowpass"].reset_index(drop=True), label="butterworth filter")
ax[0].legend(), ax[1].legend()

# Replace the lowpass filter bpm for bpm
lowpass_df["bpm"] = lowpass_df["bpm_lowpass"]
lowpass_df.drop(columns=["bpm_lowpass"], inplace=True)

# ----------------------------------------------------------------------------------------------------------
# BPM other transformations
# ----------------------------------------------------------------------------------------------------------

transform_df = lowpass_df.copy()

# perform polynomial smoothing
coefficients = np.polyfit(range(len(transform_df[["bpm"]])), transform_df["bpm"], 5)
transform_df["bpm_poly"] = np.poly1d(coefficients)(range(len(transform_df["bpm"])))
transform_df["bpm_poly"].plot()

# Calculate the time difference between consecutive time points
transform_df = transform_df.reset_index()
time_diff = transform_df["datetime"].diff().dt.total_seconds().fillna(0)

# Calculate the BPM derivative (rate of change)
transform_df["bpm_derivative"] = (
    transform_df["bpm"].diff().div(time_diff.where(time_diff != 0)).fillna(0)
)

# ----------------------------------------------------------------------------------------------------------
# Temporal abstraction
# ----------------------------------------------------------------------------------------------------------

temporal_df = lowpass_df.copy()
NumAbs = TemporalAbstraction.NumericalAbstraction()
ws = 5

# Statistical features
stat_lst = ["mean", "std", "min", "max", "median"]
for stat in stat_lst:
    temporal_df = NumAbs.abstract_numerical(temporal_df, ["bpm"], ws, stat)

# Lagged features
temporal_df["bpm_lag_1"] = temporal_df["bpm"].shift(1)
temporal_df["bpm_lag_5"] = temporal_df["bpm"].shift(5)

# ----------------------------------------------------------------------------------------------------------
# Frequency abstraction
# ----------------------------------------------------------------------------------------------------------

frequency_df = temporal_df.copy().reset_index()
FreqAbs = FrequencyAbstraction.FourierTransformation()

fs = 1 / 60
ws = 60

frequency_df = FreqAbs.abstract_frequency(frequency_df, ["bpm"], ws, fs)

subset = frequency_df[frequency_df["day"] == 23]
subset[["bpm"]].plot()
subset[["bpm_freq_weighted", "bpm_pse"]].plot()
subset[["bpm_max_freq"]].plot()

frequency_df.drop(columns="bpm_max_freq")
frequency_df = frequency_df.dropna()
frequency_df = frequency_df.set_index("datetime", drop=True)

# ----------------------------------------------------------------------------------------------------------
# PCA abstraction (Elbow method)
# ----------------------------------------------------------------------------------------------------------

pca_df = frequency_df.copy()
PCA = DataTransformation.PrincipalComponentAnalysis()

# Disclaimer: Theoratically should only work on continuous variables
pred_columns = [
    "deepSleepTime",
    "shallowSleepTime",
    "wakeTime",
    "REMTime",
    "duration",
    "day",
    "month",
    "week",
    "weekday",
    "bpm",
]

explained_variance = PCA.determine_pc_explained_variance(pca_df, pred_columns)
plt.figure(figsize=(10, 10))
plt.plot(range(1, len(pred_columns) + 1), explained_variance)
plt.xlabel("Principal components")
plt.ylabel("Explained variance")
plt.show()

pca_df = PCA.apply_pca(pca_df, pred_columns, 4)
subset = pca_df[pca_df["day"] == 5]
subset[subset.columns[-4:]].plot()

# ----------------------------------------------------------------------------------------------------------
# LDA abstraction
# ----------------------------------------------------------------------------------------------------------

lda_df = pca_df.copy()

# Disclaimer: Theoratically should only work on continuous variables
pred_columns = [
    "deepSleepTime",
    "shallowSleepTime",
    "wakeTime",
    "REMTime",
    "duration",
    "day",
    "month",
    "week",
    "weekday",
    "bpm",
]

scaled_df = StandardScaler().fit_transform(lda_df[pred_columns], lda_df["label"])
lda_df["lda_1"] = LinearDiscriminantAnalysis(n_components=1).fit_transform(
    scaled_df, lda_df["label"]
)

# ----------------------------------------------------------------------------------------------------------
# Clustering Abstraction
# ----------------------------------------------------------------------------------------------------------


def cluster_plot(col):
    plt.figure(figsize=(20, 5))
    plt.scatter(
        cluster_df.index,
        cluster_df["bpm"],
        c=cluster_df[col],
        cmap="viridis",
        s=50,
    )
    plt.xlabel("Datetime Index")
    plt.ylabel("BPM")
    plt.title("Scatter Plot of BPM vs Datetime with Cluster Labels")
    plt.colorbar(label="Cluster Label")
    plt.show()


cluster_df = lda_df.copy()

k_clusters = range(2, 6)
inertias = []
cluster_columns = cluster_df.columns.to_list()
cluster_columns.remove("label")

for k in k_clusters:
    kmeans = TimeSeriesKMeans(n_clusters=k, init="k-means++", n_init=2, random_state=5)
    cluster_labels = kmeans.fit_predict(cluster_df[cluster_columns])
    inertias.append(kmeans.inertia_)

plt.figure(figsize=(10, 10))
plt.plot(k_clusters, inertias)
plt.xlabel("K clusters")
plt.ylabel("Inertia")
plt.show()

kmeans = TimeSeriesKMeans(
    n_clusters=4, init="k-means++", n_init=2, metric="dtw", random_state=5
)
cluster_df["kmeans_cluster"] = kmeans.fit_predict(cluster_df[cluster_columns])


subset = cluster_df[cluster_df["day"] == 5]
cluster_plot(col="kmeans_cluster")

# Cluster outliers using hierarchical density-based clustering
hdbscan = HDBSCAN(min_cluster_size=20, min_samples=5)
cluster_df["hdbscan_cluster"] = hdbscan.fit_predict(cluster_df[cluster_columns])
cluster_plot(col="hdbscan_cluster")

# Correlation matrix
fig, ax = plt.subplots(figsize=(30, 30))
correlation_matrix = cluster_df.corr()
sns.heatmap(correlation_matrix, annot=True)

# ----------------------------------------------------------------------------------------------------------
# Export the dataframe
# ----------------------------------------------------------------------------------------------------------

cluster_df.to_pickle("data/all_features_df.pkl")
