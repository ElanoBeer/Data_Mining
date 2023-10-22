# ----------------------------------------------------------------------------------------------------------
# Imports
# ----------------------------------------------------------------------------------------------------------

import pandas as pd
import matplotlib.pyplot as plt
import math
import numpy as np
import scipy
from sklearn.neighbors import LocalOutlierFactor
from sklearn.cluster import DBSCAN, HDBSCAN

# ----------------------------------------------------------------------------------------------------------
# Load data and functions
# ----------------------------------------------------------------------------------------------------------

df = pd.read_pickle("data/sleep_data.pkl")
df.info()


def plot_binary_outliers(dataset, col, outlier_col, reset_index):
    """Plot outliers in case of a binary outlier score. Here, the col specifies the real data
    column and outlier_col the columns with a binary value (outlier or not).

    Args:
        dataset (pd.DataFrame): The dataset
        col (string): Column that you want to plot
        outlier_col (string): Outlier column marked with true/false
        reset_index (bool): whether to reset the index for plotting
    """

    # Taken from: https://github.com/mhoogen/ML4QS/blob/master/Python3Code/util/VisualizeDataset.py

    dataset = dataset.dropna(axis=0, subset=[col, outlier_col])
    dataset[outlier_col] = dataset[outlier_col].astype("bool")

    if reset_index:
        dataset = dataset.reset_index()

    fig, ax = plt.subplots()

    plt.xlabel("samples")
    plt.ylabel("value")

    # Plot non outliers in default color
    ax.plot(
        dataset.index[~dataset[outlier_col]],
        dataset[col][~dataset[outlier_col]],
        "+",
    )
    # Plot data points that are outliers in red
    ax.plot(
        dataset.index[dataset[outlier_col]],
        dataset[col][dataset[outlier_col]],
        "r+",
    )

    plt.legend(
        ["outlier " + col, "no outlier " + col],
        loc="upper center",
        ncol=2,
        fancybox=True,
        shadow=True,
    )
    plt.show()


def mark_outliers_iqr(dataset, col):
    """Function to mark values as outliers using the IQR method.

    Args:
        dataset (pd.DataFrame): The dataset
        col (string): The column you want apply outlier detection to

    Returns:
        pd.DataFrame: The original dataframe with an extra boolean column
        indicating whether the value is an outlier or not.
    """

    dataset = dataset.copy()

    Q1 = dataset[col].quantile(0.25)
    Q3 = dataset[col].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    dataset[col + "_outlier"] = (dataset[col] < lower_bound) | (
        dataset[col] > upper_bound
    )

    return dataset


def mark_outliers_chauvenet(dataset, col, C=2):
    """Finds outliers in the specified column of datatable and adds a binary column with
    the same name extended with '_outlier' that expresses the result per data point.

    Taken from: https://github.com/mhoogen/ML4QS/blob/master/Python3Code/Chapter3/OutlierDetection.py

    Args:
        dataset (pd.DataFrame): The dataset
        col (string): The column you want apply outlier detection to
        C (int, optional): Degree of certainty for the identification of outliers given the assumption
                           of a normal distribution, typicaly between 1 - 10. Defaults to 2.

    Returns:
        pd.DataFrame: The original dataframe with an extra boolean column
        indicating whether the value is an outlier or not.
    """

    dataset = dataset.copy()
    # Compute the mean and standard deviation.
    mean = dataset[col].mean()
    std = dataset[col].std()
    N = len(dataset.index)
    criterion = 1.0 / (C * N)

    # Consider the deviation for the data points.
    deviation = abs(dataset[col] - mean) / std

    # Express the upper and lower bounds.
    low = -deviation / math.sqrt(C)
    high = deviation / math.sqrt(C)
    prob = []
    mask = []

    # Pass all rows in the dataset.
    for i in range(0, len(dataset.index)):
        # Determine the probability of observing the point
        prob.append(
            1.0 - 0.5 * (scipy.special.erf(high[i]) - scipy.special.erf(low[i]))
        )
        # And mark as an outlier when the probability is below our criterion.
        mask.append(prob[i] < criterion)
    dataset[col + "_outlier"] = mask
    return dataset


def mark_outliers_mhd(dataset, columns, threshold):
    """
    Detect outliers in a DataFrame using Mahalanobis distance.

    Parameters:
        data (pd.DataFrame): Input DataFrame.
        columns (list): List of column names to consider for outlier detection.
        threshold (float): Mahalanobis distance threshold.

    Returns:
        pd.DataFrame: DataFrame with an additional column "outlier_mhd".
    """
    # Calculate the mean and covariance matrix of the specified columns
    mean = dataset[columns].mean().values
    cov = dataset[columns].cov().values

    # Calculate Mahalanobis distance for each data point
    mahalanobis_distances = []
    for i in range(len(dataset)):
        dist = scipy.spatial.distance.mahalanobis(
            dataset[columns].iloc[i].values, mean, np.linalg.inv(cov)
        )
        mahalanobis_distances.append(dist)

    # Create a new column "outlier_mhd" indicating whether each row is an outlier
    dataset["outlier_mhd"] = np.array(mahalanobis_distances) > threshold

    return dataset, mahalanobis_distances


def mark_outliers_lof(dataset, columns, n=20):
    """Mark values as outliers using LOF

    Args:
        dataset (pd.DataFrame): The dataset
        col (string): The column you want apply outlier detection to
        n (int, optional): n_neighbors. Defaults to 20.

    Returns:
        pd.DataFrame: The original dataframe with an extra boolean column
        indicating whether the value is an outlier or not.
    """

    dataset = dataset.copy()

    lof = LocalOutlierFactor(n_neighbors=n)
    data = dataset[columns]
    outliers = lof.fit_predict(data)
    X_scores = lof.negative_outlier_factor_

    dataset["outlier_lof"] = outliers == -1
    return dataset, outliers, X_scores


# ----------------------------------------------------------------------------------------------------------
# Plotting outliers (IQR) - Boxplots
# ----------------------------------------------------------------------------------------------------------

plt.style.use("fivethirtyeight")
plt.rcParams["figure.figsize"] = (20, 5)
plt.rcParams["figure.dpi"] = 100

# Plot a single column
outlier_column = "bpm"
df[[outlier_column, "label"]].boxplot(by="label", figsize=(20, 10))

# Plot all outliers columns
outlier_columns = ["bpm", "deepSleepTime", "shallowSleepTime", "wakeTime", "REMTime"]
df[outlier_columns + ["label"]].boxplot(by="label", figsize=(20, 10), layout=(1, 5))

# ----------------------------------------------------------------------------------------------------------
# Plotting outliers (IQR) - Timeseries
# ----------------------------------------------------------------------------------------------------------

# Plot a single column
col = "bpm"
df = mark_outliers_iqr(df, col)
plot_binary_outliers(
    dataset=df, col=col, outlier_col=col + "_outlier", reset_index=True
)

# ----------------------------------------------------------------------------------------------------------
# Distribution-based outlier detection
# ----------------------------------------------------------------------------------------------------------

# Evaluate normal distribution
for col in outlier_columns:
    df[[col, "label"]].plot.hist(by="label", figsize=(10, 7))

# Chauvenet Criterion
normal_outlier_columns = "bpm"
df = mark_outliers_chauvenet(df, normal_outlier_columns)
plot_binary_outliers(
    dataset=df,
    col=normal_outlier_columns,
    outlier_col=normal_outlier_columns + "_outlier",
    reset_index=True,
)

# ----------------------------------------------------------------------------------------------------------
# Distance-based outlier detection
# ----------------------------------------------------------------------------------------------------------

# Local outlier factor
df, outliers, X_scores = mark_outliers_lof(df, outlier_columns)
plot_binary_outliers(dataset=df, col=col, outlier_col="outlier_lof", reset_index=True)

# Mahalanobis distance
df, distances = mark_outliers_mhd(dataset=df, columns=outlier_columns, threshold=3.5)
plot_binary_outliers(dataset=df, col=col, outlier_col="outlier_mhd", reset_index=True)

# ----------------------------------------------------------------------------------------------------------
# Density-based outlier detection
# ----------------------------------------------------------------------------------------------------------

# Cluster outliers using density-based clustering
dbscan = DBSCAN(eps=1, min_samples=5)
df["outlier_dbscan"] = dbscan.fit_predict(df) == -1
plot_binary_outliers(
    dataset=df, col=col, outlier_col="outlier_dbscan", reset_index=True
)

# Cluster outliers using hierarchical density-based clustering
hdbscan = HDBSCAN(min_cluster_size=10, min_samples=5)
df["outlier_hdbscan"] = hdbscan.fit_predict(df) == -1
plot_binary_outliers(
    dataset=df, col=col, outlier_col="outlier_hdbscan", reset_index=True
)
