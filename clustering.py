# ----------------------------------------------------------------------------------------------------------
# Imports
# ----------------------------------------------------------------------------------------------------------

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from sklearn.manifold import MDS, TSNE
from sklearn.metrics import pairwise_distances
from sklearn.cluster import KMeans

df = pd.read_pickle("data/sleep_data.pkl")
df.info()
plt.style.use("fivethirtyeight")
plt.rcParams["figure.figsize"] = (20, 5)
plt.rcParams["figure.dpi"] = 100

# ----------------------------------------------------------------------------------------------------------
# Sleep distribution distances
# ----------------------------------------------------------------------------------------------------------

stages_cols = [
    "shallowSleepTime",
    "deepSleepTime",
    "REMTime",
    "wakeTime",
    "naps",
    "day",
    "duration",
    "label",
]
sleepstage_df = df[stages_cols]


class Clustering:
    def __init__(self, df):
        self.df = df

    def _calculate_distances(self, div_df, metric):
        try:
            distances = pd.DataFrame(pairwise_distances(div_df, metric=metric))
            return distances
        except Exception as e:
            print(f"Error calculating distances with {metric}: {e}")
            return None

    def plot_scaled_distances(self):
        stages_cols = [
            "shallowSleepTime",
            "deepSleepTime",
            "REMTime",
            "wakeTime",
            "duration",
            "label",
        ]
        df = self.df.groupby("day")[stages_cols].min()

        div_df = df[["shallowSleepTime", "deepSleepTime", "REMTime", "wakeTime"]].div(
            df["duration"], axis="index"
        )
        pw_distances = [
            "cityblock",
            "cosine",
            "euclidean",
            "l1",
            "l2",
            "manhattan",
            "braycurtis",
            "canberra",
            "chebyshev",
            "correlation",
            "hamming",
            "minkowski",
            "seuclidean",
            "sqeuclidean",
        ]

        for metric in pw_distances:
            distances = self._calculate_distances(div_df, metric)
            if distances is not None:
                mds = MDS(
                    n_components=2,
                    dissimilarity="precomputed",
                    normalized_stress=False,
                    random_state=4,
                )
                scaled_features = mds.fit_transform(distances)
                scaled_df = pd.DataFrame(
                    {
                        "x": scaled_features[:, 0],
                        "y": scaled_features[:, 1],
                        "label": df["label"],
                    }
                )
                if mds.stress_ < 1:
                    plt.figure(figsize=(10, 8))
                    for label in scaled_df["label"].unique():
                        data = scaled_df[scaled_df["label"] == label]
                        plt.scatter(data["x"], data["y"], label=str(label))
                    plt.title(f"MDS Plot with {metric} Distance Metric")
                    plt.show()

    def plot_tsne(self):
        stages_cols = [
            "shallowSleepTime",
            "deepSleepTime",
            "REMTime",
            "wakeTime",
            "duration",
            "label",
        ]
        df = self.df.groupby("day")[stages_cols].min()

        div_df = df[["shallowSleepTime", "deepSleepTime", "REMTime", "wakeTime"]].div(
            df["duration"], axis="index"
        )

        tsne = TSNE(
            n_components=2,
            perplexity=3,
            learning_rate=100,
            n_iter=1000,
            verbose=1,
            random_state=0,
            angle=0.75,
        )
        tsne_features = tsne.fit_transform(div_df)
        tsne_df = pd.DataFrame(tsne_features, columns=["TSNE1", "TSNE2"])
        tsne_df["label"] = df["label"].reset_index()["label"]

        plt.figure(figsize=(10, 8))
        for label in tsne_df["label"].unique():
            data = tsne_df[tsne_df["label"] == label]
            plt.scatter(data["TSNE1"], data["TSNE2"], label=str(label))
        plt.title("T-SNE Plot")
        plt.xlabel("TSNE1")
        plt.ylabel("TSNE2")
        plt.legend()
        plt.show()

    def plot_kmeans(self):
        stages_cols = [
            "shallowSleepTime",
            "deepSleepTime",
            "REMTime",
            "wakeTime",
            "duration",
            "label",
        ]
        df = self.df.groupby("day")[stages_cols].min()

        div_df = df[["shallowSleepTime", "deepSleepTime", "REMTime", "wakeTime"]].div(
            df["duration"], axis="index"
        )

        kmeans = KMeans(n_clusters=2, init="k-means++")
        scaled_features = kmeans.fit_transform(div_df)
        cluster_df = pd.DataFrame(
            {
                "x": scaled_features[:, 0],
                "y": scaled_features[:, 1],
                "label": df["label"],
            }
        )

        plt.figure(figsize=(10, 8))
        for label in cluster_df["label"].unique():
            data = cluster_df[cluster_df["label"] == label]
            plt.scatter(data["x"], data["y"], label=str(label))
        plt.title("KMeans Plot")
        plt.xlabel("Dim1")
        plt.ylabel("Dim2")
        plt.legend()
        plt.show()


# Assuming sleepstage_df is already defined
clustering = Clustering(sleepstage_df)
clustering.plot_scaled_distances()
clustering.plot_tsne()
clustering.plot_kmeans()
