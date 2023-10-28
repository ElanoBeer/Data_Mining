import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

df = pd.read_pickle("data/all_features_df.pkl")
df.info()
df = df.drop(columns=["day", "month", "week", "lda_1"])

X_train, X_test, y_train, y_test = train_test_split(
    df.drop(columns="label"),
    df["label"],
    test_size=0.2,
    random_state=4,
    stratify=df["label"],
)

rf = RandomForestClassifier().fit(X_train, y_train)
y_pred = rf.predict(X_test)
print(classification_report(y_test, y_pred))

forest_importances = pd.Series(rf.feature_importances_, index=X_train.columns)

fig, ax = plt.subplots()
forest_importances.plot.bar(ax=ax)
ax.set_title("Feature importances using MDI")
ax.set_ylabel("Mean decrease in impurity")
fig.tight_layout()

X = df.drop(columns="label")
y = df["label"]

ts_cv = TimeSeriesSplit(
    n_splits=5,
    gap=0,
    max_train_size=10000,
    test_size=1000,
)

all_splits = list(ts_cv.split(X, y))
train_0, test_0 = all_splits[0]
X.iloc[train_0]
