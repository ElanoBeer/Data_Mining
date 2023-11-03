import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import classification_report, mean_absolute_error
import matplotlib.pyplot as plt

df = pd.read_pickle("data/all_features_df.pkl")
df.info()
df = df.drop(columns=["day", "month", "week", "lda_1"])

# ----------------------------------------------------------------------------------------------------------
# Predict the existence of meditation
# ----------------------------------------------------------------------------------------------------------

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

# ----------------------------------------------------------------------------------------------------------
# Model the BPM
# ----------------------------------------------------------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    df.drop(columns="bpm"),
    df["bpm"],
    test_size=0.2,
    random_state=4,
    # stratify=df["bpm"],
)

rf = RandomForestRegressor().fit(X_train, y_train)
y_pred = rf.predict(X_test)
print(mean_absolute_error(y_test, y_pred))

forest_importances = pd.Series(rf.feature_importances_, index=X_train.columns)

fig, ax = plt.subplots()
forest_importances.plot.bar(ax=ax)
ax.set_title("Feature importances using MDI")
ax.set_ylabel("Mean decrease in impurity")
fig.tight_layout()

# ----------------------------------------------------------------------------------------------------------
# Timeseries cross-validation results
# ----------------------------------------------------------------------------------------------------------

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

# ----------------------------------------------------------------------------------------------------------
# Logistic Regression
# ----------------------------------------------------------------------------------------------------------

from sklearn.linear_model import LogisticRegression

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

model = LogisticRegression().fit(X_train, y_train)
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
print(model.coef_)

logistic_importances = pd.DataFrame(model.coef_, columns=X_train.columns).transpose()

fig, ax = plt.subplots(figsize=(20, 12))
logistic_importances.plot.bar(ax=ax)
ax.set_title("Feature coefficients using MDI")
ax.set_ylabel("Mean decrease in impurity")

# ----------------------------------------------------------------------------------------------------------
# Plot decision tree
# ----------------------------------------------------------------------------------------------------------

from sklearn import tree

estimator = rf.estimators_[2]
fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(20, 12), dpi=800)
tree.plot_tree(estimator, feature_names=X_train.columns, filled=True, rounded=True)
