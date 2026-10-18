import pandas as pd

from sklearn.dummy import DummyClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    average_precision_score,
)

# Load the original dataset
df = pd.read_csv("../data/raw/creditcard.csv")

# Separate input features from the target
X = df.drop(columns=["Class"])
y = df["Class"]

# Split data while preserving the fraud proportion
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)

print("Training rows:", len(X_train))
print("Testing rows:", len(X_test))

print("\nFraud rate in training set:")
print(round(y_train.mean() * 100, 4), "%")

print("\nFraud rate in test set:")
print(round(y_test.mean() * 100, 4), "%")

# Baseline: predict every transaction as legitimate
baseline = DummyClassifier(strategy="most_frequent")
baseline.fit(X_train, y_train)

y_pred = baseline.predict(X_test)

print("\nBaseline confusion matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nBaseline classification report:")
print(classification_report(y_test, y_pred, zero_division=0))

# A constant classifier has no useful ranking ability.
# Average precision is not informative for this baseline.
print("\nBaseline fraud detections:", int(y_pred.sum()))

from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    average_precision_score,
)

# Train Isolation Forest without fraud labels
model = IsolationForest(
    n_estimators=200,
    contamination=0.001727,
    random_state=42,
    n_jobs=-1,
)

model.fit(X_train)

# Isolation Forest returns:
#  1  = inlier (not flagged)
# -1  = outlier (flagged)
raw_predictions = model.predict(X_test)

# Convert predictions to the same format as Class:
# 0 = not flagged, 1 = flagged
y_pred = (raw_predictions == -1).astype(int)

print("\nIsolation Forest confusion matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nIsolation Forest classification report:")
print(
    classification_report(
        y_test,
        y_pred,
        target_names=["Legitimate", "Fraud"],
        zero_division=0,
    )
)

print("\nFraud-labelled transactions:", int(y_test.sum()))
print("Transactions flagged:", int(y_pred.sum()))

# Evaluate ranking using the anomaly scores.
# Higher scores below mean more anomalous.
anomaly_scores = -model.score_samples(X_test)

print(
    "\nAverage precision:",
    round(average_precision_score(y_test, anomaly_scores), 5),
)


import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
)

# 1. Split the existing training data into a smaller fit set
#    and a validation set.
X_fit, X_val, y_fit, y_val = train_test_split(
    X_train,
    y_train,
    test_size=0.20,
    random_state=43,
    stratify=y_train,
)

print("Model-fitting rows:", len(X_fit))
print("Validation rows:", len(X_val))
print("Fraud-labelled validation rows:", int(y_val.sum()))

# 2. Train Isolation Forest without using fraud labels.
validation_model = IsolationForest(
    n_estimators=200,
    contamination="auto",
    random_state=42,
    n_jobs=-1,
)

validation_model.fit(X_fit)

# 3. Generate anomaly scores.
#    Higher scores mean more anomalous.
val_scores = -validation_model.score_samples(X_val)

print(
    "\nValidation average precision:",
    round(average_precision_score(y_val, val_scores), 5),
)

# 4. Compare different transaction-review capacities.
review_rates = [0.002, 0.005, 0.01]

print("\nThreshold comparison:")
print("-" * 90)

for review_rate in review_rates:
    # Select a score threshold that flags approximately
    # the highest-scoring 0.2%, 0.5%, or 1% of transactions.
    threshold = np.quantile(val_scores, 1 - review_rate)
    val_pred = (val_scores >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_val, val_pred, labels=[0, 1]
    ).ravel()

    print(f"\nReview capacity: {review_rate:.1%}")
    print("Transactions flagged:", int(val_pred.sum()))
    print("True positives (fraud-labelled caught):", int(tp))
    print("False positives (legitimate flagged):", int(fp))
    print("Missed fraud-labelled transactions:", int(fn))
    print("Precision:", round(precision_score(y_val, val_pred, zero_division=0), 4))
    print("Recall:", round(recall_score(y_val, val_pred, zero_division=0), 4))
    print("F1-score:", round(f1_score(y_val, val_pred, zero_division=0), 4))



from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score

# Train a supervised benchmark using the fraud labels.
supervised_model = make_pipeline(
    StandardScaler(),
    LogisticRegression(
        class_weight="balanced",
        max_iter=2000,
        random_state=42,
    ),
)

supervised_model.fit(X_fit, y_fit)

# Estimate the probability of the fraud-labelled class.
supervised_scores = supervised_model.predict_proba(X_val)[:, 1]

# Compare ranking quality on the same validation records.
if_ap = average_precision_score(y_val, val_scores)
lr_ap = average_precision_score(y_val, supervised_scores)

fraud_prevalence = y_val.mean()

print("\nModel comparison on validation data")
print("------------------------------------")
print("Fraud-label prevalence:", round(fraud_prevalence, 5))
print("Isolation Forest average precision:", round(if_ap, 5))
print("Logistic Regression average precision:", round(lr_ap, 5))

# Compare both models at the same review capacities.
for review_rate in [0.002, 0.005, 0.01]:
    print(f"\nReview capacity: {review_rate:.1%}")

    for name, scores in [
        ("Isolation Forest", val_scores),
        ("Logistic Regression", supervised_scores),
    ]:
        threshold = np.quantile(scores, 1 - review_rate)
        predictions = (scores >= threshold).astype(int)

        tp = int(((predictions == 1) & (y_val.to_numpy() == 1)).sum())
        fp = int(((predictions == 1) & (y_val.to_numpy() == 0)).sum())

        print(
            f"{name}: flagged={int(predictions.sum())}, "
            f"caught={tp}, false_positives={fp}, "
            f"precision={precision_score(y_val, predictions, zero_division=0):.4f}, "
            f"recall={recall_score(y_val, predictions, zero_division=0):.4f}"
        )



import pandas as pd

df = pd.read_csv("../data/raw/creditcard.csv")

feature_cols = df.columns.drop("Class")

# Create a fingerprint for each transaction's feature values.
# Class is deliberately excluded.
df["_feature_group"] = pd.util.hash_pandas_object(
    df[feature_cols],
    index=False,
).astype("uint64")

group_sizes = df["_feature_group"].value_counts()
repeated_groups = group_sizes[group_sizes > 1]

# Identify feature groups that have conflicting Class labels.
labels_per_group = df.groupby("_feature_group")["Class"].nunique()
conflicting_groups = labels_per_group[labels_per_group > 1]

print("Total transactions:", len(df))
print("Distinct feature groups:", df["_feature_group"].nunique())
print("Feature groups appearing more than once:", len(repeated_groups))
print(
    "Rows belonging to repeated feature groups:",
    int(df["_feature_group"].isin(repeated_groups.index).sum()),
)
print("Repeated feature groups with conflicting labels:", len(conflicting_groups))



import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

df = pd.read_csv("../data/raw/creditcard.csv")

X = df.drop(columns=["Class"])
y = df["Class"]

# Group identical feature rows together.
groups = pd.util.hash_pandas_object(
    X,
    index=False,
).astype("uint64")

# Five folds: approximately 60% train, 20% validation, 20% test.
splitter = StratifiedGroupKFold(
    n_splits=5,
    shuffle=True,
    random_state=42,
)

folds = list(splitter.split(X, y, groups=groups))

train_indices = []
validation_indices = None
test_indices = None

for fold_number, (train_idx, held_out_idx) in enumerate(folds):
    if fold_number < 3:
        train_indices.extend(train_idx)
    elif fold_number == 3:
        validation_indices = held_out_idx
    else:
        test_indices = held_out_idx

# The folds above overlap in their training portions, so use a simpler
# explicit assignment instead of combining fold training indices.
# Assign each row to exactly one fold.
fold_assignment = pd.Series(index=X.index, dtype="int64")

for fold_number, (_, held_out_idx) in enumerate(folds):
    fold_assignment.iloc[held_out_idx] = fold_number

train_indices = X.index[fold_assignment.isin([0, 1, 2])]
validation_indices = X.index[fold_assignment == 3]
test_indices = X.index[fold_assignment == 4]

X_train, y_train = X.loc[train_indices], y.loc[train_indices]
X_val, y_val = X.loc[validation_indices], y.loc[validation_indices]
X_test, y_test = X.loc[test_indices], y.loc[test_indices]

print("Split sizes:")
print("Train:", len(X_train))
print("Validation:", len(X_val))
print("Test:", len(X_test))

print("\nFraud-label counts:")
print("Train:", int(y_train.sum()))
print("Validation:", int(y_val.sum()))
print("Test:", int(y_test.sum()))

# Verify that no feature group crosses split boundaries.
train_groups = set(groups.loc[train_indices])
val_groups = set(groups.loc[validation_indices])
test_groups = set(groups.loc[test_indices])

print("\nGroup overlap checks:")
print("Train-validation:", len(train_groups & val_groups))
print("Train-test:", len(train_groups & test_groups))
print("Validation-test:", len(val_groups & test_groups))


import numpy as np

from sklearn.ensemble import IsolationForest
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
)

# -----------------------------
# 1. Train Isolation Forest
# -----------------------------
if_model = IsolationForest(
    n_estimators=200,
    contamination="auto",
    random_state=42,
    n_jobs=-1,
)

if_model.fit(X_train)

if_val_scores = -if_model.score_samples(X_val)
if_test_scores = -if_model.score_samples(X_test)

# -----------------------------
# 2. Train Logistic Regression
# -----------------------------
lr_model = make_pipeline(
    StandardScaler(),
    LogisticRegression(
        class_weight="balanced",
        max_iter=2000,
        random_state=42,
    ),
)

lr_model.fit(X_train, y_train)

lr_val_scores = lr_model.predict_proba(X_val)[:, 1]
lr_test_scores = lr_model.predict_proba(X_test)[:, 1]

# -----------------------------
# 3. Compare ranking quality
# -----------------------------
print("\nValidation average precision")
print("----------------------------")
print(
    "Isolation Forest:",
    round(average_precision_score(y_val, if_val_scores), 5),
)
print(
    "Logistic Regression:",
    round(average_precision_score(y_val, lr_val_scores), 5),
)

# -----------------------------
# 4. Compare review capacities
# -----------------------------
def evaluate_capacity(y_true, scores, review_rate):
    threshold = np.quantile(scores, 1 - review_rate)
    predictions = (scores >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true, predictions, labels=[0, 1]
    ).ravel()

    return {
        "flagged": int(predictions.sum()),
        "caught": int(tp),
        "false_positives": int(fp),
        "missed": int(fn),
        "precision": precision_score(
            y_true, predictions, zero_division=0
        ),
        "recall": recall_score(
            y_true, predictions, zero_division=0
        ),
        "f1": f1_score(
            y_true, predictions, zero_division=0
        ),
    }


for rate in [0.002, 0.005, 0.01]:
    print(f"\nReview capacity: {rate:.1%}")

    for name, scores in [
        ("Isolation Forest", if_val_scores),
        ("Logistic Regression", lr_val_scores),
    ]:
        result = evaluate_capacity(y_val, scores, rate)
        print(
            f"{name}: "
            f"flagged={result['flagged']}, "
            f"caught={result['caught']}, "
            f"false_positives={result['false_positives']}, "
            f"precision={result['precision']:.4f}, "
            f"recall={result['recall']:.4f}, "
            f"f1={result['f1']:.4f}"
        )

# -----------------------------
# Final evaluation on held-out test data
# -----------------------------

# Choose the review threshold from validation data only.
review_rate = 0.002  # 0.2%

if_threshold = np.quantile(if_val_scores, 1 - review_rate)
lr_threshold = np.quantile(lr_val_scores, 1 - review_rate)

# Apply the fixed validation thresholds to test scores.
if_test_pred = (if_test_scores >= if_threshold).astype(int)
lr_test_pred = (lr_test_scores >= lr_threshold).astype(int)

print("\nFINAL TEST RESULTS")
print("==================")

print("\nTest-set average precision")
print(
    "Isolation Forest:",
    round(average_precision_score(y_test, if_test_scores), 5),
)
print(
    "Logistic Regression:",
    round(average_precision_score(y_test, lr_test_scores), 5),
)


def report_test_results(name, y_true, predictions):
    tn, fp, fn, tp = confusion_matrix(
        y_true, predictions, labels=[0, 1]
    ).ravel()

    print(f"\n{name}")
    print("-" * len(name))
    print("Transactions flagged:", int(predictions.sum()))
    print("Fraud-labelled transactions caught:", int(tp))
    print("Legitimate transactions flagged:", int(fp))
    print("Fraud-labelled transactions missed:", int(fn))
    print(
        "Precision:",
        round(precision_score(y_true, predictions, zero_division=0), 4),
    )
    print(
        "Recall:",
        round(recall_score(y_true, predictions, zero_division=0), 4),
    )
    print(
        "F1-score:",
        round(f1_score(y_true, predictions, zero_division=0), 4),
    )


report_test_results(
    "Isolation Forest at 0.2% validation review rate",
    y_test,
    if_test_pred,
)

report_test_results(
    "Logistic Regression at 0.2% validation review rate",
    y_test,
    lr_test_pred,
)

