
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.dummy import DummyClassifier
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# data loading
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "creditcard.csv"

df = pd.read_csv(DATA_PATH)

X = df.drop(columns=["Class"])
y = df["Class"]

print("Dataset rows:", len(df))
print("Fraud-labelled transactions:", int(y.sum()))
print(f"Fraud-label prevalence: {y.mean():.5%}")

# Group-aware train/validation/test split
groups = pd.util.hash_pandas_object(
    X,
    index=False,
).astype("uint64")

splitter = StratifiedGroupKFold(
    n_splits=5,
    shuffle=True,
    random_state=42,
)

fold_assignment = pd.Series(
    index=X.index,
    dtype="int8",
)

for fold_number, (_, held_out_indices) in enumerate(
    splitter.split(X, y, groups=groups)
):
    fold_assignment.iloc[held_out_indices] = fold_number

train_indices = X.index[fold_assignment.isin([0, 1, 2])]
validation_indices = X.index[fold_assignment == 3]
test_indices = X.index[fold_assignment == 4]

X_train = X.loc[train_indices]
y_train = y.loc[train_indices]

X_val = X.loc[validation_indices]
y_val = y.loc[validation_indices]

X_test = X.loc[test_indices]
y_test = y.loc[test_indices]

print("\nSplit sizes")
print("-----------")
for name, features, labels in [
    ("Train", X_train, y_train),
    ("Validation", X_val, y_val),
    ("Test", X_test, y_test),
]:
    print(
        f"{name}: rows={len(features):,}, "
        f"fraud-labelled={int(labels.sum())}, "
        f"prevalence={labels.mean():.5%}"
    )

train_groups = set(groups.loc[train_indices])
val_groups = set(groups.loc[validation_indices])
test_groups = set(groups.loc[test_indices])

overlaps = {
    "Train-validation": len(train_groups & val_groups),
    "Train-test": len(train_groups & test_groups),
    "Validation-test": len(val_groups & test_groups),
}

print("\nGroup overlap checks")
print("--------------------")

for name, overlap_count in overlaps.items():
    print(f"{name}: {overlap_count}")

assert all(count == 0 for count in overlaps.values()), (
    "Identical feature groups cross split boundaries."
)


# Dummy baseline
baseline = DummyClassifier(strategy="most_frequent")
baseline.fit(X_train, y_train)

baseline_predictions = baseline.predict(X_test)

print("\nDummy baseline")
print("--------------")
print("Transactions flagged:", int(baseline_predictions.sum()))
print("Fraud-labelled transactions caught:", int(
    ((baseline_predictions == 1) & (y_test.to_numpy() == 1)).sum()
))

print("Test confusion matrix:")
print(confusion_matrix(y_test, baseline_predictions, labels=[0, 1]))


# Isolation Forest
if_model = IsolationForest(
    n_estimators=200,
    contamination="auto",
    random_state=42,
    n_jobs=-1,
)

if_model.fit(X_train)

# Higher score means more anomalous.
if_val_scores = -if_model.score_samples(X_val)
if_test_scores = -if_model.score_samples(X_test)


# Train Logistic Regression
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


# Compare models on validation data
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


def evaluate_threshold(y_true, scores, threshold):
    """Evaluate binary predictions at a fixed score threshold."""
    y_true_array = np.asarray(y_true)
    predictions = (np.asarray(scores) >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true_array,
        predictions,
        labels=[0, 1],
    ).ravel()

    return {
        "flagged": int(predictions.sum()),
        "caught": int(tp),
        "false_positives": int(fp),
        "missed": int(fn),
        "precision": precision_score(
            y_true_array, predictions, zero_division=0
        ),
        "recall": recall_score(
            y_true_array, predictions, zero_division=0
        ),
        "f1": f1_score(
            y_true_array, predictions, zero_division=0
        ),
    }


review_rates = [0.002, 0.005, 0.01]

print("\nValidation review-capacity comparison")
print("-------------------------------------")

for review_rate in review_rates:
    print(f"\nTarget review rate: {review_rate:.1%}")

    for name, scores in [
        ("Isolation Forest", if_val_scores),
        ("Logistic Regression", lr_val_scores),
    ]:
        # Threshold is derived from validation data only.
        threshold = np.quantile(scores, 1 - review_rate)
        result = evaluate_threshold(y_val, scores, threshold)

        print(
            f"{name}: "
            f"flagged={result['flagged']}, "
            f"caught={result['caught']}, "
            f"false_positives={result['false_positives']}, "
            f"missed={result['missed']}, "
            f"precision={result['precision']:.4f}, "
            f"recall={result['recall']:.4f}, "
            f"f1={result['f1']:.4f}"
        )


# Final held-out test evaluation
review_rate = 0.002

if_threshold = np.quantile(
    if_val_scores,
    1 - review_rate,
)

lr_threshold = np.quantile(
    lr_val_scores,
    1 - review_rate,
)

if_test_predictions = (
    if_test_scores >= if_threshold
).astype(int)

lr_test_predictions = (
    lr_test_scores >= lr_threshold
).astype(int)


def report_test_results(name, y_true, scores, predictions):
    result = evaluate_threshold(
        y_true,
        scores,
        threshold=-np.inf,
    )

    y_true_array = np.asarray(y_true)
    predictions = np.asarray(predictions)

    tn, fp, fn, tp = confusion_matrix(
        y_true_array,
        predictions,
        labels=[0, 1],
    ).ravel()

    print(f"\n{name}")
    print("-" * len(name))
    print(
        "Test average precision:",
        round(average_precision_score(y_true_array, scores), 5),
    )
    print("Transactions flagged:", int(predictions.sum()))
    print("Fraud-labelled transactions caught:", int(tp))
    print("Legitimate transactions flagged:", int(fp))
    print("Fraud-labelled transactions missed:", int(fn))
    print(
        "Precision:",
        round(precision_score(y_true_array, predictions, zero_division=0), 4),
    )
    print(
        "Recall:",
        round(recall_score(y_true_array, predictions, zero_division=0), 4),
    )
    print(
        "F1-score:",
        round(f1_score(y_true_array, predictions, zero_division=0), 4),
    )


print("\nFINAL TEST RESULTS")
print("==================")

report_test_results(
    "Isolation Forest",
    y_test,
    if_test_scores,
    if_test_predictions,
)

report_test_results(
    "Logistic Regression",
    y_test,
    lr_test_scores,
    lr_test_predictions,
)

# Export frozen held-out predictions for SQL and Power BI
REPORTS_DIR = PROJECT_ROOT / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

output_path = REPORTS_DIR / "test_predictions.csv"

if output_path.exists():
    raise FileExistsError(
        f"{output_path} already exists. "
        "It will not be overwritten because test results are frozen."
    )

predictions_df = df.loc[
    test_indices, ["Time", "Amount", "Class"]
].copy()

predictions_df.insert(
    0,
    "Transaction_ID",
    test_indices.to_numpy(),
)

predictions_df["IsolationForest_Score"] = if_test_scores
predictions_df["IsolationForest_Flag"] = if_test_predictions

predictions_df["LogisticRegression_Score"] = lr_test_scores
predictions_df["LogisticRegression_Flag"] = lr_test_predictions

predictions_df.to_csv(output_path, index=False)

print(f"\nTest predictions exported to: {output_path}")
print("Exported rows:", len(predictions_df))
print("Exported columns:", predictions_df.columns.tolist())