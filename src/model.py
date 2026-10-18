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