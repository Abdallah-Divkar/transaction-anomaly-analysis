
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Data loading
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "creditcard.csv"

df = pd.read_csv(DATA_PATH)

print("Dataset shape:", df.shape)
print("\nFirst five rows:")
print(df.head())

print("\nColumn information:")
df.info()

print("\nTotal missing values:", df.isnull().sum().sum())
print("Duplicate rows:", df.duplicated().sum())


# Class distribution and duplicate analysis
class_names = {0: "Legitimate", 1: "Fraud-labelled"}

print("\nClass distribution:")
print(df["Class"].value_counts().sort_index())

print("\nFraud-label percentage:")
print(f"{df['Class'].mean() * 100:.5f}%")

class_summary = (
    df["Class"]
    .value_counts()
    .sort_index()
    .rename(index=class_names)
    .to_frame("Transaction_Count")
)

class_summary["Percentage"] = (
    class_summary["Transaction_Count"] / len(df) * 100
)

print("\nClass summary:")
print(class_summary)

duplicate_rows = df[df.duplicated(keep=False)]

print("\nRows belonging to duplicate groups:", len(duplicate_rows))
print("\nClass distribution among these rows:")
print(duplicate_rows["Class"].value_counts().sort_index())


# Class imbalance chart
class_counts = df["Class"].value_counts().sort_index()

class_counts.index = [
    class_names[value] for value in class_counts.index
]

class_counts.plot(
    kind="bar",
    xlabel="Transaction class",
    ylabel="Number of transactions",
    title="Legitimate vs Fraud-Labelled Transactions",
    rot=0,
)

plt.tight_layout()
plt.show()

# Transaction amount analysis
amount_summary = df.groupby("Class")["Amount"].agg(
    ["count", "mean", "median", "min", "max"]
)

print("\nTransaction amount summary:")
print(amount_summary)

df["LogAmount"] = np.log1p(df["Amount"])

plt.figure(figsize=(9, 5))

for class_value, label in class_names.items():
    amounts = df.loc[df["Class"] == class_value, "LogAmount"]

    plt.hist(
        amounts,
        bins=50,
        density=True,
        alpha=0.6,
        label=label,
    )

plt.xlabel("log(1 + transaction amount)")
plt.ylabel("Density")
plt.title("Transaction Amount Distribution by Class")
plt.legend()
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 5))

boxplot_data = [
    df.loc[df["Class"] == class_value, "LogAmount"]
    for class_value in [0, 1]
]

plt.boxplot(
    boxplot_data,
    tick_labels=["Legitimate", "Fraud-labelled"],
    showfliers=False,
)

plt.ylabel("log(1 + transaction amount)")
plt.title("Transaction Amount Comparison by Class")
plt.tight_layout()
plt.show()


# Zero-amount analysis
zero_amount_summary = (
    df.groupby("Class")["Amount"]
    .agg(
        Zero_Amount_Count=lambda values: values.eq(0).sum(),
        Transaction_Count="size",
    )
)

zero_amount_summary["Percentage"] = (
    zero_amount_summary["Zero_Amount_Count"]
    / zero_amount_summary["Transaction_Count"]
    * 100
)

print("\nZero-amount transaction analysis:")
print(zero_amount_summary)


# Transaction time analysis
time_summary = df.groupby("Class")["Time"].agg(
    ["count", "mean", "median", "min", "max"]
)

print("\nTransaction time summary:")
print(time_summary)

plt.figure(figsize=(9, 5))

for class_value, label in class_names.items():
    times = df.loc[df["Class"] == class_value, "Time"]

    plt.hist(
        times,
        bins=50,
        density=True,
        alpha=0.6,
        label=label,
    )

plt.xlabel("Elapsed time (seconds)")
plt.ylabel("Density")
plt.title("Transaction Time Distribution by Class")
plt.legend()
plt.tight_layout()
plt.show()

# Anonymized feature comparison
feature_columns = [
    column
    for column in df.columns
    if column.startswith("V")
]

feature_means = df.groupby("Class")[feature_columns].mean().T
feature_means.columns = ["Legitimate", "Fraud"]

feature_means["Mean_Difference"] = (
    feature_means["Fraud"] - feature_means["Legitimate"]
)

feature_means["Absolute_Difference"] = (
    feature_means["Mean_Difference"].abs()
)

top_features = feature_means.sort_values(
    "Absolute_Difference",
    ascending=False,
).head(10)

print("\nFeatures with the largest absolute mean differences:")
print(top_features)
