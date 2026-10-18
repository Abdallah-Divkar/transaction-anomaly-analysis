import pandas as pd
import numpy as np

df = pd.read_csv("../data/raw/creditcard.csv")

print("Dataset shape:", df.shape)

print(df.head())

print("\nColumn information:")
df.info()

print("\nMissing values:")
print(df.isnull().sum().sum())

print("\nDuplicate rows:", df.duplicated().sum())

print("\nClass distribution:")
print(df["Class"].value_counts())

print("\nFraud percentage:")
print(df["Class"].mean() * 100)

# Investigate duplicate rows
duplicate_rows = df[df.duplicated(keep=False)]

print("\nRows involved in duplicates:", len(duplicate_rows))

print("\nClass distribution among duplicate rows:")
print(duplicate_rows["Class"].value_counts())

print("\nDuplicate rows by class:")
print(
    duplicate_rows.groupby("Class")
    .size()
)

# Compare class counts and percentages
class_summary = (
    df["Class"]
    .value_counts()
    .rename(index={0: "Legitimate", 1: "Fraud"})
    .to_frame("Transaction_Count")
)

class_summary["Percentage"] = (
    class_summary["Transaction_Count"] / len(df) * 100
)

print("\nClass summary:")
print(class_summary)

import matplotlib.pyplot as plt

class_counts = df["Class"].value_counts().sort_index()

class_counts.plot(
    kind="bar",
    xlabel="Class (0 = legitimate, 1 = fraud)",
    ylabel="Number of transactions",
    title="Legitimate vs Fraud-Labelled Transactions",
    rot=0
)

plt.tight_layout()
plt.show()

import matplotlib.pyplot as plt

# Compare transaction amounts by class
amount_summary = df.groupby("Class")["Amount"].agg(
    ["count", "mean", "median", "min", "max"]
)

print("\nTransaction amount summary:")
print(amount_summary)

# Compare amount distributions without extreme values dominating the chart
plt.figure(figsize=(10, 5))

for class_value, label in [(0, "Legitimate"), (1, "Fraud")]:
    amounts = df.loc[df["Class"] == class_value, "Amount"]
    plt.hist(
        amounts,
        bins=50,
        alpha=0.6,
        label=label
    )

plt.xlabel("Transaction amount")
plt.ylabel("Number of transactions")
plt.title("Transaction Amount Distribution by Class")
plt.legend()
plt.tight_layout()
plt.show()

import matplotlib.pyplot as plt
import numpy as np

# Log transformation handles the skewed transaction amounts.
# log1p(x) = log(1 + x), so zero amounts remain valid.
df["LogAmount"] = np.log1p(df["Amount"])

legitimate = df.loc[df["Class"] == 0, "LogAmount"]
fraud = df.loc[df["Class"] == 1, "LogAmount"]

plt.figure(figsize=(8, 5))

plt.boxplot(
    [legitimate, fraud],
    tick_labels=["Legitimate", "Fraud-labelled"],
    showfliers=False
)

plt.ylabel("log(1 + transaction amount)")
plt.title("Transaction Amount Distribution by Class")
plt.tight_layout()
plt.show()

####
zero_amount_summary = (
    df.assign(IsZeroAmount=df["Amount"].eq(0))
    .groupby("Class")["IsZeroAmount"]
    .agg(["sum", "mean"])
)

zero_amount_summary["Percentage"] = (
    zero_amount_summary["mean"] * 100
)

print("\nZero-amount transaction analysis:")
print(zero_amount_summary)

# Compare transaction time by class
time_summary = df.groupby("Class")["Time"].agg(
    ["count", "mean", "median", "min", "max"]
)

print("\nTransaction time summary:")
print(time_summary)

# Visualize transaction time distributions
plt.figure(figsize=(10, 5))

plt.hist(
    df.loc[df["Class"] == 0, "Time"],
    bins=50,
    density=True,
    alpha=0.6,
    label="Legitimate"
)

plt.hist(
    df.loc[df["Class"] == 1, "Time"],
    bins=50,
    density=True,
    alpha=0.6,
    label="Fraud-labelled"
)

plt.xlabel("Elapsed time (seconds)")
plt.ylabel("Proportion density")
plt.title("Transaction Time Distribution by Class")
plt.legend()
plt.tight_layout()
plt.show()

# Compare feature averages by class
feature_columns = [
    col for col in df.columns
    if col.startswith("V")
]

feature_means = df.groupby("Class")[feature_columns].mean().T
feature_means.columns = ["Legitimate", "Fraud"]

feature_means["Mean_Difference"] = (
    feature_means["Fraud"] - feature_means["Legitimate"]
)

feature_means["Absolute_Difference"] = (
    feature_means["Mean_Difference"].abs()
)

print("\nFeatures with the largest absolute mean differences:")
print(
    feature_means.sort_values(
        "Absolute_Difference",
        ascending=False
    ).head(10)
)
