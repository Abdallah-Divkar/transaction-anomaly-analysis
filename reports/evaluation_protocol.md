# Final Held-Out Evaluation Protocol

## Status

The test results for this experiment are frozen. The test set must not be used to select new thresholds, tune model parameters, or choose preprocessing decisions.

## Dataset

- Total transactions: 284,807
- Fraud-labelled transactions: 492
- Fraud-label prevalence: approximately 0.17275%
- Features: `Time`, `Amount`, and anonymized variables `V1`–`V28`
- Target: `Class`, where 1 means fraud-labelled and 0 means legitimate-labelled

## Data splitting

Used `StratifiedGroupKFold` with five folds, shuffling enabled, and `random_state=42`.

- Training: folds 0, 1, and 2
- Validation: fold 3
- Test: fold 4

Rows were grouped using hashes of their feature values so identical feature rows would not cross split boundaries. The resulting split contained 170,884 training rows, 56,961 validation rows, and 56,962 test rows. Group-overlap checks returned zero overlap between each pair of splits.

## Models

1. **Dummy baseline:** `DummyClassifier(strategy="most_frequent")`.
2. **Isolation Forest:** 200 trees, `contamination="auto"`, `random_state=42`, trained without the `Class` target.
3. **Logistic Regression:** `StandardScaler` followed by Logistic Regression with `class_weight="balanced"`, `max_iter=2000`, and `random_state=42`.

## Threshold selection

The target review capacity was set to 0.2%. Each model's score threshold was calculated from its validation scores using the 99.8th percentile (`np.quantile(scores, 0.998)`). The resulting thresholds were applied unchanged to the held-out test scores.

The test results were not used to select thresholds.

## Final test results

| Metric | Isolation Forest | Logistic Regression |
|---|---:|---:|
| Average precision | 0.15035 | 0.70094 |
| Flagged transactions | 109 | 129 |
| True positives | 25 | 80 |
| False positives | 84 | 49 |
| False negatives | 74 | 19 |
| Precision | 0.2294 | 0.6202 |
| Recall | 0.2525 | 0.8081 |
| F1-score | 0.2404 | 0.7018 |

## Interpretation and limitations

Logistic Regression performed better than Isolation Forest on the reported held-out metrics. These results measure agreement with the dataset's existing labels, not verified real-world fraud detection.

The dataset does not establish a currency for `Amount`; therefore, monetary values must not be labelled CAD or another currency without further evidence. `Time` represents elapsed seconds in the dataset, not a confirmed transaction timestamp. The anonymized features have no reliable business interpretation from the provided data alone.

Flagged transactions are candidates for review, not confirmed fraud. False positives represent legitimate-labelled transactions flagged by a model; false negatives represent fraud-labelled transactions the model did not flag.
