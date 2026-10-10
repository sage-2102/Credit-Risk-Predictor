# Project Explainer & Interview Guide (Version 2)

## 1. Version 1 vs Version 2
- **Version 1**: Used only 7 numeric features, yielding a weak test ROC-AUC of 0.5731 (near random).
- **Version 2**: Added 5 categorical features via one-hot encoding, boosting test ROC-AUC to 0.8092 (5-fold CV: 0.7659).
- **Why it helps**: Repayment ability depends heavily on qualitative financial health—such as checking balance, past default history, savings buffer, and job stability—which numeric loan amounts alone do not capture.

## 2. Line-by-Line Code Explanation
### train.py
- **L1-8 (Imports)**: Import pandas, OpenML loader, split & CV tools, StandardScaler, LogisticRegression, metrics, and joblib.
- **L10-13 (Data Load)**: Downloads the German Credit dataset from OpenML as a DataFrame.
- **L15-20 (Features & Target)**: Defines 7 numeric + 5 categorical columns; extracts binary target y (1 = bad/risk, 0 = good).
- **L22-25 (One-Hot Encoding)**: `pd.get_dummies(..., drop_first=True)` converts text categories to 0/1 binary columns, dropping the first category to prevent collinearity.
- **L27-30 (Split)**: Splits data 80% train / 20% test with `random_state=42` for reproducible evaluations.
- **L32-35 (Scaling)**: `StandardScaler` fits on train set and scales both sets so large numeric values do not dominate.
- **L37-39 (Training)**: Fits Logistic Regression with `class_weight='balanced'` to prevent bias toward the majority class.
- **L41-44 (5-Fold CV)**: `cross_val_score` splits training data into 5 folds to report a robust, low-variance ROC-AUC.
- **L46-54 (Evaluation)**: Generates test predictions and prints Accuracy, Precision, Recall, and ROC-AUC.
- **L56-65 (Top Features)**: Sorts coefficients to print top 5 risk-increasing and top 5 risk-reducing factors.
- **L67-71 (Saving)**: Saves trained model, fitted scaler, and exact column list to disk using joblib.

### predict.py
- **L1-8 (Load Artifacts)**: Loads fitted scaler, trained model, and training column list from saved joblib files.
- **L10-28 (Feature Definitions)**: Lists 7 numeric features and dictionaries of allowed options for each of the 5 categorical features.
- **L30-41 (User Input)**: Prompts user for numeric values and presents numbered menus for categorical features.
- **L43-45 (Encoding & Aligning)**: Creates a 1-row DataFrame, applies `pd.get_dummies`, and aligns columns with `reindex(columns=model_cols, fill_value=0)`.
- **L47-49 (Scaling & Probability)**: Scales input with the fitted scaler and computes default probability via `predict_proba`.
- **L51-61 (Risk Tier & Output)**: Assigns risk tier (<0.3 Low, 0.3-0.6 Medium, >0.6 High) and prints results.

## 3. Core Machine Learning Concepts
- **Logistic Regression**: Models the log-odds of a binary event using the sigmoid function ($P = 1 / (1 + e^{-z})$). *Example*: Classifying an email as spam (1) or not spam (0).
- **Feature Scaling**: Centers data to mean 0 and variance 1 so feature magnitudes do not distort gradient steps. *Example*: Comparing annual income ($60,000) to age (30).
- **Train/Test Split**: Isolates unseen data to evaluate true generalization and detect overfitting. *Example*: Practicing on sample tests, then grading on the final exam.
- **Precision vs. Recall**: Precision is accuracy among predicted positives ($TP/(TP+FP)$); Recall is detection rate of actual positives ($TP/(TP+FN)$). *Example*: Catching 90% of actual loan defaulters is 90% recall.
- **ROC-AUC**: Aggregate ranking ability across all probability thresholds (0.5 = random, 1.0 = perfect). *Example*: 0.80 AUC means an 80% chance of ranking a defaulter higher than a non-defaulter.
- **One-Hot Encoding**: Converts categorical levels into separate binary (0/1) indicator columns. *Example*: Turning "loan purpose" [car, education] into two binary flag columns.
- **K-Fold Cross-Validation**: Splits data into K chunks, training on K-1 and validating on 1 iteratively. *Example*: 5-fold CV averages 5 test runs so single-split noise doesn't mislead you.

## 4. Top 15 Interview Questions & Answers
1. **Why use Logistic Regression for credit risk?** Interpretable, produces calibrated default probabilities, and complies with financial regulations.
2. **Why scale features before Logistic Regression?** Features on larger scales (e.g., credit amount) produce disproportionate gradient updates if unscaled.
3. **What is the purpose of class_weight="balanced"?** Adjusts weights inversely proportional to class frequencies to penalize missing rare defaults.
4. **Why fit_transform on train, but only transform on test?** Avoids data leakage; test data statistics must remain completely unknown to the model.
5. **How do you interpret positive vs. negative coefficients?** Positive coefficients increase log-odds of default (higher risk); negative reduce default risk.
6. **Why prefer Recall over Accuracy in banking?** False negatives (unrecognized defaulters) cause major loan losses; false positives just lose interest.
7. **What does the ROC-AUC score measure?** The model's threshold-independent ability to distinguish defaulters from non-defaulters.
8. **Why did Version 2 outperform Version 1?** Version 1 had ROC-AUC 0.57 with only numbers; adding categorical credit history and savings added vital risk signals (0.81 AUC).
9. **How does predict.py handle unseen dummy columns?** `reindex(columns=model_cols, fill_value=0)` ensures matching column order and fills absent dummies with 0.
10. **Why use joblib over pickle?** Joblib serializes large NumPy-backed data structures much faster with lower memory overhead.
11. **What is the "dummy variable trap"?** Perfect multicollinearity caused when all one-hot levels are kept; solved by using `drop_first=True`.
12. **Why did we use pd.get_dummies instead of OneHotEncoder?** `get_dummies` is beginner-friendly and returns readable column names directly in pandas.
13. **Why do 5-fold cross-validation when we already have an 80/20 split?** A 200-row test split has high variance; 5-fold CV provides an averaged, robust performance metric.
14. **What if a categorical feature has high cardinality?** One-hot encoding creates too many sparse columns; target encoding or frequency encoding is preferred.
15. **How would you deploy this model in production?** Wrap the scaler and model in an automated scikit-learn Pipeline and expose predictions via a REST API.
