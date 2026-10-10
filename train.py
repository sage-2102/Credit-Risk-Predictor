# Import only the allowed libraries for data handling and modeling
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
import joblib

# Step 1: Load the German Credit dataset directly from OpenML
print("Loading German Credit dataset...")
data = fetch_openml(name="credit-g", version=1, as_frame=True)
df = data.frame

# Step 2: Define feature columns (7 numeric + 5 categorical) and target
num_cols = [
    "duration", "credit_amount", "installment_commitment",
    "residence_since", "age", "existing_credits", "num_dependents"
]
cat_cols = ["checking_status", "credit_history", "savings_status", "employment", "purpose"]

# Step 3: One-hot encode categorical features (convert text into binary 0/1 columns).
# drop_first=True removes the first category to avoid multicollinearity (dummy trap).
X = pd.get_dummies(df[num_cols + cat_cols], drop_first=True)
y = (df["class"] == "bad").astype(int)

# Step 4: Split data into 80% training and 20% test sets (random_state=42)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Step 5: Scale all features so variables with large scales do not dominate
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 6: Train Logistic Regression model with balanced class weights
model = LogisticRegression(class_weight="balanced")
model.fit(X_train_scaled, y_train)

# Step 7: 5-Fold cross-validation provides a stable ROC-AUC estimate,
# because evaluating on a single 200-row test set is noisy and variance-prone.
cv_auc = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring="roc_auc")
print(f"\n5-Fold CV ROC-AUC: {cv_auc.mean():.4f}")

# Step 8: Evaluate model performance on the holdout test set
y_pred = model.predict(X_test_scaled)
y_prob = model.predict_proba(X_test_scaled)[:, 1]

print("\n--- Test Set Metrics ---")
print(f"Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision: {precision_score(y_test, y_pred):.4f}")
print(f"Recall:    {recall_score(y_test, y_pred):.4f}")
print(f"ROC-AUC:   {roc_auc_score(y_test, y_prob):.4f}")

# Step 9: Print top 5 risk-increasing and top 5 risk-reducing features
coef_series = pd.Series(model.coef_[0], index=X.columns)
print("\n--- Top 5 Features That Increase Risk ---")
for feat, coef in coef_series.nlargest(5).items():
    print(f"{feat:45s}: {coef:+.4f}")

print("\n--- Top 5 Features That Reduce Risk ---")
for feat, coef in coef_series.nsmallest(5).items():
    print(f"{feat:45s}: {coef:+.4f}")

# Step 10: Save model, scaler, and column names for predict.py
joblib.dump(model, "credit_model.joblib")
joblib.dump(scaler, "scaler.joblib")
joblib.dump(list(X.columns), "columns.joblib")
print("\nModel, scaler, and columns successfully saved to disk.")
