# Export trained model and scaler parameters to JSON for GitHub Pages frontend
import json
import joblib

# Load trained logistic regression model, fitted scaler, and column names
model = joblib.load("credit_model.joblib")
scaler = joblib.load("scaler.joblib")
columns = joblib.load("columns.joblib")

# Pack parameters into a clean dictionary
params = {
    "intercept": float(model.intercept_[0]),
    "coefficients": {col: float(coef) for col, coef in zip(columns, model.coef_[0])},
    "means": {col: float(m) for col, m in zip(columns, scaler.mean_)},
    "scales": {col: float(s) for col, s in zip(columns, scaler.scale_)},
    "columns": columns
}

# Save parameters to model_params.json for the web frontend
with open("model_params.json", "w") as f:
    json.dump(params, f, indent=2)

print("Exported model parameters to model_params.json successfully!")
