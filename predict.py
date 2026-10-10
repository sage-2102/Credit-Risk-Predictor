# Import pandas to format inputs and joblib to load saved artifacts
import pandas as pd
import joblib

# Step 1: Load saved model, scaler, and training column list
scaler = joblib.load("scaler.joblib")
model = joblib.load("credit_model.joblib")
model_cols = joblib.load("columns.joblib")

# Step 2: Define numeric features and categorical options
num_features = [
    "duration", "credit_amount", "installment_commitment",
    "residence_since", "age", "existing_credits", "num_dependents"
]
cat_options = {
    "checking_status": ["<0", "0<=X<200", ">=200", "no checking"],
    "credit_history": [
        "critical/other existing credit", "existing paid",
        "delayed previously", "no credits/all paid", "all paid"
    ],
    "savings_status": [
        "<100", "100<=X<500", "500<=X<1000", ">=1000", "no known savings"
    ],
    "employment": ["unemployed", "<1", "1<=X<4", "4<=X<7", ">=7"],
    "purpose": [
        "radio/tv", "education", "furniture/equipment", "new car", "used car",
        "business", "domestic appliance", "repairs", "other", "retraining"
    ]
}

# Step 3: Prompt user for numeric features and numbered categorical menus
print("--- Enter Applicant Details ---")
user_inputs = {}
for feat in num_features:
    user_inputs[feat] = float(input(f"Enter {feat}: "))

for col, options in cat_options.items():
    print(f"\nSelect {col}:")
    for i, opt in enumerate(options, 1):
        print(f"  {i}. {opt}")
    choice = int(input(f"Enter choice (1-{len(options)}): ")) - 1
    user_inputs[col] = options[choice]

# Step 4: One-hot encode inputs and match exact training columns
input_df = pd.DataFrame([user_inputs])
input_df = pd.get_dummies(input_df).reindex(columns=model_cols, fill_value=0)

# Step 5: Scale features and predict default probability
input_scaled = scaler.transform(input_df)
prob_default = model.predict_proba(input_scaled)[0, 1]

# Step 6: Determine risk label based on default probability
if prob_default < 0.3:
    risk_label = "Low risk"
elif prob_default <= 0.6:
    risk_label = "Medium risk"
else:
    risk_label = "High risk"

# Step 7: Display predicted risk assessment
print("\n--- Credit Risk Assessment ---")
print(f"Probability of Default: {prob_default:.2%}")
print(f"Risk Level: {risk_label}")
