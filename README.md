# Credit Risk Predictor (Version 2)

A beginner-friendly machine learning project in Python that predicts credit default risk (1 = bad/risky, 0 = good) using Logistic Regression on the German Credit dataset (`credit-g`).

## Version 1 vs Version 2
- **Version 1 (Baseline)**: Used only 7 numeric features and achieved a test **ROC-AUC of 0.5731** (barely above random guessing).
- **Version 2 (Improved)**: Added 5 categorical features encoded with `pd.get_dummies(drop_first=True)`. Test **ROC-AUC jumped to 0.8092** (and 5-fold CV ROC-AUC is **0.7659**).
- **Why it helps**: Financial behavior context—such as having no overdraft history, a stable savings reserve, or good past loan repayment records—is far more predictive of default than raw numeric amounts alone.

## Features Used
- **7 Numeric**: `duration`, `credit_amount`, `installment_commitment`, `residence_since`, `age`, `existing_credits`, `num_dependents`.
- **5 Categorical**: `checking_status`, `credit_history`, `savings_status`, `employment`, `purpose`.

## How to Run Locally
```bash
# 1. Train model, perform 5-fold CV, and save artifacts
python train.py

# 2. Interactive CLI prediction with numbered menus
python predict.py
```

## Free GitHub Pages Deployment (Live Web App)
This repository includes a standalone, client-side web application (`index.html`, `style.css`, `app.js`) that runs inference locally in the browser with zero backend latency:
1. Push this project to a GitHub repository:
   ```bash
   git add .
   git commit -m "Add GitHub Pages web frontend"
   git push origin main
   ```
2. In your repository on GitHub, navigate to **Settings** &rarr; **Pages**.
3. Under **Build and deployment**, set **Source** to `Deploy from a branch`, choose branch `main` and folder `/ (root)`, and click **Save**.
4. Your interactive predictor will be live at `https://<your-username>.github.io/<repository-name>/`.

## Understanding the Output
- **Probability of Default**: Estimated chance (0% to 100%) that the applicant will fail to repay.
- **Risk Tiers**:
  - **Low risk** (`< 0.3`): Safe borrower; favorable loan terms.
  - **Medium risk** (`0.3 - 0.6`): Moderate risk; manual review or collateral advised.
  - **High risk** (`> 0.6`): High default likelihood; loan denial recommended.
- **Top Risk Drivers (Coefficients)**:
  - **Increases Risk**: High `credit_amount` (+0.36), long `duration` (+0.28), high `installment_commitment` (+0.28).
  - **Reduces Risk**: Strong `credit_history` (-0.82), `no checking` account overdraft (-0.56), older `age` (-0.37).
