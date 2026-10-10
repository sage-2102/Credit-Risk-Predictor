/**
 * Credit Risk Predictor - Client-side Inference Engine
 * Evaluates calibrated Logistic Regression weights locally for zero-latency GitHub Pages deployment.
 */

// Model Parameters exported from scikit-learn
const MODEL_DATA = {
  intercept: -0.30991551836648307,
  coefficients: {
    duration: 0.2773403551364842,
    credit_amount: 0.36441197138933457,
    installment_commitment: 0.2764251267473858,
    residence_since: 0.09743544966083299,
    age: -0.370249203591297,
    existing_credits: 0.18136975893534107,
    num_dependents: 0.0018844364125952008,
    "checking_status_<0": 0.19849732403306658,
    "checking_status_>=200": -0.08951785567594375,
    "checking_status_no checking": -0.5564427125810315,
    "credit_history_critical/other existing credit": -0.8173526941933812,
    "credit_history_delayed previously": -0.3447356306825003,
    "credit_history_existing paid": -0.44760756535145624,
    "credit_history_no credits/all paid": -0.019404183623433668,
    "savings_status_500<=X<1000": -0.03611915373737573,
    "savings_status_<100": 0.03139136405535874,
    "savings_status_>=1000": -0.13353468128504817,
    "savings_status_no known savings": -0.18424850551957048,
    "employment_4<=X<7": -0.21960483105643253,
    "employment_<1": 0.06134799098895815,
    "employment_>=7": -0.0953845980305772,
    "employment_unemployed": 0.071854052287671,
    "purpose_new car": 0.24944125237845077,
    "purpose_used car": -0.2693926304933737,
    "purpose_business": 0.01579076494959733,
    "purpose_education": 0.25244101930952084,
    "purpose_furniture/equipment": -0.057879646360714264,
    "purpose_other": -0.05869422596648016,
    "purpose_radio/tv": -0.1657158489723409,
    "purpose_repairs": 0.029560855464434602,
    "purpose_retraining": -0.09811481972174728
  },
  means: {
    duration: 21.095,
    credit_amount: 3360.61875,
    installment_commitment: 2.965,
    residence_since: 2.84625,
    age: 35.55875,
    existing_credits: 1.41125,
    num_dependents: 1.14375,
    "checking_status_<0": 0.2675,
    "checking_status_>=200": 0.06125,
    "checking_status_no checking": 0.40375,
    "credit_history_critical/other existing credit": 0.2925,
    "credit_history_delayed previously": 0.09125,
    "credit_history_existing paid": 0.525,
    "credit_history_no credits/all paid": 0.04375,
    "savings_status_500<=X<1000": 0.07,
    "savings_status_<100": 0.5925,
    "savings_status_>=1000": 0.04875,
    "savings_status_no known savings": 0.18625,
    "employment_4<=X<7": 0.175,
    "employment_<1": 0.1675,
    "employment_>=7": 0.2525,
    "employment_unemployed": 0.0675,
    "purpose_new car": 0.2375,
    "purpose_used car": 0.105,
    "purpose_business": 0.09875,
    "purpose_education": 0.04625,
    "purpose_furniture/equipment": 0.17125,
    "purpose_other": 0.0125,
    "purpose_radio/tv": 0.2825,
    "purpose_repairs": 0.02375,
    "purpose_retraining": 0.00875
  },
  scales: {
    duration: 11.799829447919999,
    credit_amount: 2896.3629370986014,
    installment_commitment: 1.1219514249734701,
    residence_since: 1.1045863196237766,
    age: 11.404453009132002,
    existing_credits: 0.5784664532191992,
    num_dependents: 0.35083605501715465,
    "checking_status_<0": 0.4426553399655312,
    "checking_status_>=200": 0.2397883181057826,
    "checking_status_no checking": 0.49064848669898087,
    "credit_history_critical/other existing credit": 0.4549107055236225,
    "credit_history_delayed previously": 0.2879642990024978,
    "credit_history_existing paid": 0.4993746088859544,
    "credit_history_no credits/all paid": 0.20453835214941968,
    "savings_status_500<=X<1000": 0.25514701644346144,
    "savings_status_<100": 0.491369260332797,
    "savings_status_>=1000": 0.2153449268034889,
    "savings_status_no known savings": 0.3893082808007043,
    "employment_4<=X<7": 0.3799671038392666,
    "employment_<1": 0.37342167853513814,
    "employment_>=7": 0.4344464869233034,
    "employment_unemployed": 0.250885930255166,
    "purpose_new car": 0.42555111326373,
    "purpose_used car": 0.3065534211193866,
    "purpose_business": 0.29832605903608217,
    "purpose_education": 0.21002604005218015,
    "purpose_furniture/equipment": 0.3767272720417252,
    "purpose_other": 0.11110243021644486,
    "purpose_radio/tv": 0.4502152263084846,
    "purpose_repairs": 0.15226929270210718,
    "purpose_retraining": 0.0931312917337669
  },
  columns: [
    "duration", "credit_amount", "installment_commitment", "residence_since",
    "age", "existing_credits", "num_dependents",
    "checking_status_<0", "checking_status_>=200", "checking_status_no checking",
    "credit_history_critical/other existing credit", "credit_history_delayed previously",
    "credit_history_existing paid", "credit_history_no credits/all paid",
    "savings_status_500<=X<1000", "savings_status_<100", "savings_status_>=1000",
    "savings_status_no known savings", "employment_4<=X<7", "employment_<1",
    "employment_>=7", "employment_unemployed", "purpose_new car", "purpose_used car",
    "purpose_business", "purpose_education", "purpose_furniture/equipment",
    "purpose_other", "purpose_radio/tv", "purpose_repairs", "purpose_retraining"
  ]
};

// UI Elements
const durationInput = document.getElementById("duration");
const creditAmountInput = document.getElementById("credit_amount");
const installmentInput = document.getElementById("installment_commitment");
const residenceInput = document.getElementById("residence_since");
const ageInput = document.getElementById("age");
const existingCreditsInput = document.getElementById("existing_credits");
const dependentsInput = document.getElementById("num_dependents");

const checkingStatusInput = document.getElementById("checking_status");
const creditHistoryInput = document.getElementById("credit_history");
const savingsStatusInput = document.getElementById("savings_status");
const employmentInput = document.getElementById("employment");
const purposeInput = document.getElementById("purpose");

// Value Display Badges
const valDuration = document.getElementById("valDuration");
const valCreditAmount = document.getElementById("valCreditAmount");
const valInstallment = document.getElementById("valInstallment");
const valResidence = document.getElementById("valResidence");
const valAge = document.getElementById("valAge");
const valExistingCredits = document.getElementById("valExistingCredits");
const valDependents = document.getElementById("valDependents");

// Result Displays
const probDisplay = document.getElementById("probDisplay");
const riskBadge = document.getElementById("riskBadge");
const gaugeProgressArc = document.getElementById("gaugeProgressArc");
const decisionBox = document.getElementById("decisionBox");
const decisionIcon = document.getElementById("decisionIcon");
const decisionHeadline = document.getElementById("decisionHeadline");
const decisionText = document.getElementById("decisionText");
const factorsList = document.getElementById("factorsList");

// Presets
const presets = {
  prime: {
    duration: 6,
    credit_amount: 800,
    installment_commitment: 1,
    residence_since: 4,
    age: 50,
    existing_credits: 2,
    num_dependents: 1,
    checking_status: "no checking",
    credit_history: "critical/other existing credit",
    savings_status: ">=1000",
    employment: ">=7",
    purpose: "used car"
  },
  average: {
    duration: 24,
    credit_amount: 3500,
    installment_commitment: 2,
    residence_since: 2,
    age: 30,
    existing_credits: 1,
    num_dependents: 1,
    checking_status: "0<=X<200",
    credit_history: "existing paid",
    savings_status: "100<=X<500",
    employment: "1<=X<4",
    purpose: "radio/tv"
  },
  risky: {
    duration: 48,
    credit_amount: 8000,
    installment_commitment: 4,
    residence_since: 1,
    age: 22,
    existing_credits: 1,
    num_dependents: 2,
    checking_status: "<0",
    credit_history: "no credits/all paid",
    savings_status: "<100",
    employment: "unemployed",
    purpose: "education"
  }
};

/**
 * Update UI value display badges
 */
function updateBadges() {
  valDuration.textContent = `${durationInput.value} mo`;
  valCreditAmount.textContent = `${Number(creditAmountInput.value).toLocaleString()} DM`;
  valInstallment.textContent = `${installmentInput.value}%`;
  valResidence.textContent = `${residenceInput.value} yr`;
  valAge.textContent = `${ageInput.value} yrs`;
  valExistingCredits.textContent = existingCreditsInput.value;
  valDependents.textContent = dependentsInput.value;
}

/**
 * Predict risk probability and calculate factor contributions
 */
function evaluateRisk() {
  updateBadges();

  const userVals = {
    duration: parseFloat(durationInput.value),
    credit_amount: parseFloat(creditAmountInput.value),
    installment_commitment: parseFloat(installmentInput.value),
    residence_since: parseFloat(residenceInput.value),
    age: parseFloat(ageInput.value),
    existing_credits: parseFloat(existingCreditsInput.value),
    num_dependents: parseFloat(dependentsInput.value)
  };

  // Convert categorical choices to active dummy flags
  const selectedCatKeys = [
    `checking_status_${checkingStatusInput.value}`,
    `credit_history_${creditHistoryInput.value}`,
    `savings_status_${savingsStatusInput.value}`,
    `employment_${employmentInput.value}`,
    `purpose_${purposeInput.value}`
  ];

  // Calculate Log-Odds (z) and factor contributions
  let z = MODEL_DATA.intercept;
  const factorContributions = [];

  MODEL_DATA.columns.forEach((col) => {
    let rawVal = 0;
    if (userVals[col] !== undefined) {
      rawVal = userVals[col];
    } else if (selectedCatKeys.includes(col)) {
      rawVal = 1;
    }

    const mean = MODEL_DATA.means[col];
    const scale = MODEL_DATA.scales[col];
    const coef = MODEL_DATA.coefficients[col];

    const scaled = (rawVal - mean) / scale;
    const impact = scaled * coef;
    z += impact;

    factorContributions.push({
      col: col.replace(/_/g, " "),
      impact: impact,
      isRiskUp: impact > 0
    });
  });

  // Logistic Sigmoid Function: P(default) = 1 / (1 + exp(-z))
  const prob = 1 / (1 + Math.exp(-z));
  renderResults(prob, factorContributions);
}

/**
 * Render calculated risk, gauge chart, and decision recommendation
 */
function renderResults(prob, factors) {
  const percentText = (prob * 100).toFixed(1) + "%";
  probDisplay.textContent = percentText;

  // Arc stroke dashoffset: 314.159 is empty, 0 is full
  const maxOffset = 314.159;
  const targetOffset = maxOffset * (1 - prob);
  gaugeProgressArc.style.strokeDashoffset = targetOffset;

  // Tiers and Styling
  riskBadge.className = "risk-badge";
  if (prob < 0.3) {
    riskBadge.classList.add("risk-low");
    riskBadge.textContent = "Low Risk (<30%)";
    probDisplay.style.color = "var(--risk-low)";
    decisionIcon.textContent = "✅";
    decisionHeadline.textContent = "Underwriting Recommended: Approved";
    decisionText.textContent = "Applicant displays a disciplined financial profile with low default probability. Fast-track prime rate approval.";
  } else if (prob <= 0.6) {
    riskBadge.classList.add("risk-medium");
    riskBadge.textContent = "Medium Risk (30-60%)";
    probDisplay.style.color = "var(--risk-medium)";
    decisionIcon.textContent = "⚖️";
    decisionHeadline.textContent = "Underwriting Review: Conditional";
    decisionText.textContent = "Applicant displays moderate credit exposure. Recommend requesting secondary proof of income or a co-signer.";
  } else {
    riskBadge.classList.add("risk-high");
    riskBadge.textContent = "High Risk (>60%)";
    probDisplay.style.color = "var(--risk-high)";
    decisionIcon.textContent = "⚠️";
    decisionHeadline.textContent = "Underwriting Warning: Rejection Advised";
    decisionText.textContent = "Applicant exhibits high default risk indicators (e.g. overdrawn checking, unbacked debt, or high burden). Reject or require full collateral.";
  }

  // Render Top 4 Influencing Factors
  factors.sort((a, b) => Math.abs(b.impact) - Math.abs(a.impact));
  const topFactors = factors.slice(0, 4);

  factorsList.innerHTML = topFactors.map((f) => {
    const isUp = f.impact > 0;
    const barClass = isUp ? "risk-up" : "risk-down";
    const absVal = Math.min(Math.abs(f.impact) * 45, 100);
    const sign = isUp ? "+" : "-";
    return `
      <div class="factor-item">
        <span class="factor-name" title="${f.col}">${f.col}</span>
        <div class="factor-bar-wrapper">
          <div class="factor-bar ${barClass}" style="width: ${absVal}%"></div>
        </div>
        <span class="factor-val ${barClass}">${sign}${Math.abs(f.impact).toFixed(2)}</span>
      </div>
    `;
  }).join("");
}

/**
 * Apply a preset archetype
 */
function applyPreset(presetData) {
  durationInput.value = presetData.duration;
  creditAmountInput.value = presetData.credit_amount;
  installmentInput.value = presetData.installment_commitment;
  residenceInput.value = presetData.residence_since;
  ageInput.value = presetData.age;
  existingCreditsInput.value = presetData.existing_credits;
  dependentsInput.value = presetData.num_dependents;

  checkingStatusInput.value = presetData.checking_status;
  creditHistoryInput.value = presetData.credit_history;
  savingsStatusInput.value = presetData.savings_status;
  employmentInput.value = presetData.employment;
  purposeInput.value = presetData.purpose;

  evaluateRisk();
}

// Event Listeners
const allInputs = [
  durationInput, creditAmountInput, installmentInput, residenceInput,
  ageInput, existingCreditsInput, dependentsInput,
  checkingStatusInput, creditHistoryInput, savingsStatusInput,
  employmentInput, purposeInput
];

allInputs.forEach((el) => {
  el.addEventListener("input", evaluateRisk);
  el.addEventListener("change", evaluateRisk);
});

document.getElementById("btnPresetPrime").addEventListener("click", () => applyPreset(presets.prime));
document.getElementById("btnPresetAverage").addEventListener("click", () => applyPreset(presets.average));
document.getElementById("btnPresetRisky").addEventListener("click", () => applyPreset(presets.risky));
document.getElementById("btnReset").addEventListener("click", () => applyPreset(presets.average));

// Initial Evaluation
evaluateRisk();
