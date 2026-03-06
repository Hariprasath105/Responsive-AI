import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from fairlearn.metrics import MetricFrame, selection_rate_difference
from fairlearn.preprocessing import CorrelationRemover
from diffprivlib.models import LogisticRegression as DPLoReg
import shap

# == 1. Generate Biased Data & Define Fairness/Bias ==
np.random.seed(42)
n_samples = 1000

# Sensitive feature: Gender (0=Female, 1=Male)
gender = np.random.binomial(1, 0.5, n_samples)
income = np.random.normal(50000, 15000, n_samples)

# Introducing bias: Higher income for males, lower for females
income = income + (gender * 10000)

# Target: Loan Approved (1) or Denied (0)
loan_approved = (income + np.random.normal(0, 10000, n_samples) > 55000).astype(int)

data = pd.DataFrame({
    'gender': gender, 
    'income': income, 
    'loan_approved': loan_approved
})

X = data[['gender', 'income']]
y = data['loan_approved']
sensitive_feature = data['gender']  # 0=Female (Unprivileged), 1=Male (Privileged)

# Split the data
X_train, X_test, y_train, y_test, s_train, s_test = train_test_split(
    X, y, sensitive_feature, test_size=0.3, random_state=42
)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Check Initial Bias (Fairness)
lr = LogisticRegression(solver='liblinear')
lr.fit(X_train_scaled, y_train)
preds = lr.predict(X_test_scaled)

print("--- Initial Model (Before Fairness & Privacy) ---")
print(f"Accuracy: {accuracy_score(y_test, preds):.2f}")

# Selection Rate Difference: Should be close to 0 for fairness
sr_diff = selection_rate_difference(y_test, preds, sensitive_features=s_test)
print(f"Selection Rate Difference (Bias): {sr_diff:.2f}")

# == 2. Mitigate Bias (Fairness) ==
# Remove correlation between gender and income
cr = CorrelationRemover(sensitive_feature_ids=['gender'])
X_train_fair = cr.fit_transform(X_train)
X_test_fair = cr.transform(X_test)

# Differential Privacy Model
dp_lr = DPLoReg(epsilon=1.0, data_norm=10)  # Lower epsilon = higher privacy
dp_lr.fit(X_train_fair, y_train)
dp_preds = dp_lr.predict(X_test_fair)

print("\n--- Model After Fairness Mitigation & Differential Privacy ---")
print(f"DP Accuracy: {accuracy_score(y_test, dp_preds):.2f}")

dp_sr_diff = selection_rate_difference(y_test, dp_preds, sensitive_features=s_test)
print(f"New Selection Rate Difference: {dp_sr_diff:.2f}")

print("\n-- Explainability (SHAP values) --")
explainer = shap.LinearExplainer(dp_lr, X_train_fair)
shap_values = explainer.shap_values(X_test_fair)
shap.summary_plot(shap_values, X_test_fair, feature_names=['gender', 'income'], plot_type="bar")
