import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
np.random.seed(42)
n_samples = 1000
data = {
    'age': np.random.randint(18, 70, n_samples),
    'sex': np.random.choice(['Male', 'Female'], n_samples),
    'income': np.random.randint(20000, 150000, n_samples),
    'zip_code': np.random.randint(10000, 99999, n_samples),
    'loan_status': np.random.choice([0, 1], n_samples, p=[0.4, 0.6])
}
df = pd.DataFrame(data)
protected_attribute = 'sex'
outcome = 'loan_status'
features = ['age', 'income', 'zip_code']
X = df[features]
y = df[outcome]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
test_results = pd.DataFrame({
    'actual': y_test, 
    'predicted': y_pred
})
test_results[protected_attribute] = df.loc[X_test.index, protected_attribute].values
approval_rates = test_results.groupby(protected_attribute)['predicted'].mean() * 100
print(f"\nApproval Rates by {protected_attribute}:")
print(approval_rates)
def calculate_fnr(group):
    actual_pos = group['actual'] == 1
    pred_neg = group['predicted'] == 0
    fn = group[actual_pos & pred_neg].shape[0]
    total_actual_pos = group[actual_pos].shape[0]
    return (fn / total_actual_pos) if total_actual_pos > 0 else 0
fnr_rates = test_results.groupby(protected_attribute).apply(calculate_fnr) * 100
print(f"\nFalse Negative Rates (FNR) by {protected_attribute}:")
print(fnr_rates)
rates = test_results.groupby(protected_attribute)['predicted'].mean()
max_rate = rates.max()
min_rate = rates.min()
disparate_impact_ratio = min_rate / max_rate if max_rate > 0 else 0
print(f"\nDisparate Impact Ratio: {disparate_impact_ratio:.2f}")
if disparate_impact_ratio < 0.8:
    print("⚠️  WARNING: Potential disparate impact detected (Ratio < 0.8).")
else:
    print("✅ No immediate disparate impact detected based on approval rates.")