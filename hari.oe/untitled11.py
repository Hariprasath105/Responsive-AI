from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from fairlearn.metrics import MetricFrame, demographic_parity_difference

# 2. Train a simple model
model = DecisionTreeClassifier(random_state=42)
model.fit(X, y)
y_pred = model.predict(X) # Fixed: added underscore and equals sign

# 3. Assess fairness using Fairlearn metrics
# Fixed: accuracy_score and demographic_parity_difference need underscores
mf = MetricFrame(
    metrics={
        "accuracy": accuracy_score,
        "demographic_parity_diff": demographic_parity_difference
    },
    y_true=y,
    y_pred=y_pred,
    sensitive_features=sensitive_features['age group'] # Fixed: added underscore
)

print("Fairness Assessment Results:")
print(mf.overall)

print("\nMetrics by group:")
print(mf.by_group) # Fixed: added underscore