from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from fairlearn.metrics import MetricFrame, demographic_parity_difference

model = DecisionTreeClassifier(random_state=42)
model.fit(X, y)
y_pred = model.predict(X) 

mf = MetricFrame(
    metrics={
        "accuracy": accuracy_score,
        "demographic_parity_diff": demographic_parity_difference
    },
    y_true=y,
    y_pred=y_pred,
    sensitive_features=sensitive_features['age group'] 
)

print("Fairness Assessment Results:")
print(mf.overall)

print("\nMetrics by group:")

print(mf.by_group) 
