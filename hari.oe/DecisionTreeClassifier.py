import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from fairlearn.metrics import MetricFrame, selection_rate
from fairlearn.datasets import fetch_adult

data = fetch_adult(as_frame=True)
df = data.frame.dropna().iloc[:5000] 

X = pd.get_dummies(df.drop("income", axis=1))
y = (df["income"] == ">50K").astype(int)

sensitive_features = pd.DataFrame()
sensitive_features['age_group'] = df["age"].apply(lambda x: "old" if x >= 40 else "young")

model = DecisionTreeClassifier(random_state=42, max_depth=3)
model.fit(X, y)
y_pred = model.predict(X)

mf = MetricFrame(
    metrics={
        "accuracy": accuracy_score,
        "selection_rate": selection_rate 
    },
    y_true=y,
    y_pred=y_pred,
    sensitive_features=sensitive_features['age_group']
)

print("\n--- Fairness Assessment Results ---")
print(mf.by_group)

