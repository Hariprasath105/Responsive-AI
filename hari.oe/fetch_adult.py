import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from fairlearn.metrics import MetricFrame, selection_rate
from fairlearn.datasets import fetch_adult

# 1. Load the Data
data = fetch_adult(as_frame=True)
df = data.frame.dropna().iloc[:5000]

# 2. Define X and y (Using 'class' instead of 'income')
X = pd.get_dummies(df.drop("class", axis=1))
y = (df["class"] == ">50K").astype(int)

# 3. Create Sensitive Features
sensitive_features = df["age"].apply(lambda x: "old" if x >= 40 else "young")

# 4. Train the Model
model = DecisionTreeClassifier(random_state=42, max_depth=3)
model.fit(X, y)
y_pred = model.predict(X)

# 5. Assess Fairness
mf = MetricFrame(
    metrics={
        "accuracy": accuracy_score,
        "selection_rate": selection_rate
    },
    y_true=y,
    y_pred=y_pred,
    sensitive_features=sensitive_features
)

print("\n--- Fairness Assessment Results ---")
print(mf.by_group)
