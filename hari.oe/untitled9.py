import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from fairlearn.metrics import MetricFrame, demographic_parity_difference
from fairlearn.datasets import fetch_adult

# Load and prep
data = fetch_adult(as_frame=True)
df = data.frame.dropna()
X = pd.get_dummies(df.drop("income", axis=1))
y = (df["income"] == ">50K").astype(int)

# Define groups
sensitive_groups = df["age"].apply(lambda x: "old" if x >= 40 else "young")

# Model
model = DecisionTreeClassifier(max_depth=3).fit(X, y)
y_pred = model.predict(X)

# Metrics
mf = MetricFrame(metrics={"accuracy": lambda y_true, y_pred: (y_true == y_pred).mean()},
                 y_true=y, y_pred=y_pred,
                 sensitive_features=sensitive_groups)

print(mf.by_group)