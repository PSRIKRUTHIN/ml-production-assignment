import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier

# Read iris.csv
df = pd.read_csv("iris (2).csv", skiprows=1, header=None)

# Give the columns proper names
df.columns = [
    "sepal_length",
    "sepal_width",
    "petal_length",
    "petal_width",
    "target"
]

# Separate features and target
X = df[
    [
        "sepal_length",
        "sepal_width",
        "petal_length",
        "petal_width"
    ]
]

y = df["target"]

# Train model
model = RandomForestClassifier(random_state=42)
model.fit(X, y)

# Save trained model
joblib.dump(model, "model.pkl")

print("Model trained successfully!")
print("model.pkl created successfully!")