import pandas as pd
import pickle
import os

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# ===============================
# CONFIG
# ===============================
DATA_PATH = "winequality-red.csv"   # change to Drive path if needed
MODEL_PATH = "wine_quality_model.pkl"

# ===============================
# Load dataset
# ===============================
data = pd.read_csv(DATA_PATH, sep=";")

# Clean column names
data.columns = data.columns.str.strip().str.lower()

# ===============================
# Features & target
# ===============================
X = data.drop("quality", axis=1)
y = data["quality"]

# Binary classification
y = y.apply(lambda x: 1 if x >= 7 else 0)

# ===============================
# Train-test split
# ===============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ===============================
# Model pipeline
# ===============================
model = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", RandomForestClassifier(
        n_estimators=300,
        random_state=42
    ))
])

# ===============================
# Train
# ===============================
model.fit(X_train, y_train)

# ===============================
# Evaluate
# ===============================
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"Model Accuracy: {accuracy:.4f}")

# ===============================
# Save model
# ===============================
with open(MODEL_PATH, "wb") as f:
    pickle.dump(model, f)

print("Model saved as wine_quality_model.pkl")
