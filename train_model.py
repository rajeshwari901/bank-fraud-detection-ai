import pandas as pd
import numpy as np
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

print("Creating demonstration dataset...")

# Reproducible synthetic dataset for college demonstration
rng = np.random.default_rng(42)
n = 5000

df = pd.DataFrame({
    "Time": rng.uniform(0, 172800, n),
    "V1": rng.normal(0, 1, n),
    "V2": rng.normal(0, 1, n),
    "V3": rng.normal(0, 1, n),
    "V4": rng.normal(0, 1, n),
    "amount": rng.exponential(100, n),
    "transaction_type": rng.choice(
        ["PAYMENT", "TRANSFER", "CASH_OUT", "DEBIT", "CASH_IN"],
        n
    )
})

# Synthetic labels for demonstration, not real fraud labels
risk_score = (
    (df["amount"] > 250).astype(int)
    + (df["transaction_type"].isin(["TRANSFER", "CASH_OUT"])).astype(int)
    + (df["V4"] > 1.5).astype(int)
    + (df["V1"] < -1.5).astype(int)
)

fraud_probability = np.where(risk_score >= 3, 0.75, 0.08)
df["Class"] = rng.binomial(1, fraud_probability)

feature_names = [
    "Time", "V1", "V2", "V3", "V4",
    "amount", "transaction_type"
]

X = df[feature_names]
y = df["Class"]

numeric_features = [
    "Time", "V1", "V2", "V3", "V4", "amount"
]
categorical_features = ["transaction_type"]

preprocessor = ColumnTransformer(
    transformers=[
        ("numeric", "passthrough", numeric_features),
        (
            "category",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features
        )
    ]
)

model = Pipeline([
    ("preprocessor", preprocessor),
    (
        "classifier",
        RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            class_weight="balanced"
        )
    )
])

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Training AI model...")
model.fit(X_train, y_train)

predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print("\nModel trained successfully!")
print(f"Demo dataset size: {len(df)}")
print(f"Model evaluation accuracy: {accuracy * 100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, predictions, zero_division=0))
print("Confusion Matrix:")
print(confusion_matrix(y_test, predictions))

joblib.dump(model, "fraud_model.pkl")

print("\nModel saved as fraud_model.pkl")
print("Training completed!")
print("Note: This uses synthetic demonstration data.")