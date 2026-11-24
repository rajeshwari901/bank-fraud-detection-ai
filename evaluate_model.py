import os
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

print("=" * 50)
print("BANK FRAUD DETECTION - MODEL EVALUATION")
print("=" * 50)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "fraud_model.pkl")

if not os.path.exists(MODEL_PATH):
    print("ERROR: fraud_model.pkl not found!")
    print("Please run: python train_model.py")
    raise SystemExit(1)

print("\nLoading trained AI model...")
model = joblib.load(MODEL_PATH)

# Recreate the exact synthetic dataset used in training
print("Creating the demonstration dataset...")

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

risk_score = (
    (df["amount"] > 250).astype(int)
    + df["transaction_type"].isin(
        ["TRANSFER", "CASH_OUT"]
    ).astype(int)
    + (df["V4"] > 1.5).astype(int)
    + (df["V1"] < -1.5).astype(int)
)

fraud_probability = np.where(
    risk_score >= 3, 0.75, 0.08
)

df["Class"] = rng.binomial(1, fraud_probability)

feature_names = [
    "Time", "V1", "V2", "V3", "V4",
    "amount", "transaction_type"
]

X = df[feature_names]
y = df["Class"]

# Use the same test split as train_model.py
_, X_test, _, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Evaluating the trained model...")

predictions = model.predict(X_test)

# Calculate evaluation metrics
accuracy = accuracy_score(y_test, predictions)
precision = precision_score(
    y_test, predictions, zero_division=0
)
recall = recall_score(
    y_test, predictions, zero_division=0
)
f1 = f1_score(
    y_test, predictions, zero_division=0
)

matrix = confusion_matrix(
    y_test, predictions, labels=[0, 1]
)

report = classification_report(
    y_test,
    predictions,
    labels=[0, 1],
    target_names=["Genuine", "Fraud"],
    zero_division=0
)

# Display results
print("\n" + "=" * 50)
print("MODEL EVALUATION RESULTS")
print("=" * 50)

print(f"Total dataset records: {len(df)}")
print(f"Test records: {len(X_test)}")
print(f"Accuracy:  {accuracy * 100:.2f}%")
print(f"Precision: {precision * 100:.2f}%")
print(f"Recall:    {recall * 100:.2f}%")
print(f"F1-score:  {f1 * 100:.2f}%")

print("\nClassification Report:")
print(report)

print("Confusion Matrix:")
print("Rows = Actual, Columns = Predicted")
print("Order: Genuine (0), Fraud (1)")
print(matrix)

# Save report to a text file
report_path = os.path.join(
    BASE_DIR, "evaluation_report.txt"
)

with open(report_path, "w", encoding="utf-8") as file:
    file.write("BANK FRAUD DETECTION - MODEL EVALUATION\n")
    file.write("=" * 45 + "\n")
    file.write(f"Dataset size: {len(df)}\n")
    file.write(f"Test records: {len(X_test)}\n")
    file.write(f"Accuracy: {accuracy * 100:.2f}%\n")
    file.write(f"Precision: {precision * 100:.2f}%\n")
    file.write(f"Recall: {recall * 100:.2f}%\n")
    file.write(f"F1-score: {f1 * 100:.2f}%\n\n")
    file.write("Classification Report:\n")
    file.write(report + "\n")
    file.write("Confusion Matrix:\n")
    file.write(str(matrix) + "\n\n")
    file.write(
        "Note: Results are based on synthetic demonstration "
        "data and do not establish real-world fraud detection "
        "performance.\n"
    )

# Save confusion matrix as CSV
matrix_df = pd.DataFrame(
    matrix,
    index=["Actual Genuine", "Actual Fraud"],
    columns=["Predicted Genuine", "Predicted Fraud"]
)

matrix_df.to_csv(
    os.path.join(BASE_DIR, "confusion_matrix.csv")
)

print("\nEvaluation completed successfully!")
print("Report saved: evaluation_report.txt")
print("Confusion matrix saved: confusion_matrix.csv")
print("Note: Results use synthetic demonstration data.")