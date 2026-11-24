import streamlit as st
import pandas as pd
import joblib
import os
import sqlite3
from datetime import datetime

# ---------------- PAGE SETTINGS ----------------

st.set_page_config(
    page_title="Bank Fraud Detection AI",
    page_icon="🏦",
    layout="wide"
)

# ---------------- FILE PATHS ----------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "fraud_model.pkl")
DATABASE_PATH = os.path.join(BASE_DIR, "fraud_database.db")

st.title("🏦 Bank Fraud Detection Using AI")
st.write("AI-powered transaction classification demonstration.")

# ---------------- DATABASE ----------------

def get_connection():
    return sqlite3.connect(DATABASE_PATH)

def create_database():
    with get_connection() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS transactions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date_time TEXT,
                amount REAL,
                transaction_time REAL,
                v1 REAL,
                v2 REAL,
                v3 REAL,
                v4 REAL,
                transaction_type TEXT,
                prediction TEXT,
                confidence REAL
            )
        """)

def save_transaction(record):
    with get_connection() as conn:
        conn.execute("""
            INSERT INTO transactions (
                date_time, amount, transaction_time,
                v1, v2, v3, v4,
                transaction_type, prediction, confidence
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            record["Date and Time"],
            record["Amount"],
            record["Transaction Time"],
            record["V1"],
            record["V2"],
            record["V3"],
            record["V4"],
            record["Transaction Type"],
            record["Prediction"],
            record["Confidence (%)"]
        ))

def load_history():
    with get_connection() as conn:
        return pd.read_sql_query(
            "SELECT * FROM transactions ORDER BY id DESC",
            conn
        )

try:
    create_database()
    st.success("SQLite database connected successfully!")
except Exception as error:
    st.error(f"Database error: {error}")
    st.stop()

# ---------------- LOAD AI MODEL ----------------

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        return None
    return joblib.load(MODEL_PATH)

try:
    model = load_model()
except Exception as error:
    st.error(f"Could not load AI model: {error}")
    st.stop()

if model is None:
    st.error("AI model not found. Please train the model first.")
    st.code("python train_model.py")
    st.stop()

st.success("AI model loaded successfully!")

# ---------------- TRANSACTION INPUTS ----------------

st.subheader("💳 Enter Transaction Details")

col1, col2 = st.columns(2)

with col1:
    amount = st.number_input(
        "Transaction Amount",
        min_value=0.0,
        value=100.0,
        step=10.0
    )

    time_value = st.number_input(
        "Transaction Time",
        min_value=0.0,
        value=0.0
    )

    v1 = st.number_input("V1", value=0.0)
    v2 = st.number_input("V2", value=0.0)

with col2:
    v3 = st.number_input("V3", value=0.0)
    v4 = st.number_input("V4", value=0.0)

    transaction_type = st.selectbox(
        "Transaction Type",
        ["PAYMENT", "TRANSFER", "CASH_OUT", "DEBIT", "CASH_IN"]
    )

st.caption(
    "College project demonstration. Predictions are not "
    "reliable assessments of real banking fraud risk."
)

# ---------------- PREDICTION ----------------

if st.button("🔍 Check Transaction", type="primary"):

    try:
        input_data = pd.DataFrame([{
            "Time": time_value,
            "V1": v1,
            "V2": v2,
            "V3": v3,
            "V4": v4,
            "amount": amount,
            "transaction_type": transaction_type
        }])

        # Preserve the feature names and order used during training.
        if hasattr(model, "feature_names_in_"):
            input_data = input_data.reindex(
                columns=list(model.feature_names_in_)
            )

        prediction = int(model.predict(input_data)[0])

        confidence = None

        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(input_data)[0]
            confidence = float(max(probabilities)) * 100

        if prediction == 1:
            result = "Likely Fraudulent"
            st.error("🚨 Prediction: Likely Fraudulent")
        else:
            result = "Likely Genuine"
            st.success("✅ Prediction: Likely Genuine")

        if confidence is not None:
            st.metric(
                "Model Confidence",
                f"{confidence:.2f}%"
            )

        record = {
            "Date and Time": datetime.now().strftime(
                "%Y-%m-%d %H:%M:%S"
            ),
            "Amount": amount,
            "Transaction Time": time_value,
            "V1": v1,
            "V2": v2,
            "V3": v3,
            "V4": v4,
            "Transaction Type": transaction_type,
            "Prediction": result,
            "Confidence (%)": (
                round(confidence, 2)
                if confidence is not None else None
            )
        }

        save_transaction(record)

        st.success("Transaction saved in SQLite database!")

    except Exception as error:
        st.error(f"Prediction or database save failed: {error}")

# ---------------- TRANSACTION HISTORY ----------------

st.divider()
st.subheader("📋 Transaction History")

try:
    history = load_history()

    if not history.empty:
        st.dataframe(history, use_container_width=True)

        csv_data = history.to_csv(index=False).encode("utf-8")

        st.download_button(
            label="⬇️ Download Transaction History (CSV)",
            data=csv_data,
            file_name="transaction_history.csv",
            mime="text/csv"
        )

        st.metric("Total Transactions", len(history))

    else:
        st.info("No transactions recorded yet. Check a transaction first.")

except Exception as error:
    st.error(f"Could not load transaction history: {error}")

st.divider()

st.caption(
    "College project demonstration only. "
    "Do not use for real financial decisions."
)