import streamlit as st
import pandas as pd
import joblib
import sys
from pathlib import Path

# --------------------------------------------------
# Project Path
# --------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

from src.config import MODEL_PATH

# --------------------------------------------------
# Load Model
# --------------------------------------------------
model = joblib.load(MODEL_PATH)

# --------------------------------------------------
# Page Configuration
# --------------------------------------------------
st.set_page_config(
    page_title="Credit Scoring Dashboard",
    page_icon="💳",
    layout="wide"
)

# --------------------------------------------------
# Title
# --------------------------------------------------
st.title("💳 Credit Scoring Prediction System")
st.markdown(
    """
Predict whether a customer is likely to default on credit payments using
Machine Learning.
"""
)

# ==================================================
# Sidebar Inputs
# ==================================================
st.sidebar.header("Customer Information")

revolving = st.sidebar.number_input(
    "Revolving Utilization",
    min_value=0.0,
    value=0.50,
    step=0.01
)

age = st.sidebar.number_input(
    "Age",
    min_value=18,
    value=35
)

late30 = st.sidebar.number_input(
    "30-59 Days Late",
    min_value=0,
    value=0
)

debt = st.sidebar.number_input(
    "Debt Ratio",
    min_value=0.0,
    value=0.45,
    step=0.01
)

income = st.sidebar.number_input(
    "Monthly Income",
    min_value=0,
    value=50000
)

credit_lines = st.sidebar.number_input(
    "Open Credit Lines",
    min_value=0,
    value=8
)

late90 = st.sidebar.number_input(
    "90 Days Late",
    min_value=0,
    value=0
)

real_estate = st.sidebar.number_input(
    "Real Estate Loans",
    min_value=0,
    value=1
)

late60 = st.sidebar.number_input(
    "60-89 Days Late",
    min_value=0,
    value=0
)

dependents = st.sidebar.number_input(
    "Dependents",
    min_value=0,
    value=2
)

# ==================================================
# Feature Engineering
# ==================================================
debt_income = debt / (income + 1)
late_payments = late30 + late60 + late90
loans_per_dependent = credit_lines / (dependents + 1)

# ==================================================
# Prediction
# ==================================================
st.markdown("---")

if st.button("🔍 Predict Credit Risk"):

    sample = pd.DataFrame({

        "RevolvingUtilizationOfUnsecuredLines":[revolving],
        "age":[age],
        "NumberOfTime30-59DaysPastDueNotWorse":[late30],
        "DebtRatio":[debt],
        "MonthlyIncome":[income],
        "NumberOfOpenCreditLinesAndLoans":[credit_lines],
        "NumberOfTimes90DaysLate":[late90],
        "NumberRealEstateLoansOrLines":[real_estate],
        "NumberOfTime60-89DaysPastDueNotWorse":[late60],
        "NumberOfDependents":[dependents],
        "DebtPerIncome":[debt_income],
        "LatePayments":[late_payments],
        "LoansPerDependent":[loans_per_dependent]

    })

    prediction = model.predict(sample)[0]
    probability = model.predict_proba(sample)[0][1]

    st.subheader("Prediction Result")

    col1, col2 = st.columns(2)

    with col1:

        if prediction == 1:

            st.error("🔴 High Credit Risk")

        else:

            st.success("🟢 Low Credit Risk")

    with col2:

        st.metric(
            "Probability of Default",
            f"{probability*100:.2f}%"
        )

    st.progress(float(probability))

    if probability < 0.20:

        st.success("Very Low Risk")

    elif probability < 0.40:

        st.info("Low Risk")

    elif probability < 0.60:

        st.warning("Medium Risk")

    else:

        st.error("High Risk")

    prediction_df = pd.DataFrame({

        "Prediction":[prediction],
        "Probability":[probability]

    })

    st.download_button(

        "📥 Download Prediction",

        prediction_df.to_csv(index=False),

        "prediction.csv",

        "text/csv"

    )

# ==================================================
# Model Metrics
# ==================================================
st.markdown("---")
st.header("📊 Model Comparison")

metrics_path = ROOT_DIR / "reports" / "metrics.csv"

if metrics_path.exists():

    metrics = pd.read_csv(metrics_path)

    st.dataframe(
        metrics,
        use_container_width=True
    )

else:

    st.info("metrics.csv not found.")

# ==================================================
# Evaluation Images
# ==================================================
st.markdown("---")
st.header("📈 Model Evaluation")

fig_dir = ROOT_DIR / "reports" / "figures"

confusion = fig_dir / "confusion_matrix.png"
roc = fig_dir / "roc_curve.png"
pr = fig_dir / "precision_recall_curve.png"
feature = fig_dir / "feature_importance.png"

col1, col2 = st.columns(2)

with col1:

    if confusion.exists():

        st.image(
            str(confusion),
            caption="Confusion Matrix",
            use_container_width=True
        )

    if pr.exists():

        st.image(
            str(pr),
            caption="Precision-Recall Curve",
            use_container_width=True
        )

with col2:

    if roc.exists():

        st.image(
            str(roc),
            caption="ROC Curve",
            use_container_width=True
        )

    if feature.exists():

        st.image(
            str(feature),
            caption="Feature Importance",
            use_container_width=True
        )

st.markdown("---")

st.caption(
    "Developed for the CodeAlpha Machine Learning Internship | Credit Scoring Model"
)