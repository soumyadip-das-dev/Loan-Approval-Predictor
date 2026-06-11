import streamlit as st
import pandas as pd
import joblib
import os

# =========================
# PAGE CONFIG
# =========================

st.set_page_config(
    page_title="Loan Approval Predictor",
    layout="wide"
)

# =========================
# LOAD CSS
# =========================

css_path = os.path.join(
    os.path.dirname(__file__),
    "style.css"
)

if os.path.exists(css_path):
    with open(css_path) as f:
        st.markdown(
            f"<style>{f.read()}</style>",
            unsafe_allow_html=True
        )

# =========================
# LOAD MODEL V2
# =========================

model_path = os.path.join(
    os.path.dirname(__file__),
    "..",
    "models",
    "loan_pipeline_v2.pkl"
)

pipeline = joblib.load(model_path)

# =========================
# TITLE
# =========================

st.title("Loan Approval Prediction System")

st.markdown("---")

# =========================
# USER INPUTS
# =========================

col1, col2 = st.columns(2)

with col1:

    age = st.number_input(
        "Age",
        min_value=18,
        max_value=100,
        value=30
    )

    gender = st.selectbox(
        "Gender",
        ["male", "female"]
    )

    education = st.selectbox(
        "Education",
        [
            "High School",
            "Associate",
            "Bachelor",
            "Master",
            "Doctorate"
        ]
    )

    income = st.number_input(
        "Annual Income",
        min_value=1000,
        value=50000
    )

    experience = st.number_input(
        "Employment Experience (Years)",
        min_value=0,
        max_value=50,
        value=5
    )

    home_ownership = st.selectbox(
        "Home Ownership",
        [
            "RENT",
            "OWN",
            "MORTGAGE",
            "OTHER"
        ]
    )

with col2:

    loan_amount = st.number_input(
        "Loan Amount",
        min_value=500,
        value=10000
    )

    loan_intent = st.selectbox(
        "Loan Purpose",
        [
            "PERSONAL",
            "EDUCATION",
            "MEDICAL",
            "VENTURE",
            "HOMEIMPROVEMENT",
            "DEBTCONSOLIDATION"
        ]
    )

    interest_rate = st.number_input(
        "Interest Rate (%)",
        min_value=1.0,
        max_value=40.0,
        value=10.0
    )

    credit_score = st.number_input(
        "Credit Score",
        min_value=300,
        max_value=850,
        value=650
    )

# =========================
# FEATURE ENGINEERING
# =========================

def prepare_user_input():

    experience_ratio = (
        experience /
        max(age, 1)
    )

    loan_percent_income = (
        loan_amount /
        max(income, 1)
    )

    cred_hist_length = min(
        max(experience + 3, 2),
        25
    )

    return pd.DataFrame([{

        "person_age": age,
        "person_gender": gender,
        "person_education": education,
        "person_income": income,
        "person_emp_exp": experience,
        "person_home_ownership": home_ownership,
        "loan_amnt": loan_amount,
        "loan_intent": loan_intent,
        "loan_int_rate": interest_rate,
        "loan_percent_income": loan_percent_income,
        "cb_person_cred_hist_length": cred_hist_length,
        "credit_score": credit_score,
        "experience_ratio": experience_ratio

    }])

# =========================
# PREDICTION
# =========================

if st.button("Predict Loan Approval"):

    user_input = prepare_user_input()

    loan_percent_income = (
        user_input["loan_percent_income"]
        .values[0]
    )

    prediction = pipeline.predict(
        user_input
    )

    probability = pipeline.predict_proba(
        user_input
    )

    # IMPORTANT:
    # Class 0 = Approved
    # Class 1 = Rejected

    approval_probability = probability[0][0]

    rejection_probability = probability[0][1]

    decision = (
        "Approved"
        if prediction[0] == 0
        else "Rejected"
    )

    # =========================
    # RISK SCORE
    # =========================

    if approval_probability >= 0.85:

        risk = "Low Risk"
        risk_color = "green"

    elif approval_probability >= 0.60:

        risk = "Moderate Risk"
        risk_color = "orange"

    else:

        risk = "High Risk"
        risk_color = "red"

    # =========================
    # RESULTS
    # =========================

    st.markdown("---")

    st.subheader("Prediction Result")

    st.progress(float(approval_probability))

    st.caption(
        f"Approval Score: {approval_probability*100:.1f}%"
    )

    result_col1, result_col2, result_col3 = st.columns(3)

    with result_col1:
        st.metric(
            "Approval Probability",
            f"{approval_probability*100:.2f}%"
        )

    with result_col2:
        st.metric(
            "Rejection Probability",
            f"{rejection_probability*100:.2f}%"
        )

    with result_col3:
        st.metric(
            "Decision",
            decision
        )

    # =========================
    # RISK DISPLAY
    # =========================

    st.markdown(
        f"""
        <h2 style='color:{risk_color};'>
        {risk}
        </h2>
        """,
        unsafe_allow_html=True
    )

    if approval_probability >= 0.85:

        st.success(
            "Applicant has a strong approval profile."
        )

    elif approval_probability >= 0.60:

        st.warning(
            "Applicant has a moderate approval profile."
        )

    else:

        st.error(
            "Applicant has a high probability of rejection."
        )

    # =========================
    # FINANCIAL INSIGHTS
    # =========================

    st.subheader("Financial Insights")

    insights = []

    if loan_percent_income > 0.30:

        insights.append(
            "Loan amount is high relative to annual income."
        )

    if loan_percent_income < 0.20:

        insights.append(
            "Loan amount is comfortably within income capacity."
        )

    if credit_score < 600:

        insights.append(
            "Low credit score increases rejection risk."
        )

    elif credit_score >= 750:

        insights.append(
            "Excellent credit score strengthens approval chances."
        )

    if interest_rate > 15:

        insights.append(
            "High interest rate increases repayment burden."
        )

    if income < 30000:

        insights.append(
            "Lower income may affect approval probability."
        )

    if experience < 2:

        insights.append(
            "Limited employment experience increases risk."
        )

    if approval_probability >= 0.85:

        insights.append(
            "Strong financial profile detected."
        )

    if len(insights) == 0:

        insights.append(
            "Financial profile appears stable."
        )

    for insight in insights:
        st.write("•", insight)


# =========================
# FOOTER
# =========================

st.markdown("---")

st.markdown(
    """
    <center>
    Built with Streamlit, Scikit-Learn and XGBoost
    </center>
    """,
    unsafe_allow_html=True
)