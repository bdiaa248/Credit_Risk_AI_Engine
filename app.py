import streamlit as st
import pandas as pd
import joblib
import os

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="AI Risk Engine Core",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- INLINE CSS FOR ADVANCED STYLING ---
st.markdown("""
<style>
    /* Gradient text for the main title */
    .title-text {
        background: -webkit-linear-gradient(45deg, #14b8a6, #3b82f6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3.5rem;
        font-weight: 800;
        margin-bottom: 0px;
        text-align: left;
    }
    
    .subtitle-text {
        color: #94a3b8;
        font-size: 1.2rem;
        text-align: left;
        margin-top: 5px;
        margin-bottom: 40px;
    }

    /* Target the container of the big calculate button */
    .stButton > button {
        background: linear-gradient(135deg, #14b8a6 0%, #0ea5e9 100%) !important;
        color: white !important;
        border-radius: 8px !important;
        height: 60px !important;
        font-size: 1.3rem !important;
        font-weight: 600 !important;
        border: none !important;
        box-shadow: 0 4px 15px rgba(20, 184, 166, 0.3) !important;
        transition: all 0.3s ease !important;
    }
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(20, 184, 166, 0.5) !important;
    }
    
    /* Result metric glowing effect */
    [data-testid="stMetricValue"] {
        font-size: 4rem !important;
        color: #14b8a6 !important;
        font-weight: 900 !important;
        text-align: center !important;
        text-shadow: 0px 0px 20px rgba(20, 184, 166, 0.4) !important;
    }
    [data-testid="stMetricLabel"] {
        font-size: 1.2rem !important;
        color: #cbd5e1 !important;
        text-align: center !important;
        text-transform: uppercase !important;
        letter-spacing: 2px !important;
    }

    /* Custom stylistic hr */
    hr {
        border-top: 1px solid rgba(51, 65, 85, 0.5);
        margin-top: 2rem;
        margin-bottom: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# --- LOAD MODEL & FEATURES ---
@st.cache_resource
def load_model():
    model_path = "credit_risk_model_small.pkl"
    features_path = "features.txt"
    model, features = None, []
    if os.path.exists(model_path): model = joblib.load(model_path)
    if os.path.exists(features_path):
        with open(features_path, 'r') as f:
            features = [l.strip() for l in f.readlines() if l.strip()]
    return model, features

model, EXPECTED_FEATURES = load_model()

# --- HERO SECTION ---
st.markdown("<h1 class='title-text'>AI Risk Engine Core</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle-text'>Next-Generation Algorithm for Real-Time Loan Pricing & Risk Assessment</p>", unsafe_allow_html=True)

# --- LAYOUT: 3 Columns for Inputs to save vertical space ---
st.markdown("### Borrower Intelligence Profile")
st.markdown("<span style='color:#94a3b8; font-size: 0.95rem;'>Configure the data points below to simulate the borrower's risk profile. Our engine dynamically processes categorical and quantitative variables.</span>", unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)

c1, c2, c3 = st.columns(3, gap="large")

with c1:
    st.markdown("#### Demographics & Financials")
    age = st.slider("Primary Applicant Age", 18, 80, 35)
    gender = st.selectbox("Applicant Gender", ["Female/Other", "Male", "Unknown"], index=1)
    income_total = st.number_input("Verified Monthly Income (€)", 0, 100000, 3500, 100)
    debt_to_income = st.slider("Evaluated Debt-to-Income (%)", 0.0, 100.0, 25.0, 0.1)
    new_credit_customer = st.checkbox("First-Time Credit Applicant", value=True)

with c2:
    st.markdown("#### Employment & Residency")
    employment_duration = st.selectbox("Current Employment Tenure", ["None", "UpTo1Year", "UpTo2Years", "UpTo3Years", "UpTo4Years", "UpTo5Years", "Retiree", "TrialPeriod", "Other"], index=4)
    home_ownership = st.selectbox("Primary Residence Status", ["None", "Owner", "JointOwner", "JointTenant", "LivingWithParents", "TenantFurnished", "TenantUnfurnished", "OwnerEncumbrance", "OwnerMortgage", "Homeless", "Other"], index=1)
    education = st.selectbox("Highest Education Attained", ["None", "Higher", "Primary", "Secondary", "Vocational"], index=1)
    country = st.selectbox("Jurisdiction", ["EE", "ES", "FI", "NL", "SK"], index=0)

with c3:
    st.markdown("#### Loan Specifications & Risk")
    applied_amount = st.number_input("Requested Capital (€)", 100, 100000, 5000, 100)
    loan_duration = st.slider("Amortization Period (Months)", 1, 120, 36)
    existing_liabilities = st.number_input("Count of Active Liabilities", 0, 50, 2)
    liabilities_total = st.number_input("Total Liabilities Amount (€)", 0, 500000, 0, 100)
    amount_previous_loans = st.number_input("Amount of Previous Loans (€)", 0, 100000, 0, 100)
    rating = st.selectbox("Internal Risk Grade Category", ["None", "AA", "B", "C", "D", "E", "F", "HR"], index=2)
    verification = st.selectbox("Verification Type", ["Income Checked", "By Phone", "Not Verified", "Other Document"], index=2)

# Feature construction
def build_vector():
    # 1. Initialize all features to 0
    vector = {f: 0 for f in EXPECTED_FEATURES}
    
    # 2. Continuous variables
    if "Age" in vector: vector["Age"] = age
    if "AppliedAmount" in vector: vector["AppliedAmount"] = applied_amount
    if "Amount" in vector: vector["Amount"] = applied_amount
    if "LoanDuration" in vector: vector["LoanDuration"] = loan_duration
    if "IncomeTotal" in vector: vector["IncomeTotal"] = income_total
    if "ExistingLiabilities" in vector: vector["ExistingLiabilities"] = existing_liabilities
    if "LiabilitiesTotal" in vector: vector["LiabilitiesTotal"] = liabilities_total
    if "Debt_to_Income" in vector: vector["Debt_to_Income"] = debt_to_income
    if "AmountOfPreviousLoansBeforeLoan" in vector: vector["AmountOfPreviousLoansBeforeLoan"] = amount_previous_loans
    
    # 3. Categorical variables (One-hot mapping)
    if "NewCreditCustomer_True" in vector and new_credit_customer: vector["NewCreditCustomer_True"] = 1
    
    if gender != "Female/Other":
        col_name = f"Gender_{gender}"
        if col_name in vector: vector[col_name] = 1
        
    if verification != "Income Checked":
        col_name = f"VerificationType_{verification}"
        if col_name in vector: vector[col_name] = 1
    
    if rating != "None":
        col_name = f"Rating_{rating}"
        if col_name in vector: vector[col_name] = 1
            
    if education != "None":
        col_name = f"Education_{education}"
        if col_name in vector: vector[col_name] = 1

    if employment_duration != "None":
        # Handle exact string space matching if there is spaces, e.g. "Up To 1 Year", but features specifies "UpTo1Year"
        col_name = f"EmploymentDurationCurrentEmployer_{employment_duration}"
        if col_name in vector: vector[col_name] = 1

    if home_ownership != "None":
        col_name = f"HomeOwnershipType_{home_ownership}"
        if col_name in vector: vector[col_name] = 1
        
    if country != "EE": # EE is implicitly the baseline 0 for all other country columns as seen in features.txt
        col_name = f"Country_{country}"
        if col_name in vector: vector[col_name] = 1

    # Print for debugging only in development
    # print(pd.DataFrame([vector])[EXPECTED_FEATURES])

    return pd.DataFrame([vector])[EXPECTED_FEATURES]

# --- INFERENCE TERMINAL ---
st.markdown("---")
st.markdown("<h3 style='text-align: center;'>Inference Terminal</h3>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color:#94a3b8; font-size: 0.95rem;'>Initialize the prediction model mapping inputs to expected interest rates.</p>", unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)

_, btn_col, _ = st.columns([1, 1.5, 1])

with btn_col:
    predict_clicked = st.button("CALCULATE RISK YIELD", use_container_width=True)
    
st.markdown("<br>", unsafe_allow_html=True)

# Result area
if predict_clicked:
    if model is None:
        st.error("SYSTEM HALT: `credit_risk_model_small.pkl` not detected in memory.")
    elif not EXPECTED_FEATURES:
        st.error("SYSTEM HALT: Feature structure map `features.txt` is missing.")
    else:
        with st.spinner("Executing neural operations..."):
            try:
                df_input = build_vector()
                pred = model.predict(df_input)[0]
                _, res_col, _ = st.columns([1, 1.5, 1])
                with res_col:
                    st.success("Analysis Successfully Finished")
                    st.markdown("<div style='padding: 20px; background-color: rgba(20, 184, 166, 0.05); border-radius: 12px; border: 1px solid rgba(20, 184, 166, 0.2);'>", unsafe_allow_html=True)
                    st.metric("Predicted Annual Yield", f"{pred:.2f}%")
                    st.markdown("</div>", unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Inference failure: {e}")

# Footer
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("---")
st.markdown("<div style='text-align: center; color: #475569; font-size: 0.9rem; font-family: monospace;'>Built by <b>Diaa Shousha | GeoAi Engineer</b> <br><br> SECURE NODE • AES-256 ENCRYPTED IN-MEMORY ENGINE</div>", unsafe_allow_html=True)
