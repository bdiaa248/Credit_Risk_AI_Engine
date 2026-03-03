import joblib
import pandas as pd

model = joblib.load('credit_risk_model_small.pkl')

with open('features.txt', 'r') as f:
    features = [line.strip() for line in f.readlines() if line.strip()]

def predict_scenario(scenario_name, **kwargs):
    vector = {f: 0 for f in features}
    
    # Defaults based on our app
    vector["Age"] = 35
    vector["AppliedAmount"] = 5000
    vector["Amount"] = 5000
    vector["LoanDuration"] = 36
    vector["IncomeTotal"] = 3500
    vector["ExistingLiabilities"] = 2
    vector["Debt_to_Income"] = 25.0
    vector["NewCreditCustomer_True"] = 1
    
    # Setup categorical baseline (App defaults)
    vector["Rating_B"] = 1
    vector["Education_Higher"] = 1
    vector["EmploymentDurationCurrentEmployer_UpTo4Years"] = 1
    vector["HomeOwnershipType_Owner"] = 1
    # Country EE is 0 for all
    
    # Overrides
    for k, v in kwargs.items():
        if k in vector:
            vector[k] = v
            
        # Handle special one-hot overrides
        if k.startswith("Rating_"):
            # clear others
            for feat in features:
                if feat.startswith("Rating_"): vector[feat] = 0
            vector[k] = 1
            
        if k.startswith("Country_"):
            for feat in features:
                if feat.startswith("Country_"): vector[feat] = 0
            if k != "Country_EE":
                vector[k] = 1

    df = pd.DataFrame([vector])[features]
    pred = model.predict(df)[0]
    print(f"{scenario_name:30s} -> Predicted Interest Rate: {pred:.2f}%")

print("--- TESTING DIFFERENT PROFILES ---")

predict_scenario("Baseline (Default App Profile)")

predict_scenario("Very Risky Profile", 
                 Rating_HR=1, 
                 IncomeTotal=1000, 
                 AppliedAmount=20000, 
                 Amount=20000,
                 ExistingLiabilities=10,
                 Debt_to_Income=70.0)

predict_scenario("Extremely Safe Profile", 
                 Rating_AA=1, 
                 IncomeTotal=15000, 
                 AppliedAmount=1000, 
                 Amount=1000,
                 ExistingLiabilities=0,
                 Debt_to_Income=5.0)

predict_scenario("Medium Risk (Rating D)", Rating_D=1)

predict_scenario("High Amount Requested", AppliedAmount=50000, Amount=50000)

predict_scenario("Different Country (FI)", Country_FI=1)

