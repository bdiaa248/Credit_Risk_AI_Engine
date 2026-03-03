import joblib
import pandas as pd
import numpy as np

model = joblib.load('credit_risk_model_small.pkl')
importances = model.feature_importances_

with open('features.txt', 'r') as f:
    features = [line.strip() for line in f.readlines() if line.strip()]

feat_imp = pd.Series(importances, index=features).sort_values(ascending=False)
print("Top 15 Feature Importances:")
print(feat_imp.head(15))
