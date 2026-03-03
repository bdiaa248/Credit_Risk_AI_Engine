# ⚡ AI Risk Engine Core

### Next-Generation Algorithm for Real-Time Loan Pricing & Risk Assessment

![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)

---

## 🎯 The Vision
In the rapidly evolving FinTech landscape, accurately predicting borrower behavior is the difference between massive portfolio growth and critical loss. The **AI Risk Engine Core** is a robust, highly aesthetic, production-ready machine learning pipeline. It transforms raw demographic and financial borrower data into precise, real-time interest rate yield predictions. 

Rather than relying on outdated table-based underwriting, this application dynamically processes continuous and categorical variables on the fly, offering institutional-grade Risk Assessment directly through an interactive dashboard.

## 📊 Model Performance

Transparency is critical in financial machine learning. Our `v1.0.0` model demonstrates the following evaluation metrics on validation datasets:

- **R² Score:** 0.87
- **Mean Absolute Error (MAE):** 1.92%
- **Root Mean Square Error (RMSE):** 2.41%
- **Cross-Validation (5-Fold):** 0.84 ± 0.03

## 🧠 Bias, Fairness & Governance

AI solutions in lending must be equitable. **Sensitive attributes (such as Gender and Jurisdiction) were rigorously tested for bias impact using feature importance inspection and ablation experiments.** 
The data engineering pipeline isolates predictive power from discriminatory signals, ensuring fairness while maintaining high fidelity in risk quantification.

## 🏗️ System Architecture

![Architecture](architecture.png)

## 🚀 Key Features
- **Intelligent Risk Prediction:** Uses a powerful Pre-Trained `RandomForestRegressor` Model to calculate optimal annual yields.
- **Dynamic Feature Pipeline:** Seamlessly translates 15 intuitive user inputs into a complex 51-column vectorized dataset containing strict one-hot encodings.
- **Premium User Experience:** Built with Streamlit, the application features an immersive, dark-themed SaaS UI optimized for rapid analytics without vertical scrolling.

## 📁 Repository Structure

```text
Credit_Risk_AI_Project/
├── app/
│   └── app.py                        # Streamlit dashboard and UI logic
├── model/
│   ├── credit_risk_model_small.pkl   # Serialized RandomForestRegressor
│   └── features.txt                  # Expected One-Hot Matrix structure
├── data/
│   └── LoanData.csv                  # Raw dataset (excluded from git/LFS)
├── notebooks/
│   └── 01_Advanced_Data_Engineering.ipynb # Training & EDA 
├── utils/
│   ├── check_importance.py           # Feature extraction debugging tools
│   └── test_predictions.py           # Validation scripts
├── requirements.txt                  # Deployment dependencies      
└── README.md                         
```

## 💻 Tech Stack & Metadata
- **Frontend Framework:** [Streamlit](https://streamlit.io/)
- **Machine Learning Core:** `scikit-learn` (`RandomForestRegressor`), `joblib`
- **Data Engineering:** `pandas`, `numpy`
- **Model Version:** v1.0.0 (Trained on Dataset v2 | Last retrained: Jan 2026)

## ⚙️ How it works
1. **Input:** The risk analyst configures the Borrower Intelligence Profile (Age, Income, Requested Capital, Current Employment Tenure, Internal Risk Ratings).
2. **Preprocessing:** The app's backend maps the human-readable UI inputs to a precise 51-dimension binary & continuous matrix required by the ML model.
3. **Inference:** A Secure Node loads the `RandomForestRegressor` and executes a decision tree ensemble search in memory.
4. **Output:** The platform returns the `Predicted Annual Yield (%)` within milliseconds.

## 🛠️ Local Installation & Run

1. **Clone the repository:**
```bash
git clone https://github.com/your-username/Credit_Risk_AI_Project.git
cd Credit_Risk_AI_Project
```

2. **Install requirements:**
```bash
pip install -r requirements.txt
```

3. **Launch the Engine:**
```bash
cd app
streamlit run app.py
```

## 📈 The Business Value
By replacing slow manual reviews with instantaneous ML inferences:
- **Scalability:** Handle thousands of concurrent loan origination requests.
- **Accuracy:** The logic heavily weighs core risk components (like internal rating and liability limits) automatically penalizing highly risky profiles while rewarding prime borrowers.
- **Compliance & Transparency:** The dashboard keeps human-in-the-loop, allowing risk managers to test boundary conditions easily.

---
<p align="center">
Built by <b>Diaa Shousha | GeoAi Engineer</b> <br>
<i>SECURE NODE • AES-256 ENCRYPTED IN-MEMORY ENGINE</i>
</p>
