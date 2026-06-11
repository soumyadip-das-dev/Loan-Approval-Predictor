# Loan Approval Prediction System

An end-to-end machine learning pipeline that predicts loan approval probability from financial and demographic data. Built with XGBoost, Scikit-Learn, and Streamlit — delivering approval probabilities, risk classification, and personalized financial insights.

**[Live Demo →](#)** *https://loans-approval-prediction.streamlit.app*

---

## Overview

Financial institutions weigh dozens of variables before approving a loan. This project automates that evaluation using a calibrated XGBoost classifier trained on real-world financial features — giving applicants a transparent, probability-driven decision alongside actionable insights.

**Input features:** Age, Annual Income, Employment Experience, Home Ownership, Loan Amount, Loan Purpose, Interest Rate, Credit Score

**Model outputs:**
- Approval & rejection probabilities
- Risk category (Low / Moderate / High)
- Personalized financial insights based on applicant profile

---

## Model Performance

| Metric | Score |
|---|---|
| Accuracy | 87.72% |
| ROC-AUC | 0.9217 |

**Top predictive features:** Loan-to-Income Ratio · Interest Rate · Annual Income · Credit Score · Home Ownership · Loan Purpose

---

## Tech Stack

| Layer | Tools |
|---|---|
| ML | Python, XGBoost, Scikit-Learn, Pandas, NumPy |
| Frontend | Streamlit, Custom CSS |
| Deployment | Streamlit Community Cloud |

---

## ML Workflow

**1. Preprocessing** — Missing value handling, feature scaling (StandardScaler), categorical encoding (OneHotEncoder), train-test split

**2. Feature Engineering** — Three derived features added:
- Loan-to-Income Ratio
- Employment Experience Ratio
- Estimated Credit History Length

**3. Model Selection** — Trained and compared Logistic Regression, Decision Tree, Random Forest, and XGBoost; XGBoost selected as production model

**4. Production Pipeline** — Scikit-Learn pipeline with OneHotEncoder + StandardScaler + Calibrated XGBoost, serialized to `loan_pipeline_v2.pkl`

---

## Project Structure

```
Loan-Approval-Predictor/
├── app/
│   ├── app.py
│   └── style.css
├── data/
│   └── loan_data.csv
├── models/
│   ├── loan_pipeline.pkl
│   └── loan_pipeline_v2.pkl
├── notebooks/
│   └── loan_approval.ipynb
├── requirements.txt
└── README.md
```

---

## Example Prediction

| Feature | Value |
|---|---|
| Age | 35 |
| Annual Income | ₹3,00,000 |
| Credit Score | 820 |
| Loan Amount | ₹10,000 |
| Home Ownership | Own |

**Result:** 98.85% Approval Probability · 1.15% Rejection · **Low Risk**

---

## Setup & Installation

```bash
# Clone the repository
git clone https://github.com/your-username/loan-approval-predictor.git
cd loan-approval-predictor

# Create and activate virtual environment
python -m venv .venv

# Windows
.venv\Scripts\activate
# Linux / macOS
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Run Locally

```bash
cd app
streamlit run app.py
# Opens at http://localhost:8501
```

---

## Deployment

Deployed on Streamlit Community Cloud:

1. Push the repository to GitHub
2. Connect the repo at [share.streamlit.io](https://share.streamlit.io)
3. Set **App file** to `app/app.py`, branch to `main`
4. Deploy

---

## Author

**Soumyadip Das** — AI/ML Engineer & Full-Stack Developer · B.Tech (AI & ML), NIMS University, Jaipur

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue)](https://linkedin.com/in/soumyadip-das-4201931b5)
[![GitHub](https://img.shields.io/badge/GitHub-Profile-black)](https://github.com/soumyadip-das-dev)
