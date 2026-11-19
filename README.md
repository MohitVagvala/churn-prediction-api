# Customer Churn Prediction API

Predicts which telecom customers are likely to leave, served as a live REST API.

**Live API docs:** https://churn-prediction-api-s4ef.onrender.com/docs
*(Free hosting, cold start 50 secs)*

## Results
| Model | CV ROC-AUC |
|---|---|
| Baseline (always "stays") | 0.50 |
| Logistic regression (shipped) | 0.846 ± 0.013 |
| Random forest | 0.845 ± 0.009 |
| XGBoost (tuned) | 0.844 ± 0.009 |

- Test ROC-AUC: 0.842
- Decision threshold **0.35** (instead of 0.5): missing a churner costs more than a false alarm, so the model favors recall. Churn recall: 

## Key findings from the data
- Month-to-month customers churn at **42.7%**, against **2.8%** on two-year contracts
- Churners had stayed **18 months** on average, against **38** for loyal customers
- 11 blank `TotalCharges` values belonged to brand-new customers (tenure 0), so they were filled with 0, not the mean

## How it works
1. **Data:** IBM Telco Customer Churn dataset (7,043 customers, 26.5% churn, imbalanced)
2. **Pipeline:** scikit-learn `ColumnTransformer` (scaling + one-hot encoding) inside a `Pipeline`, which prevents data leakage
3. **Model selection:** 5-fold stratified cross-validation, with the test set used once at the end
4. **Why logistic regression:** it matched the tree models within noise, is interpretable, and avoided a fragile XGBoost dependency at deploy time
5. **API:** FastAPI with Pydantic validation (bad input returns a 422 error before it reaches the model) and a `/health` endpoint

## Run locally
    pip install -r requirements.txt
    uvicorn main:app --reload
Then open http://127.0.0.1:8000/docs

## Tech stack
Python, pandas, scikit-learn, FastAPI, Pydantic, Uvicorn, Render
