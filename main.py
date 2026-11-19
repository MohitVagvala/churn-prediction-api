from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

# Load the model ONCE when the server starts, not on every request
bundle = joblib.load("churn_model.joblib")
pipeline = bundle["pipeline"]
THRESHOLD = bundle["threshold"]

YesNo = Literal["Yes", "No"]
InternetExtra = Literal["Yes", "No", "No internet service"]

EXAMPLE = {
    "gender": "Female", "SeniorCitizen": 0, "Partner": "No", "Dependents": "No",
    "tenure": 2, "PhoneService": "Yes", "MultipleLines": "No",
    "InternetService": "Fiber optic", "OnlineSecurity": "No", "OnlineBackup": "No",
    "DeviceProtection": "No", "TechSupport": "No", "StreamingTV": "Yes",
    "StreamingMovies": "Yes", "Contract": "Month-to-month", "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check", "MonthlyCharges": 95.0, "TotalCharges": 190.0,
}

class Customer(BaseModel):
    gender: Literal["Female", "Male"]
    SeniorCitizen: int = Field(ge=0, le=1)
    Partner: YesNo
    Dependents: YesNo
    tenure: int = Field(ge=0, description="Months with the company")
    PhoneService: YesNo
    MultipleLines: Literal["Yes", "No", "No phone service"]
    InternetService: Literal["DSL", "Fiber optic", "No"]
    OnlineSecurity: InternetExtra
    OnlineBackup: InternetExtra
    DeviceProtection: InternetExtra
    TechSupport: InternetExtra
    StreamingTV: InternetExtra
    StreamingMovies: InternetExtra
    Contract: Literal["Month-to-month", "One year", "Two year"]
    PaperlessBilling: YesNo
    PaymentMethod: Literal["Electronic check", "Mailed check",
                           "Bank transfer (automatic)", "Credit card (automatic)"]
    MonthlyCharges: float = Field(ge=0)
    TotalCharges: float = Field(ge=0)

    model_config = {"json_schema_extra": {"examples": [EXAMPLE]}}

class Prediction(BaseModel):
    churn_probability: float
    will_churn: bool
    threshold: float

app = FastAPI(title="Churn Predictor API", version="1.0")

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict", response_model=Prediction)
def predict(customer: Customer):
    df = pd.DataFrame([customer.model_dump()])
    p = float(pipeline.predict_proba(df)[0, 1])
    return Prediction(churn_probability=round(p, 3), will_churn=p >= THRESHOLD, threshold=THRESHOLD)

import sklearn, xgboost, sys
print("scikit-learn:", sklearn.__version__)
print("xgboost:", xgboost.__version__)
print("python:", sys.version)




