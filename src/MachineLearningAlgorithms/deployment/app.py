# app.py

from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

# Load model
model = joblib.load("no_show_model.pkl")

# Define request schema
class AppointmentData(BaseModel):
    Age: int
    Gender: int  # 0: Female, 1: Male
    Scholarship: int
    Diabetes: int
    Alcoholism: int
    Hypertension: int
    SMS_received: int
    WaitingDays: int

app = FastAPI()

@app.post("/predict")
def predict(data: AppointmentData):
    df = pd.DataFrame([data.dict()])
    prediction = model.predict(df)[0]
    return {
        "no_show_prediction": int(prediction),
        "interpretation": "Likely to No-Show" if prediction == 1 else "Likely to Attend"
    }
