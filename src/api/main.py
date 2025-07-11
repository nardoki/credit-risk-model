from fastapi import FastAPI, HTTPException
import mlflow.pyfunc
from src.api.pydantic_models import CreditRiskRequest, CreditRiskResponse
import pandas as pd

app = FastAPI()

MODEL_NAME = "random_forest"   
MODEL_VERSION = "1"          

model = mlflow.pyfunc.load_model(r"mlruns\975347351455683709\models\m-add428fdfa0149d48a24387cb8a1f5e6\artifacts")

<<<<<<< HEAD
=======
model = mlflow.pyfunc.load_model(r"mlruns\975347351455683709\models\m-add428fdfa0149d48a24387cb8a1f5e6\artifacts")

>>>>>>> 9a8bc4d0ac6b56f1af791ee9a20c527797ae612b

@app.post("/predict", response_model=CreditRiskResponse)

def predict_risk(request: CreditRiskRequest):
    input_data = pd.DataFrame([request.dict()])
    try:
        prob = model.predict_proba(input_data)[0][1]
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction error: {e}")
    return CreditRiskResponse(risk_probability=prob)
