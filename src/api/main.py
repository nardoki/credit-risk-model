from fastapi import FastAPI, HTTPException
import mlflow.pyfunc
from src.api.pydantic_models import CreditRiskRequest, CreditRiskResponse
import pandas as pd

app = FastAPI()

MODEL_NAME = "random_forest"   # Replace with your registered model name
MODEL_VERSION = "1"            # Use the correct version

model = mlflow.pyfunc.load_model("mlruns\975347351455683709\models\m-add428fdfa0149d48a24387cb8a1f5e6\artifacts")

@app.post("/predict", response_model=CreditRiskResponse)
def predict_risk(request: CreditRiskRequest):
    input_data = pd.DataFrame([request.dict()])
    try:
        prob = model.predict_proba(input_data)[0][1]
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction error: {e}")
    return CreditRiskResponse(risk_probability=prob)
