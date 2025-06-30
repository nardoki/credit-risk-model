from pydantic import BaseModel



class CreditRiskRequest(BaseModel):
    age: float
    income: float
    loan_amount: float
    # add other numeric features here


class CreditRiskResponse(BaseModel):
    risk_probability: float
