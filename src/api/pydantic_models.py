from pydantic import BaseModel


<<<<<<< HEAD
=======

>>>>>>> 9a8bc4d0ac6b56f1af791ee9a20c527797ae612b
class CreditRiskRequest(BaseModel):
    age: float
    income: float
    loan_amount: float
    # add other numeric features here


class CreditRiskResponse(BaseModel):
    risk_probability: float
