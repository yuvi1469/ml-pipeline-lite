
from pydantic import BaseModel, Field
from typing import List

class PredictRequest(BaseModel):
    features: List[float] = Field(..., description="30 numeric features in original order")

class PredictResponse(BaseModel):
    probability: float
    prediction: int
