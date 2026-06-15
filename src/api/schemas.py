from pydantic import BaseModel
from typing import List, Dict

class ForecastOverrideRequest(BaseModel):
    sector: str  # 'residential', 'commercial', 'industrial'
    air_temperature: float
    wind_speed: float
    cloud_coverage: float

class ForecastOverrideResponse(BaseModel):
    sector: str
    air_temperature: float
    wind_speed: float
    cloud_coverage: float
    baseline_prediction: float
    xgb_prediction: float
    lstm_prediction: float
    hybrid_prediction: float
