from fastapi import APIRouter
import numpy as np
from tensorflow.keras.models import load_model

router = APIRouter()

# Load models once
res_model = load_model("models/saved/residential_lstm.h5", compile=False)
com_model = load_model("models/saved/commercial_lstm.h5", compile=False)
ind_model = load_model("models/saved/industrial_lstm.h5", compile=False)


def predict_next(model):
    dummy_input = np.random.rand(1, 24, 1)
    pred = model.predict(dummy_input)
    return float(pred[0][0])


@router.get("/forecast/residential")
def residential():
    return {
        "sector": "residential",
        "next_hour_forecast": predict_next(res_model)
    }


@router.get("/forecast/commercial")
def commercial():
    return {
        "sector": "commercial",
        "next_hour_forecast": predict_next(com_model)
    }


@router.get("/forecast/industrial")
def industrial():
    return {
        "sector": "industrial",
        "next_hour_forecast": predict_next(ind_model)
    }
