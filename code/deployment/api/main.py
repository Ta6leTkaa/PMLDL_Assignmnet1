"""Prediction API for the packaged California Housing model."""
import os
import hashlib
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict, Field, model_validator


class HousingInput(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    longitude: float = Field(ge=-180, le=180)
    latitude: float = Field(ge=-90, le=90)
    housing_median_age: float = Field(ge=0)
    total_rooms: float = Field(ge=0)
    total_bedrooms: float = Field(ge=0)
    population: float = Field(ge=0)
    households: float = Field(gt=0)
    median_income: float = Field(ge=0)
    ocean_proximity: Literal["<1H OCEAN", "INLAND", "NEAR OCEAN", "NEAR BAY", "ISLAND"]

    @model_validator(mode="after")
    def validate_rooms(self):
        if self.total_bedrooms > self.total_rooms:
            raise ValueError("total_bedrooms cannot exceed total_rooms")
        return self


class Prediction(BaseModel):
    median_house_value: float
    currency: str = "USD"


@asynccontextmanager
async def lifespan(app):
    configured = os.environ.get("MODEL_PATH")
    path = Path(configured) if configured else Path(__file__).resolve().parents[3] / "models/model.joblib"
    if not path.exists():
        raise RuntimeError(f"Model missing at {path}; run model training first")
    app.state.model = joblib.load(path)
    app.state.model_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
    yield


app = FastAPI(title="California Housing API", lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": True, "model_sha256": app.state.model_sha256}


@app.post("/predict", response_model=Prediction)
def predict(data: HousingInput):
    frame = pd.DataFrame([data.model_dump()])
    value = float(app.state.model.predict(frame)[0])
    return Prediction(median_house_value=round(value, 2))
