"""
FastAPI app for predicting penguin species using XGBoost.
Includes:
- Pydantic Enums and response models
- Input validation
- Proper one-hot encoding
- Logging
- Health and prediction endpoints
"""

import os
import xgboost as xgb
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from enum import Enum
from typing import Any
import logging

# ---------------- Logging Setup ----------------
logger = logging.getLogger(__name__)
logging.basicConfig(filename='example.log', encoding='utf-8', level=logging.DEBUG)

# ---------------- Constants --------------------
MODEL_PATH = os.path.join(os.path.dirname(__file__), "data", "model.json")
LABEL_ENCODER_PATH = os.path.join(os.path.dirname(__file__), "data", "label_encoder_classes.json")

# ---------------- FastAPI App ------------------
app = FastAPI()

# ---------------- Enums and Request Model ----------------
class Island(str, Enum):
    Torgersen = "Torgersen"
    Biscoe = "Biscoe"
    Dream = "Dream"

class Sex(str, Enum):
    Male = "male"
    Female = "female"

class PenguinFeatures(BaseModel):
    bill_length_mm: float
    bill_depth_mm: float
    flipper_length_mm: float
    body_mass_g: float
    sex: Sex
    island: Island

class PredictionResponse(BaseModel):
    prediction: str

class HealthResponse(BaseModel):
    status: str

# ---------------- Load Model ----------------
def load_model() -> xgb.XGBClassifier:
    """Load the trained XGBoost model from disk."""
    logger.info("Loading model from %s", MODEL_PATH)
    model = xgb.XGBClassifier()
    model.load_model(MODEL_PATH)
    logger.info("Model loaded successfully.")
    return model

model = load_model()

# ---------------- Preprocessing ----------------
def preprocess_features(features: PenguinFeatures) -> pd.DataFrame:
    """Preprocess input features into model-ready DataFrame."""
    input_dict: dict[str, Any] = features.model_dump()
    df = pd.DataFrame([input_dict])

    # One-hot encode with known values
    df = pd.get_dummies(df, columns=["sex", "island"])

    expected_cols = [
        "bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g",
        "sex_Female", "sex_Male", "island_Biscoe", "island_Dream", "island_Torgersen"
    ]
    for col in expected_cols:
        if col not in df.columns:
            df[col] = 0
    df = df[expected_cols]
    return df

# ---------------- Routes ----------------
@app.get("/", response_model=HealthResponse)
async def root() -> HealthResponse:
    """Root health check endpoint."""
    return HealthResponse(status="Penguin Predictor API is running.")

@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Health check endpoint."""
    return HealthResponse(status="ok")

@app.post("/predict", response_model=PredictionResponse)
async def predict(features: PenguinFeatures) -> PredictionResponse:
    """Predict the penguin species from user input features."""
    try:
        logger.debug(f"Received input: {features}")
        X_input = preprocess_features(features)
        prediction = model.predict(X_input)[0]

        # Convert to species label
        if os.path.exists(LABEL_ENCODER_PATH):
            label_df = pd.read_json(LABEL_ENCODER_PATH)
            class_names = label_df["species"].tolist()
            class_label = class_names[prediction]
        else:
            class_label = str(prediction)

        logger.info(f"Prediction successful: {class_label}")
        return PredictionResponse(prediction=str(class_label))

    except Exception as e:
        logger.exception("Prediction failed.")
        raise HTTPException(status_code=400, detail=str(e))
