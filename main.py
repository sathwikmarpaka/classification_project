
import pandas as pd
import joblib
from pathlib import Path

from fastapi import FastAPI, HTTPException
from model_components import BrakeFeatureEngineer


# --------------------------------
# 1. Configuration
# --------------------------------

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "brake_alert_model.joblib"

# --------------------------------
# 2. Create FastAPI Application
# --------------------------------

app = FastAPI(
    title="Brake Alert Prediction API",
    description="Predicts whether a vehicle requires a brake service alert.",
    version="1.0"
)


# --------------------------------
# 3. Load Trained Model
# --------------------------------

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Model file not found: {MODEL_PATH}"
    )

best_model = joblib.load(MODEL_PATH)

print("Trained model loaded successfully!")


# --------------------------------
# 4. Home Route
# --------------------------------

@app.get("/")
def home():
    return {
        "message": "Brake Alert API is running!",
        "status": "success"
    }


# --------------------------------
# 5. Health Check
# --------------------------------

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": True
    }


# --------------------------------
# 6. Prediction Route
# --------------------------------

@app.post("/predict")
def predict(data: dict):

    try:
        # Convert incoming JSON into DataFrame
        input_data = pd.DataFrame([data])

        # Get expected feature names from trained model
        expected_features = getattr(
            best_model,
            "feature_names_in_",
            None
        )

        if expected_features is not None:
            missing_features = [
                feature for feature in expected_features
                if feature not in input_data.columns
            ]

            if missing_features:
                raise HTTPException(
                    status_code=422,
                    detail={
                        "message": "Missing required input features",
                        "missing_features": missing_features
                    }
                )

            input_data = input_data[list(expected_features)]

        # Predict service alert
        prediction = best_model.predict(input_data)[0]

        # Predict probability
        probabilities = best_model.predict_proba(input_data)[0]

        classes = list(best_model.classes_)
        alert_index = classes.index(1)

        alert_probability = float(
            probabilities[alert_index]
        )

        return {
            "prediction": int(prediction),
            "service_alert": bool(prediction == 1),
            "alert_probability": round(
                alert_probability, 4
            ),
            "message": (
                "Service Alert Required"
                if prediction == 1
                else "No Service Alert Required"
            )
        }

    except HTTPException:
        raise

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )
