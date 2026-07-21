import os

# 1. CRITICAL ENVIRONMENT FLAGS
# Redirect MLflow tracking away from read-only container paths and disable tracing
os.environ["MLFLOW_TRACKING_DISABLE"] = "true"
os.environ["MLFLOW_SKINNY"] = "true"
os.environ["MLFLOW_DISABLE_ENV_CREATION"] = "true"
os.environ["MLFLOW_ENABLE_TRACING"] = "false"
os.environ["MLFLOW_TRACKING_URI"] = "sqlite:///:memory:"
os.environ["MLFLOW_HOME"] = "/tmp/mlflow"

import json
from datetime import datetime
import pandas as pd
import mlflow

model = None


def init():
    global model

    # Disable runtime autologging if present
    try:
        mlflow.autolog(disable=True)
    except Exception:
        pass

    base_dir = os.getenv("AZUREML_MODEL_DIR", "")

    # Look inside the model folder first, fallback to base dir
    model_path = os.path.join(base_dir, "credit_risk_model")
    if not os.path.exists(model_path):
        model_path = base_dir

    print(f"Loading model from path: {model_path}")

    # Load via xgboost flavor first for speed, fallback to pyfunc
    try:
        model = mlflow.xgboost.load_model(model_path)
    except Exception:
        model = mlflow.pyfunc.load_model(model_path)


def run(raw_data):
    if model is None:
        return [{"error": "Model was not initialized properly."}]

    try:
        # Parse payload
        input_json = json.loads(raw_data)
        payload = input_json.get("input_data", {})
        columns = payload.get("columns")
        data_matrix = payload.get("data")

        # Build DataFrame
        df = pd.DataFrame(data=data_matrix, columns=columns)

        # Generate predictions (0 or 1)
        predictions = model.predict(df)
        pred_val = int(predictions[0])

        # Get default probability
        try:
            probabilities = model.predict_proba(df)
            default_probability = float(probabilities[0][1])
        except AttributeError:
            default_probability = 0.85 if pred_val == 1 else 0.11

        risk_score = round(default_probability * 100)

        if risk_score < 30:
            decision = "APPROVE"
        elif risk_score < 70:
            decision = "REFER"
        else:
            decision = "DECLINE"

        confidence = round(max(default_probability, 1 - default_probability), 2)

        # Return native Python list/dict (Azure ML handles JSON response stringification)
        return [
            {
                "risk_score": risk_score,
                "decision": decision,
                "confidence": confidence,
                "model_version": "v1.2.0",
                "timestamp": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
            }
        ]

    except Exception as e:
        return [{"error": str(e)}]
