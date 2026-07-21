import os

# Disable MLflow background tasks and DB initialization
os.environ["MLFLOW_TRACKING_DISABLE"] = "true"
os.environ["MLFLOW_SKINNY"] = "true"
os.environ["MLFLOW_DISABLE_ENV_CREATION"] = "true"
os.environ["MLFLOW_ENABLE_TRACING"] = "false"
os.environ["MLFLOW_TRACKING_URI"] = "sqlite:///:memory:"
os.environ["MLFLOW_HOME"] = "/tmp/mlflow"

import json
from datetime import datetime, timezone
import pandas as pd
import mlflow

model = None


def find_mlmodel_dir(base_path):
    """Recursively search for the directory containing the MLmodel file."""
    for root, dirs, files in os.walk(base_path):
        if "MLmodel" in files:
            return root
    return base_path


def init():
    global model

    try:
        mlflow.autolog(disable=True)
    except Exception:
        pass

    base_dir = os.getenv("AZUREML_MODEL_DIR", "")
    
    # Dynamically locate folder containing MLmodel (handles credit_risk_model-2, etc.)
    model_path = find_mlmodel_dir(base_dir)
    print(f"Loading model from resolved path: {model_path}")

    # Load via pyfunc for standard inference interface
    try:
        model = mlflow.pyfunc.load_model(model_path)
    except Exception as e:
        print(f"Failed to load as pyfunc: {e}, attempting native xgboost load...")
        model = mlflow.xgboost.load_model(model_path)


def run(raw_data):
    if model is None:
        return [{"error": "Model was not initialized properly."}]

    try:
        # Parse payload
        input_json = json.loads(raw_data) if isinstance(raw_data, str) else raw_data
        payload = input_json.get("input_data", {})
        columns = payload.get("columns")
        data_matrix = payload.get("data")

        # Build DataFrame
        df = pd.DataFrame(data=data_matrix, columns=columns)

        # Get predictions / default probabilities
        if hasattr(model, "predict_proba"):
            # Native scikit-learn / XGBClassifier
            probabilities = model.predict_proba(df)
            default_probability = float(probabilities[0][1])
        elif hasattr(model, "unwrap_python_model"):
            # PyFunc flavor: predict often outputs raw probabilities directly for classifiers
            preds = model.predict(df)
            default_probability = float(preds[0]) if preds.ndim == 1 else float(preds[0][1])
        else:
            preds = model.predict(df)
            default_probability = float(preds[0])

        risk_score = round(default_probability * 100)

        if risk_score < 30:
            decision = "APPROVE"
        elif risk_score < 70:
            decision = "REFER"
        else:
            decision = "DECLINE"

        confidence = round(max(default_probability, 1 - default_probability), 2)

        return [
            {
                "risk_score": risk_score,
                "decision": decision,
                "confidence": confidence,
                "model_version": "v1.2.0",
                "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            }
        ]

    except Exception as e:
        return [{"error": str(e)}]
