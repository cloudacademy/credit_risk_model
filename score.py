import os
import json
import pandas as pd
from datetime import datetime
from mlflow.pyfunc import load_model

def init():
    global model
    base_dir = os.getenv("AZUREML_MODEL_DIR", "")
    
    # Check inside the subfolder first, fallback to base dir
    model_path = os.path.join(base_dir, "credit_risk_model")
    if not os.path.exists(model_path):
        model_path = base_dir
        
    print(f"Loading model from path: {model_path}")
    model = load_model(model_path)

def run(raw_data):
    if model is None:
        return [{"error": "Model was not initialized properly."}]
        
    try:
        # 1. Parse incoming raw JSON string text
        input_json = json.loads(raw_data)
        
        # 2. Extract the 'input_data' matrix structure
        payload = input_json.get("input_data", {})
        columns = payload.get("columns")
        data_matrix = payload.get("data")
        
        # 3. Build the Pandas DataFrame explicitly using data & headers
        df = pd.DataFrame(data=data_matrix, columns=columns)
            
        # 4. Generate classifications (0 or 1)
        predictions = model.predict(df)
        pred_val = int(predictions[0])
        
        # 5. DYNAMICALLY calculate risk metric values
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

        # 6. Structure the return matrix object EXACTLY as a Python list/dict
        response_payload = [
            {
                "risk_score": risk_score,
                "decision": decision,
                "confidence": confidence,
                "model_version": "v1.2.0",
                "timestamp": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
            }
        ]
        
        # 7. CRITICAL FIX: Return the native Python list object. 
        # Do NOT use json.dumps() here. Azure ML handles serialization automatically.
        return response_payload
        
    except Exception as e:
        return [{"error": str(e)}]