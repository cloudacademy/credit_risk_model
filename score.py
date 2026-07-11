import os
import json
from mlflow.pyfunc import load_model

def init():
    global model
    
    # 1. Get the base model directory provided by Azure
    base_dir = os.getenv("AZUREML_MODEL_DIR", "")
    
    # 2. Try your explicit path first
    model_path = os.path.join(base_dir, "credit_risk_model")
    
    # 3. Fallback: If that folder doesn't exist, look directly in the base directory
    if not os.path.exists(model_path):
        model_path = base_dir

    print(f"Loading model from path: {model_path}")
    model = load_model(model_path)

def run(raw_data):
    # Ensure the model actually loaded in init()
    if model is None:
        return json.dumps({"error": "Model was not initialized properly. Check init logs."})
        
    try:
        data = json.loads(raw_data)
        
        # If your data comes wrapped in a "data" key, extract it
        if isinstance(data, dict) and "data" in data:
            data = data["data"]
            
        predictions = model.predict(data)
        return json.dumps({"predictions": predictions.tolist()})
        
    except Exception as e:
        return json.dumps({"error": str(e)})