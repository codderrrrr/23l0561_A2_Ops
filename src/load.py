import os
import mlflow
import numpy as np
from PIL import Image
from dotenv import load_dotenv

load_dotenv()

os.environ["MLFLOW_TRACKING_USERNAME"] = os.getenv("DAGSHUB_USERNAME") # type: ignore
os.environ["MLFLOW_TRACKING_PASSWORD"] = os.getenv("DAGSHUB_TOKEN") # type: ignore
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI")) # type: ignore

client = mlflow.MlflowClient()
model_name = "VisionGuard_Model"
version = client.get_model_version_by_alias(name=model_name, alias="champion")
threshold = client.get_run(version.run_id).data.metrics.get("best_threshold", 0.5) # type: ignore
model = mlflow.keras.load_model(f"models:/{model_name}@champion") # type: ignore

def classify_image(image_path):
    img = Image.open(image_path).convert('RGB').resize((224, 224))
    img_array = np.expand_dims(np.array(img, dtype=np.float32), axis=0)
    prob = float(model.predict(img_array, verbose=0)[0][0]) # type: ignore
    
    if prob > threshold:
        return "Fake (AI-Generated)", prob
    return "Real", 1.0 - prob

if __name__ == "__main__":
    label, conf = classify_image("tests/test_Real.jpg")
    print(f"Verdict: {label} | Confidence: {conf*100:.2f}%")