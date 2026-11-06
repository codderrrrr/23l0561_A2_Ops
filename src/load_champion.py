import os
import io
from dotenv import load_dotenv
import mlflow
import tensorflow as tf
import numpy as np
from PIL import Image

load_dotenv()

username = os.environ.get("DAGSHUB_USERNAME")
token = os.environ.get("DAGSHUB_TOKEN")
tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")

if username and token and tracking_uri:
    os.environ["MLFLOW_TRACKING_USERNAME"] = username
    os.environ["MLFLOW_TRACKING_PASSWORD"] = token
    mlflow.set_tracking_uri(tracking_uri)
    print("Connected to DagsHub")
    
print("\nLoading VisionGuard Champion model from DagsHub")
client = mlflow.MlflowClient()
model_name = "VisionGuard_Model"

model_version = client.get_model_version_by_alias(name=model_name, alias="champion")
run_id = model_version.run_id

run_data = client.get_run(run_id) # type: ignore
best_threshold = run_data.data.metrics.get("best_threshold", 0.5)
print(f"✅ Loaded best threshold from champion run: {best_threshold:.4f}")

model_uri = f"models:/{model_name}@champion"
print(f"Downloading from: {model_uri}")
model = mlflow.keras.load_model(model_uri) # type: ignore
print("Champion model loaded successfully into memory!\n")

def preprocess_image(image_path):
    image = Image.open(image_path).convert('RGB')
    image = image.resize((224, 224))
    
    img_array = np.array(image, dtype=np.float32)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def predict_image(image_path):
    img_array = preprocess_image(image_path)
    raw_prob = float(model.predict(img_array, verbose=0)[0][0]) # type: ignore
    
    threshold = best_threshold
    if raw_prob > threshold:
        label = "Fake (AI-Generated)"
        confidence = raw_prob
    else:
        label = "Real"
        confidence = 1.0 - raw_prob
        
    return label, float(confidence)

if __name__ == "__main__":
    print("TESTING INFERENCE MODULE")
    test_images = ["tests/test_Real.jpg", "tests/test_AI.jpg"]
    
    for img_path in test_images:
        if os.path.exists(img_path):
            label, conf = predict_image(img_path)
            print(f"\nImage: {img_path}")
            print(f"Verdict: {label}")
            print(f"Confidence: {conf*100:.2f}%")
        else:
            print(f"\n{img_path} not found. Please put images in the 'tests' folder to test the inference module.")