import os
import io
import pytest
from dotenv import load_dotenv
import mlflow
import numpy as np
from PIL import Image

# Load environment variables from .env file (for local testing)
load_dotenv()

# ==========================================
# Global Variables & Model Loading
# ==========================================
model = None
best_threshold = 0.5

def load_champion_model():
    """Loads the VisionGuard champion model from DagsHub."""
    global model, best_threshold
    
    username = os.environ.get("DAGSHUB_USERNAME")
    token = os.environ.get("DAGSHUB_TOKEN")
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")

    # If running in CI without secrets, skip the tests gracefully
    if not (username and token and tracking_uri):
        pytest.skip("DagsHub credentials not found. Skipping model tests.")

    os.environ["MLFLOW_TRACKING_USERNAME"] = username
    os.environ["MLFLOW_TRACKING_PASSWORD"] = token
    mlflow.set_tracking_uri(tracking_uri)
    
    print("\nLoading VisionGuard Champion model from DagsHub...")
    client = mlflow.MlflowClient()
    model_name = "VisionGuard_Model"
    
    try:
        model_version = client.get_model_version_by_alias(name=model_name, alias="champion")
        run_id = model_version.run_id
        run_data = client.get_run(run_id) # type: ignore
        best_threshold = run_data.data.metrics.get("best_threshold", 0.5)
        
        model_uri = f"models:/{model_name}@champion"
        model = mlflow.keras.load_model(model_uri) # type: ignore
        print(f"✅ Loaded best threshold: {best_threshold:.4f}")
        print("✅ Champion model loaded successfully!\n")
    except Exception as e:
        pytest.skip(f"Failed to load model from DagsHub: {e}")

# Load the model once when the test file is initialized
load_champion_model()

# ==========================================
# Inference Functions (From your script)
# ==========================================
def preprocess_image(image_data):
    if isinstance(image_data, bytes):
        image = Image.open(io.BytesIO(image_data)).convert('RGB')
    else:
        image = Image.open(image_data).convert('RGB')
        
    image = image.resize((224, 224))
    img_array = np.array(image, dtype=np.float32)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def predict_image(image_data):
    if model is None:
        pytest.skip("Model is not loaded.")
        
    img_array = preprocess_image(image_data)
    raw_prob = float(model.predict(img_array, verbose=0)[0][0]) # type: ignore
    
    threshold = best_threshold
    if raw_prob > threshold:
        label = "Fake (AI-Generated)"
        confidence = raw_prob
    else:
        label = "Real"
        confidence = 1.0 - raw_prob
        
    return label, float(confidence)

# ==========================================
# Pytest Test Cases
# ==========================================
def test_model_is_loaded():
    """Test 1: Verify the model and threshold were loaded successfully."""
    assert model is not None, "Model should not be None"
    assert 0.0 <= best_threshold <= 1.0, "Threshold must be between 0 and 1"

def test_inference_on_real_image():
    """Test 2: Run inference on the Real test image and verify output format."""
    img_path = "tests/test_Real.jpg"
    if not os.path.exists(img_path):
        pytest.skip(f"{img_path} not found.")
        
    label, conf = predict_image(img_path)
    
    assert label in ["Real", "Fake (AI-Generated)"], f"Invalid label: {label}"
    assert 0.0 <= conf <= 1.0, f"Confidence {conf} out of bounds"
    print(f"\n[Real Image] Verdict: {label} | Confidence: {conf*100:.2f}%")

def test_inference_on_ai_image():
    """Test 3: Run inference on the AI test image and verify output format."""
    img_path = "tests/test_AI.jpg"
    if not os.path.exists(img_path):
        pytest.skip(f"{img_path} not found.")
        
    label, conf = predict_image(img_path)
    
    assert label in ["Real", "Fake (AI-Generated)"], f"Invalid label: {label}"
    assert 0.0 <= conf <= 1.0, f"Confidence {conf} out of bounds"
    print(f"\n[AI Image] Verdict: {label} | Confidence: {conf*100:.2f}%")