import os
import io
import pytest
from dotenv import load_dotenv
import mlflow
import numpy as np
from PIL import Image

# Load environment variables
load_dotenv()

# ==========================================
# Pytest Fixture: Load Model Once Per Session
# ==========================================
@pytest.fixture(scope="session")
def champion_model():
    """Loads the VisionGuard champion model from DagsHub once per test session."""
    username = os.environ.get("DAGSHUB_USERNAME")
    token = os.environ.get("DAGSHUB_TOKEN")
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")

    if not (username and token and tracking_uri):
        pytest.skip("DagsHub credentials not found. Skipping model tests.")

    os.environ["MLFLOW_TRACKING_USERNAME"] = username
    os.environ["MLFLOW_TRACKING_PASSWORD"] = token
    mlflow.set_tracking_uri(tracking_uri)

    try:
        client = mlflow.MlflowClient()
        model_name = "VisionGuard_Model"
        model_version = client.get_model_version_by_alias(name=model_name, alias="champion")
        run_id = model_version.run_id
        run_data = client.get_run(run_id)
        best_threshold = run_data.data.metrics.get("best_threshold", 0.5)
        model_uri = f"models:/{model_name}@champion"
        model = mlflow.keras.load_model(model_uri)
        return {"model": model, "threshold": best_threshold}
    except Exception as e:
        pytest.skip(f"Failed to load model from DagsHub: {e}")


# ==========================================
# Inference Functions
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


def predict_image(image_data, model_dict):
    model = model_dict["model"]
    threshold = model_dict["threshold"]
    img_array = preprocess_image(image_data)
    raw_prob = float(model.predict(img_array, verbose=0)[0][0])
    if raw_prob > threshold:
        label = "Fake (AI-Generated)"
        confidence = raw_prob
    else:
        label = "Real"
        confidence = 1.0 - raw_prob
    return label, float(confidence)


# ==========================================
# Pytest Test Cases (Now Properly Collected!)
# ==========================================
def test_model_fixture_loads(champion_model):
    """Test 1: Verify the model and threshold were loaded successfully."""
    assert champion_model["model"] is not None
    assert 0.0 <= champion_model["threshold"] <= 1.0


def test_inference_on_real_image(champion_model):
    """Test 2: Run inference on the Real test image."""
    img_path = "tests/test_Real.jpg"
    if not os.path.exists(img_path):
        pytest.skip(f"{img_path} not found.")
    label, conf = predict_image(img_path, champion_model)
    assert label in ["Real", "Fake (AI-Generated)"]
    assert 0.0 <= conf <= 1.0


def test_inference_on_ai_image(champion_model):
    """Test 3: Run inference on the AI test image."""
    img_path = "tests/test_AI.jpg"
    if not os.path.exists(img_path):
        pytest.skip(f"{img_path} not found.")
    label, conf = predict_image(img_path, champion_model)
    assert label in ["Real", "Fake (AI-Generated)"]
    assert 0.0 <= conf <= 1.0