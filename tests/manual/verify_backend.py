import sys
import os
import numpy as np
import tensorflow as tf
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
FRONTEND_SERVER = REPO_ROOT / "frontend" / "server"
DEFAULT_CT_MODEL = REPO_ROOT / "ml" / "models" / "nsclc_stage_classifier.keras"

# Add frontend server compatibility wrapper to path
sys.path.append(str(FRONTEND_SERVER))

# Allow callers to provide a model path without baking in local machine paths.
os.environ.setdefault("CSD_CT_MODEL_PATH", str(DEFAULT_CT_MODEL))

try:
    from main import _clinical_score, MODELS, _load_models, _prepare_image_obj
except ImportError as e:
    print(f"Import failed: {e}")
    sys.exit(1)

def test_clinical_score():
    print("\nTesting Clinical Score Logic...")
    cases = [
        ({"age": 65, "isSmoker": True, "packYears": 30, "ecog": 2, "histology": "Adenocarcinoma"}, 1.0), # 5/5
        ({"age": 40, "isSmoker": False}, 0.0), # 0/5
        ({"age": 70, "isSmoker": False}, 0.2), # 1/5 (Age)
        ({}, 0.0)
    ]
    
    for clinical, expected in cases:
        score = _clinical_score(clinical)
        print(f"Input: {clinical} -> Score: {score} (Expected: {expected})")
        assert abs(score - expected) < 1e-6, f"Failed: {score} != {expected}"
    print("✅ Clinical Score Tests Passed")

def test_model_loading():
    print("\nTesting Model Loading...")
    model_path = Path(os.environ["CSD_CT_MODEL_PATH"])
    if not model_path.exists():
        print(f"Skipping model-loading check; model file not present: {model_path}")
        return
    try:
        _load_models()
        if "ct" in MODELS:
            print(f"✅ CT Model Loaded: {MODELS['ct'].path}")
            model = MODELS['ct'].model
            print(f"   Input Shape: {model.input_shape}")
            # Verify input shape is (None, 224, 224, 3)
            input_shape = model.input_shape
            if isinstance(input_shape, list): input_shape = input_shape[0]
            assert input_shape[1:] == (224, 224, 3), f"Expected (224,224,3), got {input_shape}"
        else:
            print("❌ CT Model NOT Loaded")
    except Exception as e:
        print(f"❌ Model Loading Failed: {e}")

if __name__ == "__main__":
    test_clinical_score()
    test_model_loading()
