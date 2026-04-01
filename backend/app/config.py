from __future__ import annotations

import logging
import os
from pathlib import Path

logger = logging.getLogger("csd.config")

ROOT_DIR = Path(__file__).resolve().parents[2]
FRONTEND_DIR = ROOT_DIR / "frontend"
STORAGE_DIR = FRONTEND_DIR / "storage"
HEATMAP_DIR = ROOT_DIR / "backend" / "app" / "outputs" / "heatmaps"
HEATMAP_DIR.mkdir(parents=True, exist_ok=True)

DISCLAIMER = (
    "This tool provides model outputs for decision support and is not a diagnosis."
)

MODEL_PATHS = {
    "ct": Path(
        os.getenv(
            "CSD_CT_MODEL_PATH",
            ROOT_DIR / "ml" / "models" / "nsclc_stage_classifier.keras",
        )
    ),
    "pathology": Path(
        os.getenv(
            "CSD_PATHOLOGY_MODEL_PATH",
            ROOT_DIR
            / "ml"
            / "models"
            / "pathology"
            / "checkpoints"
            / "1"
            / "best_lung_tf.keras",
        )
    ),
    "xray": Path(
        os.getenv(
            "CSD_XRAY_MODEL_PATH",
            ROOT_DIR / "ml" / "models" / "archive" / "nih_binary_xray_model.h5",
        )
    ),
}

MODEL_LABELS = {
    "ct": "CT cancer classifier",
    "pathology": "Pathology lung classifier",
    "xray": "X-ray binary classifier",
}

PATHOLOGY_LABELS = [
    label.strip()
    for label in os.getenv("CSD_PATHOLOGY_LABELS", "ACA,SCC,Normal").split(",")
    if label.strip()
]

ALLOW_ORIGINS = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]

FASTAPI_HOST = os.getenv("CSD_HOST", "127.0.0.1")
FASTAPI_PORT = int(os.getenv("CSD_PORT", "8000"))


def configure_runtime() -> None:
    """Apply runtime defaults before TensorFlow is imported."""
    if os.environ.get("CSD_USE_GPU") != "1":
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
        logger.info("CSD_USE_GPU not set; forcing CPU inference.")
