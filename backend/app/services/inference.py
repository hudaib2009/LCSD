from __future__ import annotations

import base64
import logging
import os
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np
from fastapi import UploadFile
from PIL import Image

from backend.app.config import DISCLAIMER, HEATMAP_DIR, MODEL_LABELS, MODEL_PATHS, PATHOLOGY_LABELS, STORAGE_DIR, configure_runtime
from backend.app.services.cxr_embeddings import embed_cxr_foundation_image, embed_cxr_foundation_path
from backend.app.services.reporting import build_final_report
from backend.app.services.risk import combine_risk, compute_clinical_score, compute_ct_stage_risk, risk_bucket

configure_runtime()

import tensorflow as tf

logger = logging.getLogger("csd.inference")


class InferenceError(Exception):
    def __init__(self, status_code: int, message: str, detail: str):
        super().__init__(message)
        self.status_code = status_code
        self.message = message
        self.detail = detail


@dataclass
class ModelBundle:
    name: str
    path: Path
    model: tf.keras.Model
    feature_extractor: Optional[tf.keras.Model] = None
    target_layer: Optional[str] = None


MODELS: dict[str, ModelBundle] = {}


def startup_models() -> None:
    if MODELS:
        return
    if os.environ.get("CSD_SKIP_MODEL_LOAD") == "1":
        logger.info("Skipping model loading because CSD_SKIP_MODEL_LOAD=1")
        return

    tf.get_logger().setLevel("ERROR")
    for modality, model_path in MODEL_PATHS.items():
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        model = tf.keras.models.load_model(model_path)
        height, width, channels, data_format = input_spec(model)
        if data_format == "channels_first":
            dummy = tf.zeros([1, channels, height, width], dtype=tf.float32)
        else:
            dummy = tf.zeros([1, height, width, channels], dtype=tf.float32)
        _ = model(dummy, training=False)

        feature_extractor = None
        target_layer = None
        conv_layer = find_last_conv2d_layer(model)
        if conv_layer is not None:
            try:
                feature_extractor = tf.keras.models.Model(
                    inputs=model.inputs,
                    outputs=conv_layer.output,
                )
                target_layer = conv_layer.name
            except Exception as exc:  # pragma: no cover - model-specific guard
                logger.warning("Feature extractor build failed: %s", exc)
        else:
            logger.warning("No Conv2D layer found for modality=%s", modality)

        MODELS[modality] = ModelBundle(
            name=MODEL_LABELS[modality],
            path=model_path,
            model=model,
            feature_extractor=feature_extractor,
            target_layer=target_layer,
        )
        logger.info("Loaded %s model from %s", modality, model_path)


def health_payload() -> dict[str, Any]:
    return {
        "ok": True,
        "models_loaded": {
            modality: {
                "name": bundle.name,
                "path": str(bundle.path),
                "loaded": True,
            }
            for modality, bundle in MODELS.items()
        },
    }


def normalize_image(image: Image.Image) -> np.ndarray:
    return np.asarray(image).astype("float32")


def center_crop(image: Image.Image) -> Image.Image:
    width, height = image.size
    side = min(width, height)
    left = int((width - side) / 2)
    upper = int((height - side) / 2)
    right = left + side
    lower = upper + side
    return image.crop((left, upper, right, lower))


def input_spec(model: tf.keras.Model) -> tuple[int, int, int, str]:
    input_shape = model.input_shape
    if isinstance(input_shape, list):
        input_shape = input_shape[0]
    if input_shape is None or len(input_shape) != 4:
        return 224, 224, 3, "channels_last"

    _, height, width, channels = input_shape
    data_format = "channels_last"
    if height in {1, 3} and channels not in {1, 3}:
        channels = height
        height = input_shape[2]
        width = input_shape[3]
        data_format = "channels_first"

    return int(height or 224), int(width or 224), int(channels or 3), data_format


def load_image(image_path: Path) -> Image.Image:
    if image_path.suffix.lower() == ".dcm":
        raise InferenceError(400, "Invalid image.", "DICOM files are not supported yet.")
    try:
        image = Image.open(image_path)
        image.load()
        if getattr(image, "n_frames", 1) > 1:
            raise InferenceError(400, "Invalid image.", "Multi-frame images are not supported yet.")
        return image
    except InferenceError:
        raise
    except Exception as exc:
        raise InferenceError(400, "Invalid image.", f"Unsupported image format: {exc}")


def load_upload_image(file: UploadFile) -> Image.Image:
    try:
        file.file.seek(0)
        image = Image.open(file.file)
        image.load()
        if getattr(image, "n_frames", 1) > 1:
            raise ValueError("Multi-frame images are not supported.")
        return image
    except Exception as exc:
        raise InferenceError(400, "Invalid image upload.", str(exc))


def prepare_image(image_path: Path, modality: str, model: tf.keras.Model) -> tuple[tf.Tensor, np.ndarray, str]:
    image = load_image(image_path)
    return prepare_image_obj(image, modality, model)


def prepare_image_obj(
    image: Image.Image,
    modality: str,
    model: tf.keras.Model,
) -> tuple[tf.Tensor, np.ndarray, str]:
    height, width, channels, data_format = input_spec(model)

    if modality == "pathology":
        image = center_crop(image.convert("RGB"))
    elif modality == "ct":
        image = image.convert("RGB")
    else:
        image = image.convert("RGB") if channels == 3 else image.convert("L")

    image = image.resize((width, height))
    normalized = normalize_image(image)

    if channels == 1:
        if normalized.ndim == 2:
            normalized = np.expand_dims(normalized, axis=-1)
        elif normalized.shape[-1] != 1:
            normalized = normalized[:, :, :1]
    else:
        if normalized.ndim == 2:
            normalized = np.stack([normalized] * 3, axis=-1)
        elif normalized.shape[-1] == 1:
            normalized = np.repeat(normalized, 3, axis=-1)

    if data_format == "channels_first":
        normalized = np.transpose(normalized, (2, 0, 1))

    image_tensor = tf.convert_to_tensor(normalized, dtype=tf.float32)
    if image_tensor.shape.rank == 3:
        image_tensor = image_tensor[None, ...]

    display_array = np.asarray(image.convert("RGB")).astype(np.uint8)
    return image_tensor, display_array, data_format


def probability_from_prediction(prediction: np.ndarray) -> float:
    flat = prediction.ravel()
    if flat.size == 0:
        return 0.0
    if flat.size == 1:
        return float(flat[0])
    return float(np.max(flat))


def label_from_probability(probability: float) -> str:
    return "High Risk" if probability >= 0.5 else "Low Risk"


def find_last_conv2d_layer(model: tf.keras.Model) -> Optional[tf.keras.layers.Layer]:
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer
        if isinstance(layer, tf.keras.Model):
            nested = find_last_conv2d_layer(layer)
            if nested is not None:
                return nested
    return None


def compute_score(predictions: tf.Tensor) -> tf.Tensor:
    if predictions.shape.rank == 2 and predictions.shape[-1] == 1:
        p = tf.cast(predictions[:, 0], tf.float32)
        p = tf.clip_by_value(p, 1e-6, 1.0 - 1e-6)
        return tf.math.log(p / (1.0 - p))
    if predictions.shape.rank == 1:
        p = tf.cast(predictions, tf.float32)
        p = tf.clip_by_value(p, 1e-6, 1.0 - 1e-6)
        return tf.math.log(p / (1.0 - p))
    if predictions.shape.rank == 2:
        class_index = tf.argmax(predictions[0])
        return predictions[:, class_index]
    return predictions[:, 0]


def generate_gradcam(
    model: tf.keras.Model,
    feature_extractor: Optional[tf.keras.Model],
    image_tensor: tf.Tensor,
    data_format: str,
    target_layer: Optional[str],
) -> tuple[Optional[np.ndarray], Optional[str], Optional[str]]:
    if feature_extractor is None:
        return None, "Grad-CAM failed: no Conv2D layer found.", target_layer

    with tf.GradientTape() as tape:
        conv_acts = feature_extractor(image_tensor, training=False)
        tape.watch(conv_acts)
        predictions = model(image_tensor, training=False)
        score = compute_score(predictions)

    grads = tape.gradient(score, conv_acts)
    if grads is None:
        return None, "Grad-CAM failed: gradients unavailable.", target_layer

    if data_format == "channels_first":
        pooled_grads = tf.reduce_mean(grads, axis=(2, 3))
        conv_acts = conv_acts[0]
        heatmap = tf.reduce_sum(conv_acts * pooled_grads[0][:, None, None], axis=0)
    else:
        pooled_grads = tf.reduce_mean(grads, axis=(1, 2))
        conv_acts = conv_acts[0]
        heatmap = tf.reduce_sum(conv_acts * pooled_grads[0], axis=-1)

    heatmap = tf.maximum(heatmap, 0)
    max_val = tf.reduce_max(heatmap)
    if max_val == 0:
        return None, "Grad-CAM failed: heatmap was empty.", target_layer

    heatmap /= max_val
    return heatmap.numpy(), None, target_layer


def generate_saliency(
    model: tf.keras.Model,
    image_tensor: tf.Tensor,
    data_format: str,
) -> tuple[Optional[np.ndarray], Optional[str], Optional[str]]:
    tensor = tf.convert_to_tensor(image_tensor, dtype=tf.float32)
    with tf.GradientTape() as tape:
        tape.watch(tensor)
        predictions = model(tensor, training=False)
        score = compute_score(predictions)

    grads = tape.gradient(score, tensor)
    if grads is None:
        return None, "Saliency failed: gradients unavailable.", None

    saliency = tf.abs(grads)[0]
    if data_format == "channels_first":
        saliency = tf.reduce_max(saliency, axis=0)
    elif saliency.shape.rank == 3:
        saliency = tf.reduce_max(saliency, axis=-1)

    hm_min = tf.reduce_min(saliency)
    hm_max = tf.reduce_max(saliency)
    denom = tf.maximum(hm_max - hm_min, 1e-12)
    saliency = (saliency - hm_min) / denom
    warning = None
    if hm_max < 1e-10:
        warning = "Low-gradient saliency; heatmap may be uninformative."

    return saliency.numpy(), None, warning


def save_explainability_assets(case_id: str, modality: str, heatmap: np.ndarray, original: np.ndarray, output_dir: Path) -> tuple[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    height, width = original.shape[:2]
    heatmap_resized = cv2.resize(heatmap, (width, height))
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    original_bgr = cv2.cvtColor(original, cv2.COLOR_RGB2BGR)
    overlay = cv2.addWeighted(original_bgr, 0.6, heatmap_color, 0.4, 0)

    heatmap_path = output_dir / f"heatmap_{modality}.png"
    overlay_path = output_dir / f"overlay_{modality}.png"
    cv2.imwrite(str(heatmap_path), heatmap_color)
    cv2.imwrite(str(overlay_path), overlay)

    return (
        f"{case_id}/explainability/{heatmap_path.name}",
        f"{case_id}/explainability/{overlay_path.name}",
    )


def save_static_heatmap_assets(prefix: str, heatmap: np.ndarray, original: np.ndarray) -> tuple[str, str]:
    HEATMAP_DIR.mkdir(parents=True, exist_ok=True)
    height, width = original.shape[:2]
    heatmap_resized = cv2.resize(heatmap, (width, height))
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    original_bgr = cv2.cvtColor(original, cv2.COLOR_RGB2BGR)
    overlay = cv2.addWeighted(original_bgr, 0.6, heatmap_color, 0.4, 0)

    heatmap_path = HEATMAP_DIR / f"{prefix}_heatmap.png"
    overlay_path = HEATMAP_DIR / f"{prefix}_overlay.png"
    cv2.imwrite(str(heatmap_path), heatmap_color)
    cv2.imwrite(str(overlay_path), overlay)

    return f"/static/{heatmap_path.name}", f"/static/{overlay_path.name}"


def compute_explainability(
    bundle: ModelBundle,
    image_tensor: tf.Tensor,
    data_format: str,
    original: np.ndarray,
    *,
    case_id: str | None = None,
    modality: str,
) -> dict[str, Any]:
    warning = None
    try:
        heatmap, error, layer_name = generate_gradcam(
            bundle.model,
            bundle.feature_extractor,
            image_tensor,
            data_format,
            bundle.target_layer,
        )
    except Exception as exc:  # pragma: no cover - model-specific guard
        heatmap, error, layer_name = None, f"Grad-CAM failed: {exc}", bundle.target_layer

    method = "gradcam" if heatmap is not None else None
    if heatmap is None:
        try:
            heatmap, fallback_error, warning = generate_saliency(
                bundle.model,
                image_tensor,
                data_format,
            )
        except Exception as exc:  # pragma: no cover - model-specific guard
            heatmap, fallback_error, warning = None, f"Saliency failed: {exc}", None
        if heatmap is not None:
            method = "saliency"
            error = None
        else:
            error = fallback_error

    if heatmap is None:
        return {
            "method": method,
            "heatmap_path": None,
            "overlay_path": None,
            "warning": warning,
            "error": error,
            "target_layer": layer_name or bundle.target_layer,
        }

    if case_id:
        explain_dir = STORAGE_DIR / case_id / "explainability"
        heatmap_path, overlay_path = save_explainability_assets(
            case_id,
            modality,
            heatmap,
            original,
            explain_dir,
        )
    else:
        prefix = f"{modality}_{uuid.uuid4().hex}"
        heatmap_path, overlay_path = save_static_heatmap_assets(prefix, heatmap, original)

    return {
        "method": method,
        "heatmap_path": heatmap_path,
        "overlay_path": overlay_path,
        "warning": warning,
        "error": error,
        "target_layer": layer_name or bundle.target_layer,
    }


def encode_image_base64(image_path: Path) -> str:
    payload = image_path.read_bytes()
    return f"data:image/png;base64,{base64.b64encode(payload).decode('utf-8')}"


def get_model_bundle(modality: str) -> ModelBundle:
    if modality not in MODELS:
        raise InferenceError(503, f"{modality} model unavailable.", "Model not loaded.")
    return MODELS[modality]


def predict_xray_upload(file: UploadFile) -> dict[str, Any]:
    bundle = get_model_bundle("xray")
    image = load_upload_image(file)
    image_tensor, display_array, data_format = prepare_image_obj(image, "xray", bundle.model)
    prediction = bundle.model(image_tensor, training=False)
    probability = probability_from_prediction(np.asarray(prediction))
    explainability = compute_explainability(bundle, image_tensor, data_format, display_array, modality="xray")

    return {
        "abnormal_score": float(probability),
        "pred_label": label_from_probability(probability),
        "heatmap": explainability["overlay_path"] or explainability["heatmap_path"],
        "heatmap_method": explainability["method"],
        "heatmap_warning": explainability["warning"],
        "heatmap_error": explainability["error"],
        "target_layer": explainability["target_layer"],
        "disclaimer": DISCLAIMER,
    }


def predict_ct_payload(payload: dict[str, Any]) -> dict[str, Any]:
    bundle = get_model_bundle("ct")
    image_path_str = payload.get("image_path")
    if not image_path_str:
        raise InferenceError(400, "Missing image_path.", "Provide path to study.")

    image_path = Path(image_path_str)
    if not image_path.exists():
        raise InferenceError(404, "File not found.", f"Path: {image_path}")

    clinical_data = payload.get("clinical", {})
    image_tensor, _, _ = prepare_image(image_path, "ct", bundle.model)
    prediction = bundle.model(image_tensor, training=False)
    probabilities = np.asarray(prediction).ravel().astype(float).tolist()
    imaging_risk, predicted_stage = compute_ct_stage_risk(probabilities)
    clinical_risk = compute_clinical_score(clinical_data)
    final_risk = combine_risk(imaging_risk, clinical_risk)

    return {
        "prediction_probs": probabilities,
        "predicted_stage": predicted_stage,
        "imaging_risk": float(imaging_risk),
        "clinical_risk": float(clinical_risk),
        "risk_score": float(final_risk),
        "risk": risk_bucket(final_risk),
        "disclaimer": DISCLAIMER,
    }


def predict_pathology_upload(file: UploadFile) -> dict[str, Any]:
    bundle = get_model_bundle("pathology")
    image = load_upload_image(file)
    image_tensor, display_array, data_format = prepare_image_obj(
        image,
        "pathology",
        bundle.model,
    )
    prediction = bundle.model(image_tensor, training=False)
    logits = np.asarray(prediction).ravel()
    if logits.size == 0:
        raise InferenceError(500, "Pathology inference failed.", "Empty logits.")

    if logits.size == 1:
        confidence = float(logits[0])
        label = PATHOLOGY_LABELS[0] if PATHOLOGY_LABELS else "Unknown"
    else:
        index = int(np.argmax(logits))
        confidence = float(logits[index])
        label = PATHOLOGY_LABELS[index] if index < len(PATHOLOGY_LABELS) else f"class_{index}"

    explainability = compute_explainability(
        bundle,
        image_tensor,
        data_format,
        display_array,
        modality="pathology",
    )

    return {
        "subtype": label,
        "confidence": confidence,
        "heatmap": explainability["overlay_path"] or explainability["heatmap_path"],
        "heatmap_method": explainability["method"],
        "heatmap_warning": explainability["warning"],
        "heatmap_error": explainability["error"],
        "target_layer": explainability["target_layer"],
        "disclaimer": DISCLAIMER,
    }


def generate_plan_payload(payload: dict[str, Any]) -> dict[str, Any]:
    return build_final_report(payload)


def cxr_embeddings_upload(file: UploadFile) -> tuple[int, dict[str, Any]]:
    try:
        image = Image.open(file.file)
        image.load()
    except Exception as exc:
        return 400, {
            "ok": False,
            "embeddings": {
                "cxr_foundation": {
                    "model": "cxr-foundation",
                    "dims": None,
                    "vector": None,
                    "error": f"Invalid image upload: {exc}",
                }
            },
            "error": "Invalid image upload.",
            "detail": str(exc),
            "disclaimer": DISCLAIMER,
        }

    result = embed_cxr_foundation_image(image)
    payload = {
        "ok": result.error is None,
        "embeddings": {
            "cxr_foundation": {
                "model": result.model,
                "dims": result.dims,
                "vector": result.vector,
                "error": result.error,
            }
        },
        "error": result.error,
        "detail": None,
        "disclaimer": DISCLAIMER,
    }
    return (503 if result.error else 200), payload


def infer_case(payload: dict[str, Any]) -> dict[str, Any]:
    case_id = payload.get("case_id")
    modality = payload.get("modality")
    image_path = payload.get("image_path")
    return_explainability = bool(payload.get("return_explainability", True))
    clinical = payload.get("clinical") or {}

    if not case_id or not modality or not image_path:
        raise InferenceError(
            400,
            "Missing required fields.",
            "case_id, modality, and image_path are required.",
        )

    if modality not in MODELS:
        raise InferenceError(
            400,
            "Unsupported modality.",
            f"Supported: {', '.join(MODELS.keys())}.",
        )

    resolved_path = Path(image_path).expanduser().resolve()
    if not resolved_path.exists():
        raise InferenceError(400, "Image not found.", str(resolved_path))

    try:
        resolved_path.relative_to(STORAGE_DIR.resolve())
    except ValueError as exc:
        raise InferenceError(400, "Invalid image path.", "Image must be inside frontend/storage.") from exc

    bundle = MODELS[modality]
    image_tensor, display_array, data_format = prepare_image(resolved_path, modality, bundle.model)
    prediction = bundle.model(image_tensor, training=False)
    raw_prediction = np.asarray(prediction).ravel().astype(float)
    probability = probability_from_prediction(raw_prediction)

    imaging_risk = probability
    predicted_stage: str | None = None
    if modality == "ct":
        imaging_risk, predicted_stage = compute_ct_stage_risk(raw_prediction.tolist())

    clinical_risk = compute_clinical_score(clinical)
    final_risk = combine_risk(imaging_risk, clinical_risk)
    risk = risk_bucket(final_risk)

    explainability = {
        "method": None,
        "heatmap_path": None,
        "overlay_path": None,
        "warning": None,
        "error": None,
        "target_layer": None,
    }
    if return_explainability:
        explainability = compute_explainability(
            bundle,
            image_tensor,
            data_format,
            display_array,
            case_id=case_id,
            modality=modality,
        )

    embeddings: dict[str, Any] = {}
    if modality == "xray":
        embedding = embed_cxr_foundation_path(resolved_path)
        embeddings["cxr_foundation"] = {
            "model": embedding.model,
            "dims": embedding.dims,
            "vector": embedding.vector,
            "error": embedding.error,
        }

    return {
        "case_id": case_id,
        "modality": modality,
        "model": {
            "name": bundle.name,
            "path": str(bundle.path),
            "framework": "tensorflow",
            "loaded": True,
        },
        "prediction": {
            "label": "positive" if probability >= 0.5 else "negative",
            "probability": float(probability),
        },
        "risk": risk,
        "risk_score": float(final_risk),
        "imaging_risk": float(imaging_risk),
        "clinical_risk": float(clinical_risk),
        "predicted_stage": predicted_stage,
        "explainability": explainability,
        "embeddings": embeddings or None,
        "disclaimer": DISCLAIMER,
    }
