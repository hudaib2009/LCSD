from __future__ import annotations

import io
import inspect
import logging
import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
from PIL import Image

logger = logging.getLogger("csd.cxr_embeddings")


@dataclass
class CxrEmbeddingResult:
    model: str
    dims: Optional[int]
    vector: Optional[list[float]]
    error: Optional[str] = None


_MODEL: Any = None
_MODEL_ERROR: Optional[str] = None
_MODEL_LOCK = threading.Lock()


def _call_with_known_kwargs(func: Callable[..., Any], **kwargs: Any) -> Any:
    try:
        signature = inspect.signature(func)
    except (TypeError, ValueError):
        return func(**kwargs)

    accepted = {
        name: value
        for name, value in kwargs.items()
        if name in signature.parameters and value is not None
    }
    return func(**accepted)


def _image_to_png_bytes(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _image_to_tf_example_bytes(image_bytes: bytes) -> bytes:
    # Avoid importing tensorflow unless needed.
    try:
        import tensorflow as tf
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(f"TensorFlow unavailable for TF Example export: {exc}")

    example = tf.train.Example(
        features=tf.train.Features(
            feature={
                "image/encoded": tf.train.Feature(
                    bytes_list=tf.train.BytesList(value=[image_bytes])
                )
            }
        )
    )
    return example.SerializeToString()


def _ensure_grayscale(image: Image.Image) -> Image.Image:
    if image.mode != "L":
        return image.convert("L")
    return image


def _resolve_model(module: Any) -> Any:
    model_name = os.environ.get("CXR_FOUNDATION_MODEL")
    model_path = os.environ.get("CXR_FOUNDATION_MODEL_PATH")

    candidates: list[Any] = []
    for name in (
        "CXRFoundationEmbeddingModel",
        "CXRFoundationModel",
        "CxrFoundationEmbeddingModel",
        "CxrFoundationModel",
        "CXRFoundation",
        "CxrFoundation",
    ):
        if hasattr(module, name):
            candidates.append(getattr(module, name))

    for name in (
        "get_embedding_model",
        "load_embedding_model",
        "get_model",
        "load_model",
        "load",
    ):
        if hasattr(module, name):
            candidates.append(getattr(module, name))

    for submodule_name in (
        "model",
        "embedding",
        "embeddings",
        "client",
        "inference",
    ):
        try:
            submodule = __import__(
                f"{module.__name__}.{submodule_name}", fromlist=["*"]
            )
        except Exception:
            continue
        for name in (
            "CXRFoundationEmbeddingModel",
            "CXRFoundationModel",
            "CxrFoundationEmbeddingModel",
            "CxrFoundationModel",
            "CXRFoundation",
            "CxrFoundation",
            "get_embedding_model",
            "load_embedding_model",
            "get_model",
            "load_model",
            "load",
        ):
            if hasattr(submodule, name):
                candidates.append(getattr(submodule, name))

    last_error: Optional[Exception] = None
    for candidate in candidates:
        try:
            if inspect.isclass(candidate):
                if hasattr(candidate, "from_pretrained"):
                    return _call_with_known_kwargs(
                        candidate.from_pretrained,
                        model_name=model_name,
                        model_id=model_name,
                        model_path=model_path,
                    )
                return _call_with_known_kwargs(
                    candidate,
                    model_name=model_name,
                    model_id=model_name,
                    model_path=model_path,
                )
            return _call_with_known_kwargs(
                candidate,
                model_name=model_name,
                model_id=model_name,
                model_path=model_path,
            )
        except Exception as exc:
            last_error = exc
            continue

    if last_error is not None:
        raise RuntimeError(
            "Unable to initialize a CXR Foundation model."
        ) from last_error

    raise RuntimeError("CXR Foundation module has no known model entrypoint.")


def _load_model() -> None:
    global _MODEL
    global _MODEL_ERROR

    if _MODEL is not None or _MODEL_ERROR is not None:
        return

    with _MODEL_LOCK:
        if _MODEL is not None or _MODEL_ERROR is not None:
            return
        try:
            import cxr_foundation
        except Exception as exc:
            _MODEL_ERROR = (
                "cxr-foundation dependency unavailable. Install with: "
                "pip install git+https://github.com/Google-Health/"
                "cxr-foundation.git#subdirectory=python"
            )
            logger.warning("CXR Foundation import failed: %s", exc)
            return

        try:
            _MODEL = _resolve_model(cxr_foundation)
        except Exception as exc:
            _MODEL_ERROR = f"CXR Foundation model init failed: {exc}"
            logger.warning("CXR Foundation init failed: %s", exc)
            return


def _extract_vector(output: Any) -> Optional[np.ndarray]:
    if output is None:
        return None

    if isinstance(output, dict):
        for key in ("embedding", "embeddings", "vector", "features", "output"):
            if key in output:
                output = output[key]
                break

    if isinstance(output, (list, tuple)):
        output = np.asarray(output)

    if hasattr(output, "numpy"):
        output = output.numpy()

    if isinstance(output, np.ndarray):
        if output.ndim > 1:
            output = np.squeeze(output)
        return output.astype("float32")

    return None


def _model_name(model: Any) -> str:
    for attr in ("model_name", "name", "__class__"):
        value = getattr(model, attr, None)
        if isinstance(value, str):
            return value
        if attr == "__class__":
            return value.__name__
    return "cxr-foundation"


def _run_embedding(model: Any, image: Image.Image, image_path: Optional[Path]) -> Any:
    image_bytes = _image_to_png_bytes(image)
    tf_example: Optional[bytes] = None

    call_targets: list[Callable[..., Any]] = []
    for name in ("embed", "get_embeddings", "get_embedding", "encode", "__call__"):
        if hasattr(model, name):
            call_targets.append(getattr(model, name))

    if not call_targets:
        raise RuntimeError("CXR Foundation model has no embedding callable.")

    for target in call_targets:
        try:
            return _call_with_known_kwargs(
                target,
                image=image,
                images=[image],
                image_bytes=image_bytes,
                image_bytes_list=[image_bytes],
                image_path=str(image_path) if image_path else None,
                inputs=image_bytes,
                examples=image_bytes,
                tf_examples=image_bytes,
            )
        except Exception:
            pass

    # Try TF Example bytes if supported.
    try:
        tf_example = _image_to_tf_example_bytes(image_bytes)
    except Exception:
        tf_example = None

    if tf_example is not None:
        for target in call_targets:
            try:
                return _call_with_known_kwargs(
                    target,
                    inputs=tf_example,
                    examples=tf_example,
                    tf_examples=tf_example,
                    serialized_examples=tf_example,
                )
            except Exception:
                pass

    raise RuntimeError("CXR Foundation embedding call failed for all candidates.")


def embed_cxr_foundation_image(
    image: Image.Image, image_path: Optional[Path] = None
) -> CxrEmbeddingResult:
    _load_model()

    if _MODEL_ERROR is not None:
        return CxrEmbeddingResult(
            model="cxr-foundation",
            dims=None,
            vector=None,
            error=_MODEL_ERROR,
        )

    if _MODEL is None:
        return CxrEmbeddingResult(
            model="cxr-foundation",
            dims=None,
            vector=None,
            error="CXR Foundation model is not initialized.",
        )

    grayscale = _ensure_grayscale(image)
    try:
        output = _run_embedding(_MODEL, grayscale, image_path)
    except Exception as exc:
        return CxrEmbeddingResult(
            model=_model_name(_MODEL),
            dims=None,
            vector=None,
            error=f"CXR Foundation embedding failed: {exc}",
        )

    vector = _extract_vector(output)
    if vector is None:
        return CxrEmbeddingResult(
            model=_model_name(_MODEL),
            dims=None,
            vector=None,
            error="CXR Foundation returned no embedding vector.",
        )

    vector_list = [float(value) for value in vector.reshape(-1).tolist()]
    return CxrEmbeddingResult(
        model=_model_name(_MODEL),
        dims=len(vector_list),
        vector=vector_list,
        error=None,
    )


def embed_cxr_foundation_path(image_path: Path) -> CxrEmbeddingResult:
    try:
        image = Image.open(image_path)
        image.load()
    except Exception as exc:
        return CxrEmbeddingResult(
            model="cxr-foundation",
            dims=None,
            vector=None,
            error=f"Failed to load image for embedding: {exc}",
        )

    return embed_cxr_foundation_image(image, image_path=image_path)
