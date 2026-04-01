"""
tfdata_utils.py — TensorFlow tf.data input pipeline utilities.

Handles:
  - Building tf.data.Dataset from CSV of PNG paths + labels
  - BALANCED class sampling via tf.data.Dataset.sample_from_datasets
  - Moderate data augmentation (train only)
  - Class weight computation
  - Collapse detection
"""

import os
import logging
from typing import Tuple, Dict

import numpy as np
import tensorflow as tf

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Image loading (shared by all dataset builders)
# ---------------------------------------------------------------------------

def _load_and_preprocess(path: tf.Tensor,
                         label: tf.Tensor,
                         target_size: Tuple[int, int] = (224, 224)
                         ) -> Tuple[tf.Tensor, tf.Tensor]:
    """Load PNG, normalize to [0,1], convert grayscale to 3-channel."""
    raw = tf.io.read_file(path)
    image = tf.io.decode_png(raw, channels=1)  # (H, W, 1)
    image = tf.image.resize(image, target_size)
    image = tf.cast(image, tf.float32) / 255.0
    image = tf.repeat(image, repeats=3, axis=-1)  # (H, W, 3)
    return image, label


# ---------------------------------------------------------------------------
# BALANCED dataset — guarantees equal class representation
# ---------------------------------------------------------------------------

def build_balanced_dataset(csv_path: str,
                           base_dir: str,
                           image_col: str = 'path_image',
                           label_col: str = 'label',
                           batch_size: int = 16,
                           augment: bool = False,
                           target_size: Tuple[int, int] = (224, 224),
                           seed: int = 42) -> Tuple[tf.data.Dataset, int]:
    """
    Build a class-balanced dataset using tf.data.Dataset.sample_from_datasets.

    Creates one infinite-repeat dataset per class, then samples uniformly
    across classes. This guarantees each class gets ~equal representation
    regardless of how imbalanced the original data is.

    Returns:
        (dataset, steps_per_epoch) — you MUST pass steps_per_epoch to model.fit()
    """
    import pandas as pd

    df = pd.read_csv(csv_path)
    classes = sorted(df[label_col].unique())
    n_classes = len(classes)

    print(f"Building balanced dataset from {csv_path}")
    per_class_datasets = []
    class_counts = []

    for cls in classes:
        cls_df = df[df[label_col] == cls]
        cls_paths = [os.path.join(base_dir, p) for p in cls_df[image_col].values]
        cls_labels = cls_df[label_col].values.astype(np.int32)
        class_counts.append(len(cls_df))

        ds = tf.data.Dataset.from_tensor_slices((cls_paths, cls_labels))
        ds = ds.shuffle(len(cls_paths), seed=seed, reshuffle_each_iteration=True)
        ds = ds.repeat()  # infinite stream
        per_class_datasets.append(ds)

        print(f"  Class {cls}: {len(cls_df)} slices")

    # Sample uniformly across classes
    weights = [1.0 / n_classes] * n_classes
    balanced_ds = tf.data.Dataset.sample_from_datasets(
        per_class_datasets, weights=weights, seed=seed
    )

    # Epoch size: each class contributes min_class_count samples
    # so minority class gets seen once, others are subsampled
    min_count = min(class_counts)
    epoch_size = min_count * n_classes
    steps_per_epoch = epoch_size // batch_size

    print(f"  Balanced epoch: {epoch_size} samples "
          f"({min_count} per class × {n_classes} classes)")
    print(f"  Steps per epoch: {steps_per_epoch}")

    # Load images
    balanced_ds = balanced_ds.map(
        lambda p, l: _load_and_preprocess(p, l, target_size),
        num_parallel_calls=tf.data.AUTOTUNE
    )

    # Augment if training
    if augment:
        balanced_ds = balanced_ds.map(_augment,
                                       num_parallel_calls=tf.data.AUTOTUNE)

    balanced_ds = balanced_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return balanced_ds, steps_per_epoch


# ---------------------------------------------------------------------------
# Standard dataset (for val/test — no balancing needed)
# ---------------------------------------------------------------------------

def build_dataset(csv_path: str,
                  base_dir: str,
                  image_col: str = 'path_image',
                  label_col: str = 'label',
                  batch_size: int = 16,
                  augment: bool = False,
                  shuffle: bool = False,
                  target_size: Tuple[int, int] = (224, 224),
                  seed: int = 42) -> tf.data.Dataset:
    """
    Build a standard (unbalanced) tf.data.Dataset from a CSV file.
    Use for validation and test sets where you want true distribution.
    """
    import pandas as pd

    df = pd.read_csv(csv_path)
    paths = [os.path.join(base_dir, p) for p in df[image_col].values]
    labels = df[label_col].values.astype(np.int32)

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))

    if shuffle:
        ds = ds.shuffle(buffer_size=len(paths), seed=seed,
                        reshuffle_each_iteration=True)

    ds = ds.map(lambda p, l: _load_and_preprocess(p, l, target_size),
                num_parallel_calls=tf.data.AUTOTUNE)

    if augment:
        ds = ds.map(_augment, num_parallel_calls=tf.data.AUTOTUNE)

    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


# ---------------------------------------------------------------------------
# Augmentation — MODERATE (preserve medical signal)
# ---------------------------------------------------------------------------

def _augment(image: tf.Tensor,
             label: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    Moderate augmentations:
      - Random horizontal flip
      - Small rotation (±10°)
      - Slight brightness/contrast jitter
      - Mild Gaussian noise
    """
    image = tf.image.random_flip_left_right(image)

    # Small rotation ±10 degrees
    angle = tf.random.uniform([], -10.0, 10.0) * (3.14159265 / 180.0)
    image = _rotate_image(image, angle)

    # Mild brightness and contrast
    image = tf.image.random_brightness(image, max_delta=0.08)
    image = tf.image.random_contrast(image, lower=0.92, upper=1.08)

    # Mild Gaussian noise
    noise = tf.random.normal(shape=tf.shape(image), mean=0.0, stddev=0.015)
    image = image + noise

    image = tf.clip_by_value(image, 0.0, 1.0)
    return image, label


def _rotate_image(image: tf.Tensor, angle: tf.Tensor) -> tf.Tensor:
    """Rotate image by angle (radians) around center."""
    cos_a = tf.cos(angle)
    sin_a = tf.sin(angle)
    h = tf.cast(tf.shape(image)[0], tf.float32)
    w = tf.cast(tf.shape(image)[1], tf.float32)
    cx, cy = w / 2.0, h / 2.0

    transform = [
        cos_a, -sin_a, cx - cx * cos_a + cy * sin_a,
        sin_a, cos_a, cy - cx * sin_a - cy * cos_a,
        0.0, 0.0
    ]
    transform = tf.cast(tf.stack(transform), tf.float32)

    image_4d = tf.expand_dims(image, 0)
    rotated = tf.raw_ops.ImageProjectiveTransformV3(
        images=image_4d,
        transforms=tf.expand_dims(transform, 0),
        output_shape=tf.shape(image)[:2],
        interpolation='BILINEAR',
        fill_mode='CONSTANT',
        fill_value=0.0,
    )
    return tf.squeeze(rotated, 0)


# ---------------------------------------------------------------------------
# Class weights
# ---------------------------------------------------------------------------

def get_class_weights(labels: np.ndarray) -> Dict[int, float]:
    """Compute balanced class weights."""
    from sklearn.utils.class_weight import compute_class_weight

    classes = np.unique(labels)
    weights = compute_class_weight('balanced', classes=classes, y=labels)
    return {int(c): float(w) for c, w in zip(classes, weights)}


# ---------------------------------------------------------------------------
# Collapse detection
# ---------------------------------------------------------------------------

def detect_collapse(y_prob: np.ndarray,
                    y_pred: np.ndarray = None,
                    threshold: float = 0.9,
                    class_names: list = None) -> bool:
    """
    Detect if model has collapsed to predicting a single class.

    Returns True if collapse detected.
    """
    if y_pred is None:
        y_pred = y_prob.argmax(axis=1)

    n = len(y_pred)
    unique, counts = np.unique(y_pred, return_counts=True)

    mean_probs = y_prob.mean(axis=0)
    std_probs = y_prob.std(axis=0)

    print(f"  Mean softmax:  {mean_probs}")
    print(f"  Std  softmax:  {std_probs}")
    print(f"  Prediction distribution:")
    for cls, cnt in zip(unique, counts):
        name = class_names[cls] if class_names else f"Class {cls}"
        pct = cnt / n * 100
        print(f"    {name}: {cnt}/{n} ({pct:.1f}%)")

    max_frac = counts.max() / n
    if max_frac >= threshold:
        dominant = unique[counts.argmax()]
        name = class_names[dominant] if class_names else f"Class {dominant}"
        print(f"\n  ⚠️ COLLAPSE DETECTED: {name} has {max_frac*100:.1f}% of predictions")
        return True
    else:
        print(f"\n  ✅ No collapse (max class = {max_frac*100:.1f}%)")
        return False
