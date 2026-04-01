"""
eval_utils.py — Evaluation metrics, ROC curves, confusion matrices,
patient-level aggregation, and Grad-CAM.

This module handles:
  - Slice-level and patient-level metric computation
  - Confusion matrix and classification report generation
  - ROC AUC (one-vs-rest) plotting
  - Patient-level prediction aggregation (mean/max pooling)
  - Grad-CAM heatmap generation and overlay
"""

import os
import json
import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Metrics computation
# ---------------------------------------------------------------------------

def compute_metrics(y_true: np.ndarray,
                    y_pred: np.ndarray,
                    y_prob: np.ndarray,
                    class_names: Optional[List[str]] = None
                    ) -> Dict:
    """
    Compute classification metrics.

    Parameters
    ----------
    y_true : np.ndarray
        True integer labels (N,).
    y_pred : np.ndarray
        Predicted integer labels (N,).
    y_prob : np.ndarray
        Predicted probabilities (N, num_classes).
    class_names : list of str, optional
        Names for each class.

    Returns
    -------
    dict
        Contains 'accuracy', 'macro_f1', 'weighted_f1',
        'classification_report' (dict), 'roc_auc' (dict or None).
    """
    from sklearn.metrics import (accuracy_score, f1_score,
                                 classification_report, roc_auc_score)

    if class_names is None:
        class_names = [f'Stage_{i}' for i in range(y_prob.shape[1])]

    # Only include classes that exist in y_true for proper evaluation
    present_classes = sorted(set(y_true))
    target_names = [class_names[i] for i in present_classes]

    report = classification_report(y_true, y_pred,
                                   labels=present_classes,
                                   target_names=target_names,
                                   output_dict=True,
                                   zero_division=0)

    metrics = {
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'macro_f1': float(f1_score(y_true, y_pred, average='macro',
                                    labels=present_classes,
                                    zero_division=0)),
        'weighted_f1': float(f1_score(y_true, y_pred, average='weighted',
                                       labels=present_classes,
                                       zero_division=0)),
        'classification_report': report,
    }

    # ROC AUC (one-vs-rest) — only if more than 1 class present
    if len(present_classes) > 1:
        try:
            # Filter probabilities to present classes for OVR
            auc = roc_auc_score(y_true, y_prob, multi_class='ovr',
                                average='macro',
                                labels=present_classes)
            metrics['roc_auc_macro'] = float(auc)
        except Exception as e:
            logger.warning(f"Could not compute ROC AUC: {e}")
            metrics['roc_auc_macro'] = None
    else:
        metrics['roc_auc_macro'] = None

    return metrics


# ---------------------------------------------------------------------------
# Confusion matrix
# ---------------------------------------------------------------------------

def plot_confusion_matrix(y_true: np.ndarray,
                          y_pred: np.ndarray,
                          class_names: List[str],
                          save_dir: str,
                          prefix: str = '') -> None:
    """
    Plot and save normalized + raw confusion matrices.

    Parameters
    ----------
    y_true, y_pred : np.ndarray
    class_names : list of str
    save_dir : str
    prefix : str
        Filename prefix (e.g. 'slice_' or 'patient_').
    """
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

    os.makedirs(save_dir, exist_ok=True)
    present = sorted(set(y_true) | set(y_pred))
    names = [class_names[i] for i in present]

    for normalize, suffix in [(None, 'raw'), ('true', 'normalized')]:
        cm = confusion_matrix(y_true, y_pred, labels=present,
                              normalize=normalize)
        fig, ax = plt.subplots(figsize=(8, 6))
        disp = ConfusionMatrixDisplay(cm, display_labels=names)
        fmt = '.2f' if normalize else 'd'
        disp.plot(ax=ax, cmap='Blues', values_format=fmt)
        ax.set_title(f'{prefix}Confusion Matrix ({suffix})')
        plt.tight_layout()
        path = os.path.join(save_dir, f'{prefix}confusion_matrix_{suffix}.png')
        fig.savefig(path, dpi=150)
        plt.close(fig)
        logger.info(f"Saved {path}")


# ---------------------------------------------------------------------------
# ROC curves
# ---------------------------------------------------------------------------

def plot_roc_curves(y_true: np.ndarray,
                    y_prob: np.ndarray,
                    class_names: List[str],
                    save_path: str) -> None:
    """
    Plot one-vs-rest ROC curves for each class.

    Parameters
    ----------
    y_true : np.ndarray, shape (N,)
    y_prob : np.ndarray, shape (N, C)
    class_names : list of str
    save_path : str
    """
    from sklearn.metrics import roc_curve, auc
    from sklearn.preprocessing import label_binarize

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    present = sorted(set(y_true))
    n_classes = y_prob.shape[1]

    y_bin = label_binarize(y_true, classes=list(range(n_classes)))
    if y_bin.ndim == 1:
        y_bin = np.column_stack([1 - y_bin, y_bin])

    fig, ax = plt.subplots(figsize=(8, 6))
    for i in present:
        if i >= n_classes:
            continue
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_prob[:, i])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, label=f'{class_names[i]} (AUC={roc_auc:.3f})')

    ax.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curves (One-vs-Rest)')
    ax.legend(loc='lower right')
    plt.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    logger.info(f"Saved ROC curves to {save_path}")


# ---------------------------------------------------------------------------
# Patient-level aggregation
# ---------------------------------------------------------------------------

def aggregate_patient_predictions(patient_ids: np.ndarray,
                                  y_prob: np.ndarray,
                                  y_true_slices: np.ndarray,
                                  method: str = 'mean'
                                  ) -> Tuple[np.ndarray, np.ndarray,
                                             np.ndarray, np.ndarray]:
    """
    Aggregate slice-level probabilities to patient-level.

    Parameters
    ----------
    patient_ids : np.ndarray, shape (N,)
        Patient ID per slice.
    y_prob : np.ndarray, shape (N, C)
        Slice-level predicted probabilities.
    y_true_slices : np.ndarray, shape (N,)
        True label per slice (same for all slices of a patient).
    method : str
        'mean' or 'max' pooling.

    Returns
    -------
    unique_ids : np.ndarray
    patient_probs : np.ndarray, shape (P, C)
    patient_preds : np.ndarray, shape (P,)
    patient_labels : np.ndarray, shape (P,)
    """
    unique_ids = np.unique(patient_ids)
    n_classes = y_prob.shape[1]
    patient_probs = np.zeros((len(unique_ids), n_classes))
    patient_labels = np.zeros(len(unique_ids), dtype=np.int32)

    for i, pid in enumerate(unique_ids):
        mask = patient_ids == pid
        probs = y_prob[mask]
        if method == 'mean':
            patient_probs[i] = probs.mean(axis=0)
        elif method == 'max':
            patient_probs[i] = probs.max(axis=0)
        else:
            patient_probs[i] = probs.mean(axis=0)
        patient_labels[i] = y_true_slices[mask][0]

    patient_preds = patient_probs.argmax(axis=1)
    return unique_ids, patient_probs, patient_preds, patient_labels


# ---------------------------------------------------------------------------
# Grad-CAM
# ---------------------------------------------------------------------------

def generate_gradcam(model: keras.Model,
                     image: np.ndarray,
                     layer_name: Optional[str] = None,
                     pred_index: Optional[int] = None
                     ) -> np.ndarray:
    """
    Generate a Grad-CAM heatmap for a single image.

    Parameters
    ----------
    model : keras.Model
    image : np.ndarray
        Single image, shape (H, W, 3), float32 in [0, 1].
    layer_name : str, optional
        Name of the convolutional layer to use. If None, uses
        the last Conv2D layer in the model.
    pred_index : int, optional
        Class index to compute Grad-CAM for. If None, uses the
        predicted class.

    Returns
    -------
    np.ndarray
        Heatmap of shape (H, W), float32 in [0, 1].
    """
    if layer_name is None:
        # Find last Conv2D layer
        for layer in reversed(model.layers):
            if 'conv' in layer.name.lower():
                layer_name = layer.name
                break
            if isinstance(layer, keras.Model):
                # Search inside backbone
                for sub in reversed(layer.layers):
                    if 'conv' in sub.name.lower():
                        layer_name = sub.name
                        break
                if layer_name is not None:
                    break
        if layer_name is None:
            raise ValueError("Could not find a Conv2D layer for Grad-CAM")

    # Build gradient model
    # Handle nested models (backbone inside main model)
    try:
        target_layer = model.get_layer(layer_name)
    except ValueError:
        # Try searching inside sub-models
        target_layer = None
        for layer in model.layers:
            if isinstance(layer, keras.Model):
                try:
                    target_layer = layer.get_layer(layer_name)
                    break
                except ValueError:
                    continue
        if target_layer is None:
            raise ValueError(f"Layer '{layer_name}' not found in model")

    grad_model = keras.Model(
        inputs=model.input,
        outputs=[target_layer.output, model.output]
    )

    img_tensor = tf.expand_dims(tf.cast(image, tf.float32), 0)

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_tensor)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.nn.relu(heatmap)
    heatmap = heatmap / (tf.reduce_max(heatmap) + 1e-8)

    # Resize to original image size
    heatmap = tf.image.resize(
        heatmap[..., tf.newaxis],
        (image.shape[0], image.shape[1])
    )
    return tf.squeeze(heatmap).numpy()


def save_gradcam_overlay(image: np.ndarray,
                         heatmap: np.ndarray,
                         save_path: str,
                         alpha: float = 0.4,
                         title: str = '') -> None:
    """
    Overlay Grad-CAM heatmap on the original image and save.

    Parameters
    ----------
    image : np.ndarray
        Original image (H, W, 3), float32 in [0, 1].
    heatmap : np.ndarray
        Grad-CAM heatmap (H, W), float32 in [0, 1].
    save_path : str
    alpha : float
        Heatmap overlay opacity.
    title : str
        Plot title.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Original
    axes[0].imshow(image[:, :, 0], cmap='gray')
    axes[0].set_title('Original')
    axes[0].axis('off')

    # Heatmap
    axes[1].imshow(heatmap, cmap='jet')
    axes[1].set_title('Grad-CAM Heatmap')
    axes[1].axis('off')

    # Overlay
    axes[2].imshow(image[:, :, 0], cmap='gray')
    axes[2].imshow(heatmap, cmap='jet', alpha=alpha)
    axes[2].set_title('Overlay')
    axes[2].axis('off')

    if title:
        fig.suptitle(title, fontsize=14)

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"Saved Grad-CAM overlay to {save_path}")


# ---------------------------------------------------------------------------
# Metric saving
# ---------------------------------------------------------------------------

def save_metrics(metrics: Dict,
                 save_dir: str,
                 prefix: str = '') -> None:
    """
    Save metrics dict as JSON and key values as CSV.

    Parameters
    ----------
    metrics : dict
    save_dir : str
    prefix : str
    """
    import pandas as pd

    os.makedirs(save_dir, exist_ok=True)

    # JSON
    json_path = os.path.join(save_dir, f'{prefix}metrics.json')
    with open(json_path, 'w') as f:
        json.dump(metrics, f, indent=2, default=str)
    logger.info(f"Saved metrics to {json_path}")

    # CSV summary
    summary = {k: v for k, v in metrics.items()
               if not isinstance(v, dict)}
    csv_path = os.path.join(save_dir, f'{prefix}metrics_summary.csv')
    pd.DataFrame([summary]).to_csv(csv_path, index=False)
    logger.info(f"Saved summary to {csv_path}")
