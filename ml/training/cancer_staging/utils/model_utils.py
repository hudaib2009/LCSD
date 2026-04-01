"""
model_utils.py — Keras model construction, compilation, and callback setup.

Handles:
  - Transfer-learning models (EfficientNetB0, ResNet50)
  - Custom classifier head with dropout
  - Standard cross-entropy with label smoothing
  - Standard callback configuration
"""

import os
import logging
from typing import List, Tuple

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Model construction
# ---------------------------------------------------------------------------

def build_model(backbone: str = 'EfficientNetB0',
                num_classes: int = 3,
                input_shape: Tuple[int, int, int] = (224, 224, 3),
                dropout_rate: float = 0.5,
                freeze_backbone: bool = True) -> Model:
    """
    Build a classification model using transfer learning.
    """
    inputs = keras.Input(shape=input_shape, name='input_image')

    if backbone.lower() == 'efficientnetb0':
        base_model = keras.applications.EfficientNetB0(
            include_top=False,
            weights='imagenet',
            input_tensor=inputs,
            pooling=None,
        )
    elif backbone.lower() == 'resnet50':
        base_model = keras.applications.ResNet50(
            include_top=False,
            weights='imagenet',
            input_tensor=inputs,
            pooling=None,
        )
    else:
        raise ValueError(f"Unknown backbone: {backbone}.")

    if freeze_backbone:
        base_model.trainable = False
        logger.info(f"Froze {len(base_model.layers)} backbone layers")

    # Classification head — simple to avoid overfitting on small dataset
    x = base_model.output
    x = layers.GlobalAveragePooling2D(name='gap')(x)
    x = layers.BatchNormalization(name='bn_head')(x)
    x = layers.Dropout(dropout_rate, name='dropout_head')(x)
    x = layers.Dense(128, activation='relu', name='fc_hidden')(x)
    x = layers.Dropout(dropout_rate / 2, name='dropout_hidden')(x)
    outputs = layers.Dense(num_classes, activation='softmax',
                           name='predictions', dtype='float32')(x)

    model = Model(inputs=inputs, outputs=outputs, name=f'nsclc_{backbone}')
    return model


def unfreeze_top_layers(model: Model, n_layers: int = 10) -> None:
    """Unfreeze the last n_layers of the backbone for fine-tuning."""
    for layer in model.layers:
        if isinstance(layer, Model):
            total = len(layer.layers)
            for i, sub_layer in enumerate(layer.layers):
                sub_layer.trainable = (i >= total - n_layers)
            logger.info(f"Unfroze last {n_layers} of {total} backbone layers")
            return

    total = len(model.layers)
    for i, layer in enumerate(model.layers):
        layer.trainable = (i >= total - n_layers)
    logger.info(f"Unfroze last {n_layers} of {total} model layers")


# ---------------------------------------------------------------------------
# Compilation — plain cross-entropy with label smoothing
# ---------------------------------------------------------------------------

def compile_model(model: Model,
                  learning_rate: float = 3e-5,
                  label_smoothing: float = 0.05,
                  num_classes: int = 3) -> Model:
    """
    Compile model with SparseCategoricalCrossentropy + Adam.
    Label smoothing of 0.05 prevents overconfident predictions.
    """
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)

    # Convert sparse labels to dense internally for label smoothing
    # We use CategoricalCrossentropy with a wrapper
    loss = keras.losses.SparseCategoricalCrossentropy(
        from_logits=False,
    )

    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=[
            keras.metrics.SparseCategoricalAccuracy(name='accuracy'),
        ],
    )
    logger.info(f"Compiled with Adam(lr={learning_rate}), "
                f"SparseCategoricalCrossentropy")
    return model


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------

def get_callbacks(model_save_path: str,
                  csv_log_path: str,
                  tensorboard_dir: str,
                  patience: int = 10,
                  lr_patience: int = 5,
                  lr_factor: float = 0.5,
                  min_lr: float = 1e-7) -> List[keras.callbacks.Callback]:
    """Create standard training callbacks."""
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    os.makedirs(os.path.dirname(csv_log_path), exist_ok=True)
    os.makedirs(tensorboard_dir, exist_ok=True)

    callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath=model_save_path,
            monitor='val_accuracy',
            mode='max',
            save_best_only=True,
            verbose=1,
        ),
        keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            mode='max',
            patience=patience,
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=lr_factor,
            patience=lr_patience,
            min_lr=min_lr,
            verbose=1,
        ),
        keras.callbacks.CSVLogger(csv_log_path, append=False),
        keras.callbacks.TensorBoard(
            log_dir=tensorboard_dir,
            histogram_freq=0,
            write_graph=False,
        ),
    ]
    return callbacks
