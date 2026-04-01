import os, sys, json
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

print(f'TensorFlow: {tf.__version__}')
gpus = tf.config.list_physical_devices('GPU')
print(f'GPUs: {len(gpus)}')
for g in gpus:
    tf.config.experimental.set_memory_growth(g, True)
# ══════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════
DATA_DIR        = 'processed_data'
MODEL_DIR       = 'models'
FIGURES_DIR     = 'figures'

BACKBONE        = 'EfficientNetB0'
NUM_CLASSES     = 3
IMG_SIZE        = (224, 224)
BATCH_SIZE      = 16
DROPOUT         = 0.5

LR_PHASE1       = 1e-4
EPOCHS_PHASE1   = 30
LR_PHASE2       = 1e-5
EPOCHS_PHASE2   = 30
UNFREEZE_LAYERS = 10

STAGE_NAMES     = ['Stage I', 'Stage II', 'Stage III']
MODEL_PATH      = os.path.join(MODEL_DIR, 'nsclc_stage_classifier.keras')
CSV_LOG_PATH    = os.path.join(MODEL_DIR, 'training_log.csv')
TB_DIR          = os.path.join(MODEL_DIR, 'tensorboard_logs')

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

print('Strategy: balanced batches, NO /255 (EfficientNet does it internally)')
# ══════════════════════════════════════════════════════════════════
# UTILITY FUNCTIONS
# ══════════════════════════════════════════════════════════════════

def load_and_preprocess(path, label, target_size=IMG_SIZE):
    """
    Load PNG → float32 [0, 255] → 3-channel.
    DO NOT divide by 255 — EfficientNetB0 has internal Rescaling(1/255).
    """
    raw = tf.io.read_file(path)
    image = tf.io.decode_png(raw, channels=1)
    image = tf.image.resize(image, target_size)
    image = tf.cast(image, tf.float32)  # [0, 255] — NOT /255!
    image = tf.repeat(image, repeats=3, axis=-1)
    return image, label


def augment_fn(image, label):
    """Moderate augmentation (values stay in [0, 255])."""
    image = tf.image.random_flip_left_right(image)
    # Small rotation ±10°
    angle = tf.random.uniform([], -10.0, 10.0) * (3.14159265 / 180.0)
    cos_a = tf.cos(angle)
    sin_a = tf.sin(angle)
    h = tf.cast(tf.shape(image)[0], tf.float32)
    w = tf.cast(tf.shape(image)[1], tf.float32)
    cx, cy = w/2.0, h/2.0
    transform = tf.cast(tf.stack([
        cos_a, -sin_a, cx - cx*cos_a + cy*sin_a,
        sin_a,  cos_a, cy - cx*sin_a - cy*cos_a,
        0.0, 0.0
    ]), tf.float32)
    img4d = tf.expand_dims(image, 0)
    rotated = tf.raw_ops.ImageProjectiveTransformV3(
        images=img4d, transforms=tf.expand_dims(transform, 0),
        output_shape=tf.shape(image)[:2],
        interpolation='BILINEAR', fill_mode='CONSTANT', fill_value=0.0)
    image = tf.squeeze(rotated, 0)
    # Brightness/contrast in [0,255] range
    image = tf.image.random_brightness(image, max_delta=20.0)  # ±20 out of 255
    image = tf.image.random_contrast(image, lower=0.92, upper=1.08)
    image = tf.clip_by_value(image, 0.0, 255.0)
    return image, label


def build_balanced_dataset(csv_path, base_dir, batch_size=16,
                           do_augment=False, seed=42):
    """
    Class-balanced dataset via sample_from_datasets.
    Returns (dataset, steps_per_epoch).
    """
    df = pd.read_csv(csv_path)
    classes = sorted(df['label'].unique())
    n_classes = len(classes)
    
    per_class = []
    class_counts = []
    for cls in classes:
        cdf = df[df['label'] == cls]
        paths = [os.path.join(base_dir, p) for p in cdf['path_image']]
        labels = cdf['label'].values.astype(np.int32)
        class_counts.append(len(cdf))
        ds = tf.data.Dataset.from_tensor_slices((paths, labels))
        ds = ds.shuffle(len(paths), seed=seed, reshuffle_each_iteration=True)
        ds = ds.repeat()
        per_class.append(ds)
        print(f'  Class {cls} ({STAGE_NAMES[cls]}): {len(cdf)} slices')
    
    weights = [1.0/n_classes] * n_classes
    balanced = tf.data.Dataset.sample_from_datasets(per_class, weights=weights, seed=seed)
    
    min_count = min(class_counts)
    epoch_size = min_count * n_classes
    steps = epoch_size // batch_size
    print(f'  Balanced epoch: {epoch_size} samples ({min_count}/class), {steps} steps')
    
    balanced = balanced.map(load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    if do_augment:
        balanced = balanced.map(augment_fn, num_parallel_calls=tf.data.AUTOTUNE)
    balanced = balanced.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return balanced, steps


def build_dataset(csv_path, base_dir, batch_size=16, shuffle=False):
    """Standard dataset for val/test."""
    df = pd.read_csv(csv_path)
    paths = [os.path.join(base_dir, p) for p in df['path_image']]
    labels = df['label'].values.astype(np.int32)
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if shuffle:
        ds = ds.shuffle(len(paths))
    ds = ds.map(load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


def detect_collapse(y_prob, y_pred=None, threshold=0.9, class_names=None):
    if y_pred is None:
        y_pred = y_prob.argmax(axis=1)
    n = len(y_pred)
    unique, counts = np.unique(y_pred, return_counts=True)
    print(f'  Mean softmax: {y_prob.mean(axis=0)}')
    print(f'  Std softmax:  {y_prob.std(axis=0)}')
    for cls, cnt in zip(unique, counts):
        name = class_names[cls] if class_names else f'Class {cls}'
        print(f'    {name}: {cnt}/{n} ({cnt/n*100:.1f}%)')
    max_frac = counts.max() / n
    if max_frac >= threshold:
        dom = unique[counts.argmax()]
        name = class_names[dom] if class_names else f'Class {dom}'
        print(f'  ⚠️ COLLAPSE: {name} = {max_frac*100:.1f}%')
        return True
    print(f'  ✅ No collapse (max={max_frac*100:.1f}%)')
    return False


print('✅ Functions defined')
# ── Label distributions ──────────────────────────────────────────
train_csv = os.path.join(DATA_DIR, 'train_slices.csv')
val_csv   = os.path.join(DATA_DIR, 'val_slices.csv')

for name, path in [('Train', train_csv), ('Val', val_csv)]:
    df = pd.read_csv(path)
    dist = dict(df['label'].value_counts().sort_index())
    pts = dict(df.groupby('label')['patient_id'].nunique())
    print(f'{name}: {len(df)} slices | dist={dist} | patients={pts}')
# ── Build BALANCED training + standard val ───────────────────────
print('Building balanced train dataset:')
train_ds, steps_per_epoch = build_balanced_dataset(
    train_csv, DATA_DIR, batch_size=BATCH_SIZE, do_augment=True
)

print('\nBuilding val dataset:')
val_ds = build_dataset(val_csv, DATA_DIR, batch_size=BATCH_SIZE)
# ── Verify: images must be [0, 255], NOT [0, 1] ──────────────────
print('── Image range check ──')
for imgs, labels in train_ds.take(1):
    mn = tf.reduce_min(imgs).numpy()
    mx = tf.reduce_max(imgs).numpy()
    me = tf.reduce_mean(imgs).numpy()
    print(f'  shape={tuple(imgs.shape)}, min={mn:.1f}, max={mx:.1f}, mean={me:.1f}')
    if mx <= 1.0:
        print('  ❌ ERROR: Images in [0,1] — EfficientNet will double-normalize!')
    elif mx > 1.0:
        print('  ✅ Images in [0,255] range — correct for EfficientNet')
    print(f'  Labels in batch: {sorted(set(labels.numpy().tolist()))}')

# Verify balance
lc = {}
for imgs, labels in train_ds.take(steps_per_epoch):
    for l in labels.numpy():
        lc[int(l)] = lc.get(int(l), 0) + 1
print(f'\nBalance check: {dict(sorted(lc.items()))}')
vals = list(lc.values())
ratio = max(vals) / max(min(vals), 1)
print(f'Ratio: {ratio:.2f} {"✅" if ratio < 1.5 else "⚠️"}')
# ── Build model ──────────────────────────────────────────────────
from tensorflow import keras
from tensorflow.keras import layers, Model

def build_model(num_classes=3, input_shape=(224,224,3),
                dropout_rate=0.5, freeze_backbone=True):
    inputs = keras.Input(shape=input_shape, name='input_image')
    # EfficientNetB0 has internal Rescaling(1/255) + Normalization
    # Feed it [0, 255] images!
    base = keras.applications.EfficientNetB0(
        include_top=False, weights='imagenet',
        input_tensor=inputs, pooling=None)
    if freeze_backbone:
        base.trainable = False
    x = base.output
    x = layers.GlobalAveragePooling2D(name='gap')(x)
    x = layers.BatchNormalization(name='bn_head')(x)
    x = layers.Dropout(dropout_rate, name='drop1')(x)
    x = layers.Dense(128, activation='relu', name='fc')(x)
    x = layers.Dropout(dropout_rate/2, name='drop2')(x)
    outputs = layers.Dense(num_classes, activation='softmax',
                           name='predictions', dtype='float32')(x)
    return Model(inputs=inputs, outputs=outputs)

model = build_model(num_classes=NUM_CLASSES, dropout_rate=DROPOUT)

last = model.layers[-1]
print(f'Final: {last.name}, units={last.units}')
assert last.units == NUM_CLASSES
print(f'Params: {model.count_params():,}')
# ══════════════════════════════════════════════════════════════════
# Phase 1: frozen backbone
# ══════════════════════════════════════════════════════════════════
print(f'PHASE 1: Head only | LR={LR_PHASE1} | balanced batches | [0,255]')

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=LR_PHASE1),
    loss=keras.losses.SparseCategoricalCrossentropy(),
    metrics=[keras.metrics.SparseCategoricalAccuracy(name='accuracy')]
)

callbacks_p1 = [
    keras.callbacks.ModelCheckpoint(MODEL_PATH, monitor='val_accuracy',
                                    mode='max', save_best_only=True, verbose=1),
    keras.callbacks.EarlyStopping(monitor='val_accuracy', mode='max',
                                  patience=12, restore_best_weights=True, verbose=1),
    keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                                      patience=4, min_lr=1e-7, verbose=1),
    keras.callbacks.CSVLogger(CSV_LOG_PATH, append=False),
]

history_p1 = model.fit(
    train_ds, validation_data=val_ds,
    epochs=EPOCHS_PHASE1,
    steps_per_epoch=steps_per_epoch,
    callbacks=callbacks_p1, verbose=1,
)
# ── Phase 1 collapse check ───────────────────────────────────────
print('── Phase 1 collapse check ──')
vp = []
for imgs, labels in val_ds:
    vp.append(model.predict(imgs, verbose=0))
vp = np.vstack(vp)
detect_collapse(vp, class_names=STAGE_NAMES)
# ══════════════════════════════════════════════════════════════════
# Phase 2: fine-tune
# ══════════════════════════════════════════════════════════════════
print(f'PHASE 2: Unfreeze last {UNFREEZE_LAYERS} layers | LR={LR_PHASE2}')

for layer in model.layers:
    if isinstance(layer, Model):
        total = len(layer.layers)
        for i, sub in enumerate(layer.layers):
            sub.trainable = (i >= total - UNFREEZE_LAYERS)
        print(f'  Unfroze last {UNFREEZE_LAYERS} of {total} backbone layers')
        break

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=LR_PHASE2),
    loss=keras.losses.SparseCategoricalCrossentropy(),
    metrics=[keras.metrics.SparseCategoricalAccuracy(name='accuracy')]
)

csv_log_p2 = os.path.join(MODEL_DIR, 'training_log_phase2.csv')
callbacks_p2 = [
    keras.callbacks.ModelCheckpoint(MODEL_PATH, monitor='val_accuracy',
                                    mode='max', save_best_only=True, verbose=1),
    keras.callbacks.EarlyStopping(monitor='val_accuracy', mode='max',
                                  patience=15, restore_best_weights=True, verbose=1),
    keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                                      patience=5, min_lr=1e-7, verbose=1),
    keras.callbacks.CSVLogger(csv_log_p2, append=False),
]

history_p2 = model.fit(
    train_ds, validation_data=val_ds,
    epochs=EPOCHS_PHASE2,
    steps_per_epoch=steps_per_epoch,
    callbacks=callbacks_p2, verbose=1,
)
# ── Phase 2 collapse check ───────────────────────────────────────
print('── Phase 2 collapse check ──')
vp2 = []
for imgs, labels in val_ds:
    vp2.append(model.predict(imgs, verbose=0))
vp2 = np.vstack(vp2)
detect_collapse(vp2, class_names=STAGE_NAMES)
# ── Training curves ──────────────────────────────────────────────
def plot_history(h1, h2, metric, ylabel, title, save_path):
    tv = h1.history[metric] + h2.history[metric]
    vv = h1.history[f'val_{metric}'] + h2.history[f'val_{metric}']
    ep = range(1, len(tv) + 1)
    p1 = len(h1.history[metric])
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(ep, tv, 'b-', label='Train')
    ax.plot(ep, vv, 'r-', label='Val')
    ax.axvline(x=p1, color='gray', ls='--', alpha=0.7, label='Phase 1→2')
    ax.set_xlabel('Epoch'); ax.set_ylabel(ylabel)
    ax.set_title(title); ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout(); fig.savefig(save_path, dpi=150); plt.show()

plot_history(history_p1, history_p2, 'loss', 'Loss',
             'Training & Validation Loss',
             os.path.join(FIGURES_DIR, 'training_loss.png'))
plot_history(history_p1, history_p2, 'accuracy', 'Accuracy',
             'Training & Validation Accuracy',
             os.path.join(FIGURES_DIR, 'training_accuracy.png'))
print(f'\n✅ Model saved to: {MODEL_PATH}')
print(f'   Size: {os.path.getsize(MODEL_PATH) / 1e6:.1f} MB')
