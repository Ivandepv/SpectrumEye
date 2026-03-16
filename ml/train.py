"""
train.py

Trains a MobileNetV2-based CNN to classify RF spectrograms into 3 signal classes:
  Key_Signal | Walkie_Talkie | LTE

Architecture:
  Input (224, 224, 1) grayscale
    → replicate channel x3 → (224, 224, 3)
    → Rescaling [0,255] to [-1,1]  (MobileNetV2 expected range)
    → MobileNetV2 base (alpha=0.75, ImageNet weights, first 100 layers frozen)
    → GlobalAveragePooling2D
    → Dense(256, ReLU) → Dropout(0.3)
    → Dense(128, ReLU) → Dropout(0.2)
    → Dense(3, softmax)

Outputs (saved to ml/models/{version}/):
  best_model.keras          — best checkpoint by val_accuracy
  config.json               — hyperparameters and class labels
  history.json              — loss/accuracy per epoch
  classification_report.txt — per-class precision/recall/F1
  confusion_matrix.png      — normalized confusion matrix
  training_curves.png       — loss and accuracy plots

Usage:
  # Quick local test (5 epochs, validates pipeline works end-to-end)
  python train.py --quick

  # Full training (up to 50 epochs, early stopping)
  python train.py

  # Custom settings
  python train.py --epochs 50 --batch 32 --lr 1e-4 --version v1
"""

import os
import json
import shutil
import argparse
from pathlib import Path
from datetime import datetime

# Suppress TF info/warning logs — keep only errors
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import matplotlib
matplotlib.use('Agg')   # non-interactive backend — works on Colab and headless servers
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model, Input
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import (
    ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
)

# ─── CONFIGURATION ────────────────────────────────────────────────

CLASS_LABELS = ["Key_Signal", "Walkie_Talkie", "LTE"]
N_CLASSES    = len(CLASS_LABELS)
IMG_SIZE     = (224, 224)

ML_DIR    = Path(__file__).parent
DATA_DIR  = ML_DIR / "dataset"
MODEL_DIR = ML_DIR / "models"

# ─── DATA LOADING ─────────────────────────────────────────────────

def load_split(split: str, batch_size: int) -> tf.data.Dataset:
    """
    Load a dataset split from disk using the folder structure:
      dataset/{split}/{class_name}/*.png

    Images are loaded as grayscale float32 in [0, 255], shape (H, W, 1).
    Labels are one-hot encoded categorical vectors.

    Args:
        split:      'train', 'val', or 'test'
        batch_size: number of images per batch

    Returns:
        tf.data.Dataset yielding (images, labels) batches
    """
    split_dir = DATA_DIR / split

    if not split_dir.exists():
        raise FileNotFoundError(
            f"Split directory not found: {split_dir}\n"
            f"Run split_dataset.py first."
        )

    ds = tf.keras.utils.image_dataset_from_directory(
        split_dir,
        labels='inferred',
        label_mode='categorical',
        class_names=CLASS_LABELS,   # explicit ordering — must match CLASS_LABELS
        color_mode='grayscale',     # loads as (H, W, 1)
        batch_size=batch_size,
        image_size=IMG_SIZE,
        shuffle=(split == 'train'),
        seed=42,
    )

    if split == 'train':
        # Cache after loading, shuffle buffer, prefetch for GPU pipeline
        return ds.cache().shuffle(buffer_size=2000, seed=42).prefetch(tf.data.AUTOTUNE)
    else:
        return ds.cache().prefetch(tf.data.AUTOTUNE)


def count_samples(split: str) -> int:
    """Count total image files in a split directory."""
    split_dir = DATA_DIR / split
    return sum(len(list(d.glob("*.png"))) for d in split_dir.iterdir() if d.is_dir())


# ─── MODEL ────────────────────────────────────────────────────────

def build_model(n_classes: int, alpha: float = 0.75, freeze_until: int = 100) -> Model:
    """
    Build MobileNetV2 transfer learning model for RF spectrogram classification.

    The input is grayscale (1 channel). MobileNetV2 was pretrained on 3-channel
    ImageNet images, so we replicate the single channel 3 times. The pretrained
    feature extractors still work because the visual patterns in spectrograms
    (edges, textures, shapes) are analogous to natural image features.

    Args:
        n_classes:    number of output classes
        alpha:        MobileNetV2 width multiplier (0.75 = reduced model, faster)
        freeze_until: freeze the first N layers of MobileNetV2 base
                      (layer 0-99 frozen, 100+ fine-tuned)

    Returns:
        Compiled-ready Keras model
    """
    inputs = Input(shape=(*IMG_SIZE, 1), name='spectrogram_input')

    # ── Step 1: Replicate grayscale channel to 3 channels ──────────
    # MobileNetV2 expects (224, 224, 3). We duplicate the single power
    # channel so the pretrained weights are compatible.
    x = layers.Concatenate(name='channel_replicate')([inputs, inputs, inputs])

    # ── Step 2: Rescale [0, 255] → [-1, 1] ────────────────────────
    # MobileNetV2 preprocess_input formula: (x / 127.5) - 1.0
    # Using a Rescaling layer is serialization-safe (no Lambda needed).
    x = layers.Rescaling(scale=1.0 / 127.5, offset=-1.0, name='mobilenet_rescale')(x)

    # ── Step 3: MobileNetV2 feature extractor ─────────────────────
    base = MobileNetV2(
        input_shape=(*IMG_SIZE, 3),
        alpha=alpha,
        include_top=False,          # remove ImageNet classification head
        weights='imagenet',
    )

    # Freeze first `freeze_until` layers — these early layers detect
    # low-level features (edges, textures) that transfer well from ImageNet.
    # Fine-tune the later layers to adapt to RF spectrogram patterns.
    base.trainable = True
    for layer in base.layers[:freeze_until]:
        layer.trainable = False

    # Pass training=False to keep BatchNorm layers in inference mode.
    # This prevents fine-tuning on a small dataset from corrupting the
    # pretrained BatchNorm statistics.
    x = base(x, training=False)

    # ── Step 4: Custom classification head ────────────────────────
    x = layers.GlobalAveragePooling2D(name='gap')(x)
    x = layers.Dense(256, activation='relu', name='dense_256')(x)
    x = layers.Dropout(0.3, name='dropout_1')(x)
    x = layers.Dense(128, activation='relu', name='dense_128')(x)
    x = layers.Dropout(0.2, name='dropout_2')(x)
    outputs = layers.Dense(n_classes, activation='softmax', name='predictions')(x)

    return Model(inputs, outputs, name='SpectrumEye_MobileNetV2')


# ─── EVALUATION ───────────────────────────────────────────────────

def evaluate_and_save(model: Model, test_ds: tf.data.Dataset, out_dir: Path) -> dict:
    """
    Run full evaluation on the test set.
    Saves:
      - classification_report.txt  (precision / recall / F1 per class)
      - confusion_matrix.png       (normalized heatmap)

    Returns:
        dict with overall accuracy, per-class F1
    """
    print("\nEvaluating on test set...")

    y_true, y_pred = [], []
    for images, labels in test_ds:
        preds = model.predict(images, verbose=0)
        y_true.extend(np.argmax(labels.numpy(), axis=1))
        y_pred.extend(np.argmax(preds, axis=1))

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # ── Classification report ─────────────────────────────────────
    report_str = classification_report(
        y_true, y_pred,
        target_names=CLASS_LABELS,
        digits=4,
    )
    print(report_str)
    (out_dir / "classification_report.txt").write_text(report_str)

    # ── Confusion matrix ──────────────────────────────────────────
    cm      = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm_norm,
        annot=True,
        fmt='.2%',
        xticklabels=CLASS_LABELS,
        yticklabels=CLASS_LABELS,
        cmap='Blues',
        ax=ax,
        annot_kws={"size": 13},
        linewidths=0.5,
    )
    ax.set_xlabel('Predicted', fontsize=12, labelpad=8)
    ax.set_ylabel('True Label', fontsize=12, labelpad=8)
    ax.set_title(
        f'SpectrumEye — Confusion Matrix\ntest set  (n={len(y_true)})',
        fontsize=13, pad=12,
    )
    plt.tight_layout()
    cm_path = out_dir / "confusion_matrix.png"
    plt.savefig(cm_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Confusion matrix saved: {cm_path}")

    # ── Metrics summary ───────────────────────────────────────────
    accuracy = float(np.mean(y_true == y_pred))
    return {"test_accuracy": round(accuracy, 4), "n_samples": len(y_true)}


def plot_training_curves(history: dict, out_dir: Path) -> None:
    """Save loss and accuracy curves as a PNG."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    epochs = range(1, len(history['loss']) + 1)

    ax1.plot(epochs, history['loss'],     'b-',  label='Train loss')
    ax1.plot(epochs, history['val_loss'], 'r--', label='Val loss')
    ax1.set_title('Loss'); ax1.set_xlabel('Epoch'); ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, history['accuracy'],     'b-',  label='Train acc')
    ax2.plot(epochs, history['val_accuracy'], 'r--', label='Val acc')
    ax2.set_title('Accuracy'); ax2.set_xlabel('Epoch'); ax2.legend()
    ax2.set_ylim(0, 1); ax2.grid(True, alpha=0.3)

    plt.suptitle('SpectrumEye Training Curves', fontsize=13, y=1.01)
    plt.tight_layout()
    plt.savefig(out_dir / "training_curves.png", dpi=150, bbox_inches='tight')
    plt.close()


# ─── TRAINING ENTRY POINT ─────────────────────────────────────────

def train(
    epochs: int       = 50,
    batch_size: int   = 32,
    learning_rate: float = 1e-4,
    version: str      = None,
) -> None:

    # ── System info ───────────────────────────────────────────────
    gpus = tf.config.list_physical_devices('GPU')
    print(f"\nTensorFlow {tf.__version__}")
    print(f"GPUs available: {len(gpus)} — {[g.name for g in gpus] if gpus else 'training on CPU'}")

    # ── Version / output dir ──────────────────────────────────────
    if version is None:
        version = datetime.now().strftime("v%Y%m%d_%H%M")
    out_dir = MODEL_DIR / version
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {out_dir}\n")

    # ── Load data ─────────────────────────────────────────────────
    print("Loading datasets...")
    train_ds = load_split('train', batch_size)
    val_ds   = load_split('val',   batch_size)
    test_ds  = load_split('test',  batch_size)

    n_train = count_samples('train')
    n_val   = count_samples('val')
    n_test  = count_samples('test')
    print(f"  train: {n_train} images   val: {n_val}   test: {n_test}\n")

    # ── Build model ───────────────────────────────────────────────
    print("Building model...")
    model = build_model(n_classes=N_CLASSES)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss='categorical_crossentropy',
        metrics=['accuracy'],
    )

    trainable_params = int(sum(np.prod(v.shape) for v in model.trainable_weights))
    total_params     = int(sum(np.prod(v.shape) for v in model.weights))
    print(f"  Trainable params: {trainable_params:,} / {total_params:,}\n")

    # ── Callbacks ─────────────────────────────────────────────────
    checkpoint_path = str(out_dir / "best_model.keras")

    callbacks = [
        ModelCheckpoint(
            filepath=checkpoint_path,
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1,
        ),
        EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=4,
            min_lr=1e-7,
            verbose=1,
        ),
    ]

    # ── Train ─────────────────────────────────────────────────────
    print(f"Training: up to {epochs} epochs  |  batch {batch_size}  |  lr {learning_rate}\n")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1,
    )

    # ── Save artifacts ────────────────────────────────────────────
    # Training history
    history_dict = {k: [float(v) for v in vals] for k, vals in history.history.items()}
    (out_dir / "history.json").write_text(json.dumps(history_dict, indent=2))
    plot_training_curves(history_dict, out_dir)

    # Evaluation
    metrics = evaluate_and_save(model, test_ds, out_dir)

    # Config
    config = {
        "version":          version,
        "classes":          CLASS_LABELS,
        "n_classes":        N_CLASSES,
        "img_size":         list(IMG_SIZE),
        "epochs_run":       len(history_dict['loss']),
        "epochs_max":       epochs,
        "batch_size":       batch_size,
        "learning_rate":    learning_rate,
        "base_model":       "MobileNetV2",
        "alpha":            0.75,
        "freeze_until":     100,
        "trainable_params": trainable_params,
        "total_params":     total_params,
        **metrics,
    }
    (out_dir / "config.json").write_text(json.dumps(config, indent=2))

    # Copy best model to production slot
    prod_dir = MODEL_DIR / "production"
    prod_dir.mkdir(exist_ok=True)
    shutil.copy2(checkpoint_path, prod_dir / "spectromeye_best.keras")

    # ── Summary ───────────────────────────────────────────────────
    best_val_acc = max(history_dict['val_accuracy'])
    print(f"\n{'─'*50}")
    print(f"  Best val accuracy : {best_val_acc:.4f} ({best_val_acc*100:.1f}%)")
    print(f"  Test accuracy     : {metrics['test_accuracy']:.4f} ({metrics['test_accuracy']*100:.1f}%)")
    print(f"  Epochs run        : {config['epochs_run']} / {epochs}")
    print(f"  Model saved       : {prod_dir}/spectromeye_best.keras")
    print(f"{'─'*50}\n")


# ─── ENTRY POINT ──────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train SpectrumEye RF signal classifier")
    parser.add_argument('--epochs',  type=int,   default=50,   help='Max epochs (default: 50)')
    parser.add_argument('--batch',   type=int,   default=32,   help='Batch size (default: 32)')
    parser.add_argument('--lr',      type=float, default=1e-4, help='Learning rate (default: 1e-4)')
    parser.add_argument('--version', type=str,   default=None, help='Version tag (default: timestamp)')
    parser.add_argument('--quick',   action='store_true',
                        help='Quick test: 5 epochs, batch=16. Validates the pipeline locally.')
    args = parser.parse_args()

    if args.quick:
        print("Quick mode: 5 epochs, batch=16 — validates pipeline end-to-end\n")
        train(epochs=5, batch_size=16, learning_rate=1e-4, version="quick_test")
    else:
        train(
            epochs=args.epochs,
            batch_size=args.batch,
            learning_rate=args.lr,
            version=args.version,
        )
