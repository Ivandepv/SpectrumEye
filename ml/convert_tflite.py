"""
ml/convert_tflite.py — Convert best_model.keras to INT8 TFLite

Run on dev machine (Python 3.12, TF 2.18+) from the project root:
    python ml/convert_tflite.py

Output: ml/models/production/best_model.tflite

Deploy to Pi 5 — copy the .tflite file alongside the .keras file:
    scp ml/models/production/best_model.tflite porphyras@192.168.0.156:~/Desktop/SpectrumEye/ml/models/production/

classifier.py auto-detects the .tflite file and uses it instead of the .keras model.
Expected speedup on Pi 5: ~4–8x (from ~500ms to ~60–120ms per frame).
"""

import numpy as np
from pathlib import Path

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf

# ─── PATHS ───────────────────────────────────────────────────────

KERAS_PATH  = Path("ml/models/production/best_model.keras")
OUTPUT_PATH = Path("ml/models/production/best_model.tflite")

# Number of sample spectrograms used for INT8 calibration.
# More samples → better calibration → closer to original accuracy.
N_CALIBRATION = 200


# ─── REPRESENTATIVE DATASET ──────────────────────────────────────

def _representative_dataset():
    """
    Generate synthetic spectrograms for INT8 calibration.

    Covers the full range of inputs the model will see in production:
    - Dark frames (noise floor, no signal)
    - Frames with narrow horizontal bands (narrowband signals)
    - Frames with wide bright areas (wideband signals)
    - Mixed noise patterns

    Input range is float32 [0, 255] — the model's internal Rescaling
    layer handles normalisation to [-1, 1].
    """
    rng = np.random.default_rng(42)

    for i in range(N_CALIBRATION):
        frame_type = i % 4

        if frame_type == 0:
            # Noise floor — mostly dark with slight texture
            spec = rng.integers(0, 30, (224, 224), dtype=np.uint8)

        elif frame_type == 1:
            # Narrowband signal — one or two bright horizontal bands
            spec = rng.integers(0, 20, (224, 224), dtype=np.uint8)
            n_bands = rng.integers(1, 4)
            for _ in range(n_bands):
                y = rng.integers(10, 214)
                h = rng.integers(2, 12)
                spec[y:y+h, :] = rng.integers(160, 255)

        elif frame_type == 2:
            # Wideband signal — large bright region (LTE/FM style)
            spec = rng.integers(0, 40, (224, 224), dtype=np.uint8)
            y1 = rng.integers(20, 80)
            y2 = rng.integers(140, 200)
            spec[y1:y2, :] = rng.integers(100, 200, (y2 - y1, 224), dtype=np.uint8)

        else:
            # Random medium-intensity noise (generic signal)
            spec = rng.integers(20, 180, (224, 224), dtype=np.uint8)

        yield [spec.reshape(1, 224, 224, 1).astype(np.float32)]


# ─── CONVERSION ──────────────────────────────────────────────────

def convert() -> None:
    if not KERAS_PATH.exists():
        raise FileNotFoundError(
            f"Model not found: {KERAS_PATH}\n"
            f"Make sure best_model.keras is in ml/models/production/"
        )

    print(f"Loading: {KERAS_PATH}")
    model = tf.keras.models.load_model(str(KERAS_PATH))
    print(f"  Input:  {model.input_shape}")
    print(f"  Output: {model.output_shape}")
    print(f"  Params: {model.count_params():,}")

    print(f"\nConverting to INT8 TFLite ({N_CALIBRATION} calibration samples)...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    # Full INT8 quantization (weights + activations)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = _representative_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]

    # Keep float32 I/O so classifier.py needs no preprocessing changes
    converter.inference_input_type  = tf.float32
    converter.inference_output_type = tf.float32

    tflite_model = converter.convert()

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_bytes(tflite_model)

    keras_mb  = KERAS_PATH.stat().st_size  / 1024 / 1024
    tflite_mb = OUTPUT_PATH.stat().st_size / 1024 / 1024
    reduction = (1 - tflite_mb / keras_mb) * 100

    print(f"\nDone.")
    print(f"  Keras  model : {keras_mb:.1f} MB  →  {KERAS_PATH}")
    print(f"  TFLite model : {tflite_mb:.1f} MB  →  {OUTPUT_PATH}")
    print(f"  Size reduction : {reduction:.0f}%")
    print(f"\nDeploy to Pi 5:")
    print(f"  scp {OUTPUT_PATH} porphyras@192.168.0.156:~/Desktop/SpectrumEye/{OUTPUT_PATH}")


if __name__ == "__main__":
    convert()
