"""
edge/classifier.py — CNN Inference Wrapper

Loads the trained SpectrumEye MobileNetV2 model and runs inference on
224×224 uint8 spectrogram arrays.

Returns Interface B classification dicts as defined in the Interface Contract.

Confidence thresholds:
  ≥ 0.85  → HIGH   — classify as predicted class
  0.60 – 0.84 → MEDIUM — classify but confidence_level flagged
  < 0.60  → LOW    — override predicted_class to "Unknown"

Usage:
    from edge.classifier import SpectrumClassifier
    clf = SpectrumClassifier()
    result = clf.classify(spectrogram, frame_id=1, timestamp_ms=..., center_freq_hz=...)
    print(result["predicted_class"], result["confidence"])

Run standalone for a self-test (loads model, runs dummy inference):
    python classifier.py --test
"""

import time
import argparse
from pathlib import Path
from typing import Optional

import numpy as np

# ─── CONFIGURATION ────────────────────────────────────────────────

# 3-class scope for Phase 2 (matches training)
CLASS_LABELS = ["Key_Signal", "Walkie_Talkie", "LTE"]

# Path to production model (relative to this file)
_DEFAULT_MODEL = Path(__file__).parent.parent / "ml" / "models" / "production" / "spectromeye_best.keras"

# Confidence thresholds (Interface Contract §Interface B)
CONF_HIGH   = 0.85   # ≥ 0.85 → HIGH, full confidence
CONF_MEDIUM = 0.60   # 0.60 – 0.84 → MEDIUM, flagged
# < 0.60 → LOW, override to "Unknown"


# ─── CLASSIFIER ───────────────────────────────────────────────────

class SpectrumClassifier:
    """
    CNN inference wrapper for SpectrumEye MobileNetV2 model.

    Thread-safe for read (classify) calls. Not thread-safe for load/reload.
    """

    def __init__(self, model_path: Path = _DEFAULT_MODEL) -> None:
        """
        Load the Keras model from disk and warm it up.

        Args:
            model_path: path to best_model.keras or spectromeye_best.keras
        """
        self._model_path = Path(model_path)
        self._model      = None
        self._version    = self._model_path.stem   # e.g. "spectromeye_best"
        self._loaded     = False

        self._load()

    # ── Model loading ─────────────────────────────────────────────

    def _load(self) -> None:
        """Import TensorFlow lazily (heavy import) and load the model."""
        if not self._model_path.exists():
            raise FileNotFoundError(
                f"Model not found: {self._model_path}\n"
                f"Train the model first with:  python ml/train.py\n"
                f"Or run quick test with:       python ml/train.py --quick"
            )

        # Suppress TF startup noise
        import os
        os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

        from tensorflow import keras
        print(f"[Classifier] Loading model: {self._model_path}")
        self._model = keras.models.load_model(str(self._model_path))
        print(f"[Classifier] Model loaded — {self._model.count_params():,} params")

        # Warm-up inference (first call triggers JIT compilation)
        dummy = np.zeros((1, 224, 224, 1), dtype=np.float32)
        self._model.predict(dummy, verbose=0)
        print("[Classifier] Warm-up complete — ready")
        self._loaded = True

    # ── Inference ─────────────────────────────────────────────────

    def classify(
        self,
        spectrogram:    np.ndarray,
        frame_id:       int,
        timestamp_ms:   int,
        center_freq_hz: int,
    ) -> dict:
        """
        Run CNN inference on a 224×224 uint8 spectrogram.

        Args:
            spectrogram:    numpy array, shape (224, 224), dtype uint8
            frame_id:       monotonic frame counter (from sweep_frame)
            timestamp_ms:   Unix timestamp in milliseconds (from sweep_frame)
            center_freq_hz: center frequency Hz (from sweep_frame)

        Returns:
            Interface B classification dict:
            {
                "frame_id":           int,
                "timestamp_ms":       int,
                "center_freq_hz":     int,
                "predicted_class":    str,   # one of CLASS_LABELS or "Unknown"
                "confidence":         float, # 0.0 – 1.0
                "confidence_level":   str,   # "HIGH" | "MEDIUM" | "LOW"
                "all_probabilities":  dict,  # {class_label: prob}
                "inference_time_ms":  float,
                "model_version":      str,
            }
        """
        if not self._loaded:
            raise RuntimeError("Classifier not loaded. Call _load() first.")

        # Validate input
        if spectrogram.shape != (224, 224):
            raise ValueError(f"Expected (224, 224) spectrogram, got {spectrogram.shape}")

        # Prepare input tensor: (1, 224, 224, 1) float32
        # The model's Rescaling layer handles [0,255] → [-1,1] internally
        x = spectrogram.astype(np.float32)[np.newaxis, :, :, np.newaxis]

        # Run inference
        t0    = time.perf_counter()
        probs = self._model.predict(x, verbose=0)[0]   # shape (3,)
        inference_ms = (time.perf_counter() - t0) * 1000.0

        # Decode predictions
        top_idx    = int(np.argmax(probs))
        confidence = float(probs[top_idx])

        # Apply confidence threshold (Interface Contract §Confidence Thresholds)
        if confidence >= CONF_HIGH:
            conf_level      = "HIGH"
            predicted_class = CLASS_LABELS[top_idx]
        elif confidence >= CONF_MEDIUM:
            conf_level      = "MEDIUM"
            predicted_class = CLASS_LABELS[top_idx]
        else:
            conf_level      = "LOW"
            predicted_class = "Unknown"   # override — too uncertain to classify

        all_probs = {label: round(float(probs[i]), 6) for i, label in enumerate(CLASS_LABELS)}

        return {
            "frame_id":          frame_id,
            "timestamp_ms":      timestamp_ms,
            "center_freq_hz":    center_freq_hz,
            "predicted_class":   predicted_class,
            "confidence":        round(confidence, 4),
            "confidence_level":  conf_level,
            "all_probabilities": all_probs,
            "inference_time_ms": round(inference_ms, 2),
            "model_version":     self._version,
        }

    # ── Properties ────────────────────────────────────────────────

    @property
    def is_ready(self) -> bool:
        return self._loaded

    @property
    def model_version(self) -> str:
        return self._version


# ─── SELF-TEST ────────────────────────────────────────────────────

def _run_test(model_path: Optional[Path] = None) -> None:
    """Load the model and run inference on a synthetic 224×224 frame."""
    path = Path(model_path) if model_path else _DEFAULT_MODEL

    print("=" * 50)
    print("SpectrumClassifier self-test")
    print("=" * 50)

    clf = SpectrumClassifier(model_path=path)

    # Test 1: All-zeros (noise floor)
    print("\nTest 1 — noise floor (all-zeros spectrogram):")
    zeros = np.zeros((224, 224), dtype=np.uint8)
    r = clf.classify(zeros, frame_id=0, timestamp_ms=1000, center_freq_hz=433000000)
    print(f"  predicted: {r['predicted_class']}  confidence: {r['confidence']:.4f}  "
          f"({r['confidence_level']})  {r['inference_time_ms']:.1f} ms")

    # Test 2: Random noise
    print("\nTest 2 — random noise spectrogram:")
    noise = np.random.randint(0, 30, (224, 224), dtype=np.uint8)
    r = clf.classify(noise, frame_id=1, timestamp_ms=2000, center_freq_hz=462000000)
    print(f"  predicted: {r['predicted_class']}  confidence: {r['confidence']:.4f}  "
          f"({r['confidence_level']})  {r['inference_time_ms']:.1f} ms")

    # Test 3: Bright signal (full-scale)
    print("\nTest 3 — bright signal (mostly 255):")
    bright = np.full((224, 224), 10, dtype=np.uint8)
    bright[100:124, :] = 220   # narrow horizontal band
    r = clf.classify(bright, frame_id=2, timestamp_ms=3000, center_freq_hz=1800000000)
    print(f"  predicted: {r['predicted_class']}  confidence: {r['confidence']:.4f}  "
          f"({r['confidence_level']})  {r['inference_time_ms']:.1f} ms")

    # Test 4: Wrong input shape
    print("\nTest 4 — wrong shape raises ValueError:")
    try:
        clf.classify(np.zeros((100, 100), dtype=np.uint8), frame_id=3, timestamp_ms=4000, center_freq_hz=0)
        print("  FAIL — expected ValueError")
    except ValueError as e:
        print(f"  PASS — caught ValueError: {e}")

    print("\n" + "=" * 50)
    print("Self-test complete.")


# ─── ENTRY POINT ──────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SpectrumEye CNN Classifier")
    parser.add_argument("--test",  action="store_true", help="Run self-test with dummy spectrograms")
    parser.add_argument("--model", type=str, default=None, help="Path to .keras model file (default: production)")
    args = parser.parse_args()

    if args.test:
        _run_test(model_path=args.model)
    else:
        parser.print_help()
