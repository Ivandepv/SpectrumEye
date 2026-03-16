"""
ml/evaluate.py — Model Evaluation on New Data

Tests the trained model on data it has never seen.

Three usage modes:

  1. QUICK — generate fresh simulation images and run inference
     python ml/evaluate.py --quick

  2. FOLDER — test on a labeled folder you provide
     The folder must be organized as:
       my_test_data/
       ├── Key_Signal/    *.png
       ├── Walkie_Talkie/ *.png
       └── LTE/           *.png
     python ml/evaluate.py --folder path/to/my_test_data

  3. SINGLE IMAGE — classify one PNG file
     python ml/evaluate.py --image path/to/spectrogram.png

All modes load the production model by default:
  ml/models/production/spectromeye_best.keras
Use --model to point to a different .keras file.
"""

import os
import time
import argparse
from pathlib import Path

import numpy as np
from PIL import Image

# ─── CONFIG ───────────────────────────────────────────────────────

CLASS_LABELS   = ["Key_Signal", "Walkie_Talkie", "LTE"]
IMG_SIZE       = (224, 224)
CONF_THRESHOLD = 0.60   # below this → reported as "Unknown"

_DEFAULT_MODEL = (
    Path(__file__).parent / "models" / "production" / "spectromeye_best.keras"
)


# ─── MODEL LOADER ─────────────────────────────────────────────────

def load_model(model_path: Path):
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    from tensorflow import keras
    print(f"Loading model: {model_path}")
    model = keras.models.load_model(str(model_path))
    model.predict(np.zeros((1, 224, 224, 1), dtype=np.float32), verbose=0)
    print("Model ready.\n")
    return model


# ─── INFERENCE HELPERS ────────────────────────────────────────────

def load_image(path: Path) -> np.ndarray:
    """Load any PNG as 224×224 uint8 grayscale array."""
    img = Image.open(path).convert("L").resize(IMG_SIZE, Image.LANCZOS)
    return np.array(img, dtype=np.uint8)


def predict_one(model, img_array: np.ndarray) -> dict:
    """Run inference on a single 224×224 uint8 array."""
    x     = img_array.astype(np.float32)[np.newaxis, :, :, np.newaxis]
    t0    = time.perf_counter()
    probs = model.predict(x, verbose=0)[0]
    ms    = (time.perf_counter() - t0) * 1000

    top_idx    = int(np.argmax(probs))
    confidence = float(probs[top_idx])
    predicted  = CLASS_LABELS[top_idx] if confidence >= CONF_THRESHOLD else "Unknown"

    return {
        "predicted":  predicted,
        "confidence": round(confidence, 4),
        "probs":      {c: round(float(p), 4) for c, p in zip(CLASS_LABELS, probs)},
        "latency_ms": round(ms, 1),
    }


# ─── MODE 1: QUICK (generate + infer) ────────────────────────────

def run_quick(model) -> None:
    """
    Generate 10 fresh images per class with a new random seed
    (seed=999, never used during training) and run inference.
    No files are saved to disk.
    """
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent / "simulation"))

    try:
        from simulation_final import (
            generate_t_sampling, key_signal_simulation,
            walkie_talkie_simulation, lte_simulation,
            awgn, spectrogram_to_image,
        )
    except ImportError:
        print("ERROR: simulation/simulation_final.py not found.")
        print("Run from the project root: python ml/evaluate.py --quick")
        return

    N   = 10
    FS  = 1000
    t   = generate_t_sampling(FS, duration=2)
    rng = np.random.default_rng(seed=999)   # different seed → truly unseen data

    sim_fns = {
        "Key_Signal":    key_signal_simulation,
        "Walkie_Talkie": walkie_talkie_simulation,
        "LTE":           lte_simulation,
    }

    print(f"Quick test — {N} fresh images per class (seed=999, not used in training)\n")
    print(f"  {'True Class':<16}  {'Predicted':<16}  {'Conf':>6}  {'OK':>3}  {'ms':>5}")
    print(f"  {'─'*16}  {'─'*16}  {'─'*6}  {'─'*3}  {'─'*5}")

    correct = 0
    total   = 0

    for true_cls, sim_fn in sim_fns.items():
        for _ in range(N):
            np.random.seed(int(rng.integers(0, 99999)))
            pure  = sim_fn(t, FS)
            noisy = awgn(pure, noise_level=0.5)
            img   = spectrogram_to_image(noisy, FS)
            arr   = np.array(img, dtype=np.uint8)

            r  = predict_one(model, arr)
            ok = "✓" if r["predicted"] == true_cls else "✗"
            if r["predicted"] == true_cls:
                correct += 1
            total += 1

            print(f"  {true_cls:<16}  {r['predicted']:<16}  {r['confidence']:>6.4f}  {ok:>3}  {r['latency_ms']:>4.0f}ms")

    acc = correct / total * 100
    print(f"\n  Accuracy: {correct}/{total} = {acc:.1f}%")

    if acc >= 90:
        print("  ✓ GOOD — model generalises well to unseen simulation data.")
    elif acc >= 70:
        print("  ~ OK — some errors, review the pattern above.")
    else:
        print("  ✗ POOR — model may have overfit. Consider more training data or epochs.")


# ─── MODE 2: FOLDER ───────────────────────────────────────────────

def run_folder(model, folder: Path) -> None:
    """
    Full evaluation on a labeled folder.
    Expects: folder/ClassName/*.png
    Prints classification report + confusion matrix.
    """
    from sklearn.metrics import classification_report, confusion_matrix

    y_true, y_pred = [], []
    latencies      = []

    for true_cls in CLASS_LABELS:
        cls_dir = folder / true_cls
        if not cls_dir.exists():
            print(f"  WARNING: {cls_dir} not found — skipping {true_cls}")
            continue

        pngs = sorted(cls_dir.glob("*.png"))
        if not pngs:
            print(f"  WARNING: no PNGs in {cls_dir}")
            continue

        print(f"  {true_cls}: {len(pngs)} images...", end="", flush=True)

        for png in pngs:
            arr = load_image(png)
            r   = predict_one(model, arr)

            y_true.append(CLASS_LABELS.index(true_cls))
            pred_cls = r["predicted"] if r["predicted"] in CLASS_LABELS else CLASS_LABELS[int(np.argmax([r["probs"][c] for c in CLASS_LABELS]))]
            y_pred.append(CLASS_LABELS.index(pred_cls))
            latencies.append(r["latency_ms"])

        n = len(pngs)
        cls_correct = sum(t == p for t, p in zip(y_true[-n:], y_pred[-n:]))
        print(f" {cls_correct}/{n} correct")

    if not y_true:
        print("No images found. Check folder structure.")
        return

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    present = sorted(set(y_true))

    print("\n" + "=" * 55)
    print("CLASSIFICATION REPORT")
    print("=" * 55)
    print(classification_report(
        y_true, y_pred,
        labels=present,
        target_names=[CLASS_LABELS[i] for i in present],
        digits=4,
    ))

    print("CONFUSION MATRIX  (rows = true, cols = predicted)")
    cm = confusion_matrix(y_true, y_pred, labels=present)
    header = "".join(f"{CLASS_LABELS[i]:>16}" for i in present)
    print(f"{'':>16}{header}")
    for i, row in zip(present, cm):
        print(f"{CLASS_LABELS[i]:>16}{''.join(f'{v:>16}' for v in row)}")

    acc = float(np.mean(y_true == y_pred))
    print(f"\nOverall accuracy : {acc*100:.2f}%  ({int(acc*len(y_true))}/{len(y_true)})")
    print(f"Avg inference    : {float(np.mean(latencies)):.1f} ms/image")


# ─── MODE 3: SINGLE IMAGE ─────────────────────────────────────────

def run_single(model, image_path: Path) -> None:
    """Classify one PNG and show detailed probabilities."""
    print(f"Image: {image_path}\n")

    arr = load_image(image_path)
    r   = predict_one(model, arr)

    print(f"  Predicted : {r['predicted']}")
    print(f"  Confidence: {r['confidence']*100:.2f}%")
    print(f"  Latency   : {r['latency_ms']:.1f} ms")
    print(f"\n  All probabilities:")
    for cls, prob in sorted(r["probs"].items(), key=lambda x: -x[1]):
        bar = "█" * int(prob * 40)
        print(f"    {cls:<16} {prob*100:5.1f}%  {bar}")

    if r["confidence"] < CONF_THRESHOLD:
        print(f"\n  NOTE: confidence below {CONF_THRESHOLD*100:.0f}% — output is 'Unknown'")


# ─── ENTRY POINT ──────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="SpectrumEye — evaluate model on new data",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--quick", action="store_true",
        help="Generate 10 fresh images per class and classify them\n(no files needed — uses the simulation directly)",
    )
    mode.add_argument(
        "--folder", metavar="PATH",
        help="Evaluate on a labeled folder:\n  PATH/Key_Signal/*.png\n  PATH/Walkie_Talkie/*.png\n  PATH/LTE/*.png",
    )
    mode.add_argument(
        "--image", metavar="PATH",
        help="Classify a single PNG file",
    )
    parser.add_argument(
        "--model", metavar="PATH", default=None,
        help=f"Path to .keras model file\n(default: ml/models/production/spectromeye_best.keras)",
    )

    args = parser.parse_args()

    model_path = Path(args.model) if args.model else _DEFAULT_MODEL
    if not model_path.exists():
        print(f"ERROR: Model not found: {model_path}")
        print("Train first:  python ml/train.py")
        print("Or copy the downloaded best_model.keras to ml/models/production/spectromeye_best.keras")
        return

    model = load_model(model_path)

    if args.quick:
        run_quick(model)
    elif args.folder:
        run_folder(model, Path(args.folder))
    elif args.image:
        run_single(model, Path(args.image))


if __name__ == "__main__":
    main()
