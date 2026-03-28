"""
ml/prepare_colab_zip.py

Creates ml_training_v3.zip ready to upload to Google Colab.

Contents:
  ml/
  ├── dataset/raw/aircraft_tracking/    (500 images)
  ├── dataset/raw/air_traffic/          (500 images)
  ├── dataset/raw/cellular_network/     (500 images)
  ├── dataset/raw/local_repeaters/      (500 images)
  ├── dataset/raw/maritime/             (500 images)
  ├── dataset/raw/noaa/                 (500 images)
  ├── dataset/raw/radio_fm/             (500 images)
  ├── dataset/raw/short_range_devices/  (500 images)
  ├── dataset/raw/walkie_talkie/        (500 images)
  ├── dataset/raw/wireless_controllers/ (500 images)
  ├── augment.py
  ├── split_dataset.py
  └── train.py

On Colab the notebook will run:
  augment.py       →  5 000 raw  →  35 000 augmented
  split_dataset.py →  17 500     →  train / val / test
  train.py         →  CNN training (50 epochs, T4 GPU)

Usage:
  python ml/prepare_colab_zip.py
  # → creates ml_training_v3.zip in the project root
"""

import os
import shutil
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
ML_DIR       = PROJECT_ROOT / "ml"
SIM_DATA_DIR = PROJECT_ROOT / "simulation" / "dataset_rf"
OUTPUT_ZIP   = PROJECT_ROOT / "ml_training_v3.zip"

CLASS_LABELS = [
    "aircraft_tracking",
    "air_traffic",
    "cellular_network",
    "local_repeaters",
    "maritime",
    "noaa",
    "radio_fm",
    "short_range_devices",
    "walkie_talkie",
    "wireless_controllers",
]


def build_zip() -> None:
    print("Building ml_training_v3.zip...\n")

    with zipfile.ZipFile(OUTPUT_ZIP, "w", compression=zipfile.ZIP_DEFLATED) as zf:

        # ── Raw dataset (from partner simulation) ─────────────────
        total_images = 0
        for cls in CLASS_LABELS:
            src_dir = SIM_DATA_DIR / cls
            if not src_dir.exists():
                print(f"  WARNING: {src_dir} not found — skipping {cls}")
                continue

            pngs = sorted(src_dir.glob("*.png"))
            for png in pngs:
                arcname = f"ml/dataset/raw/{cls}/{png.name}"
                zf.write(png, arcname)
            total_images += len(pngs)
            print(f"  {cls}: {len(pngs)} images")

        print(f"  Total raw images: {total_images}\n")

        # ── Python scripts ────────────────────────────────────────
        for script in ["augment.py", "split_dataset.py", "train.py"]:
            src = ML_DIR / script
            if not src.exists():
                print(f"  WARNING: {src} not found — skipping")
                continue
            zf.write(src, f"ml/{script}")
            print(f"  Added: ml/{script}")

    size_mb = OUTPUT_ZIP.stat().st_size / 1024 / 1024
    print(f"\nCreated: {OUTPUT_ZIP.name}  ({size_mb:.1f} MB)")
    print("Upload this file to Google Colab (Cell 3 of SpectrumEye_Training.ipynb).")


if __name__ == "__main__":
    build_zip()
