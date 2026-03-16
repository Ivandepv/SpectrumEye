"""
augment.py

Takes raw spectrograms and produces 6 augmented variants per image.
Augmentation multiplies the dataset by 7x (1 original + 6 augmented).

Augmentations applied:
  1. time_shift   — horizontal roll (signal appears at different moment)
  2. freq_shift   — vertical roll (signal at slightly different frequency)
  3. awgn         — add Gaussian noise (simulates lower SNR conditions)
  4. amplitude    — scale brightness (simulates distance / path loss variation)
  5. noise_mix    — blend with noise floor (simulates near-threshold signal)
  6. time_flip    — horizontal mirror (time reversal, pattern invariance)

Input:  ml/dataset/raw/{class}/*.png
Output: ml/dataset/augmented/{class}/*.png  (originals + 6 variants each)

Usage:
  python augment.py
"""

import random
from pathlib import Path

import numpy as np
from PIL import Image

# ─── PATHS ────────────────────────────────────────────────────────

RAW_DIR = Path(__file__).parent / "dataset" / "raw"
AUG_DIR = Path(__file__).parent / "dataset" / "augmented"

CLASS_LABELS = ["Key_Signal", "Walkie_Talkie", "LTE"]

# ─── IMAGE I/O ────────────────────────────────────────────────────

def _load(path: Path) -> np.ndarray:
    """Load PNG as float32 in [0, 1]."""
    return np.array(Image.open(path).convert("L"), dtype=np.float32) / 255.0


def _save(arr: np.ndarray, path: Path) -> None:
    """Save float32 [0, 1] array as uint8 PNG."""
    path.parent.mkdir(parents=True, exist_ok=True)
    img = (np.clip(arr, 0, 1) * 255).astype(np.uint8)
    Image.fromarray(img, mode="L").save(path)


# ─── AUGMENTATION FUNCTIONS ───────────────────────────────────────

def aug_time_shift(img: np.ndarray) -> np.ndarray:
    """
    Horizontal roll along the time axis.
    Signal appears shifted earlier or later in the capture window.
    Shift range: ±40 columns (~±10 ms).
    """
    shift = random.randint(-40, 40)
    return np.roll(img, shift, axis=1)


def aug_freq_shift(img: np.ndarray) -> np.ndarray:
    """
    Vertical roll along the frequency axis.
    Signal appears at a slightly different center frequency.
    Shift range: ±20 pixels (~±183 kHz).
    """
    shift = random.randint(-20, 20)
    return np.roll(img, shift, axis=0)


def aug_awgn(img: np.ndarray) -> np.ndarray:
    """
    Add Gaussian noise (AWGN — Additive White Gaussian Noise).
    Simulates captures at lower SNR. Noise std: 2–8% of full scale.
    """
    std = random.uniform(0.02, 0.08)
    noisy = img + np.random.normal(0, std, img.shape).astype(np.float32)
    return np.clip(noisy, 0, 1)


def aug_amplitude(img: np.ndarray) -> np.ndarray:
    """
    Scale pixel amplitude.
    Simulates varying signal strength (distance / path loss / gain setting).
    Scale range: 0.65–1.15x.
    """
    scale = random.uniform(0.65, 1.15)
    return np.clip(img * scale, 0, 1)


def aug_noise_mix(img: np.ndarray) -> np.ndarray:
    """
    Blend signal with a synthetic noise floor.
    Simulates a near-threshold, barely detectable signal.
    Signal weight: 65–85%, noise weight: 15–35%.
    """
    noise_lvl = random.uniform(0.03, 0.07)
    noise = np.random.normal(noise_lvl, noise_lvl * 0.4, img.shape).astype(np.float32)
    noise = np.clip(noise, 0, 1)
    alpha = random.uniform(0.65, 0.85)   # signal contribution
    return np.clip(alpha * img + (1 - alpha) * noise, 0, 1)


def aug_time_flip(img: np.ndarray) -> np.ndarray:
    """
    Horizontal mirror (time reversal).
    Makes the CNN invariant to whether a signal is arriving or departing.
    """
    return np.fliplr(img)


# ─── REGISTRY ─────────────────────────────────────────────────────

AUGMENTATIONS = [
    ("timeshift",  aug_time_shift),
    ("freqshift",  aug_freq_shift),
    ("awgn",       aug_awgn),
    ("amplitude",  aug_amplitude),
    ("noisemix",   aug_noise_mix),
    ("timeflip",   aug_time_flip),
]

# ─── MAIN ─────────────────────────────────────────────────────────

def augment_dataset() -> None:
    n_aug = len(AUGMENTATIONS)
    print(f"Augmenting dataset: {n_aug} variants per image (x{n_aug + 1} total)\n")

    grand_total_in  = 0
    grand_total_out = 0

    for class_name in CLASS_LABELS:
        raw_dir = RAW_DIR / class_name
        aug_dir = AUG_DIR / class_name
        aug_dir.mkdir(parents=True, exist_ok=True)

        pngs = sorted(raw_dir.glob("*.png"))
        if not pngs:
            print(f"  {class_name}: no source images in {raw_dir} — skipping")
            continue

        print(f"  {class_name}: {len(pngs)} source images -> ", end="", flush=True)

        for png in pngs:
            img = _load(png)

            # Copy original into augmented dir
            _save(img, aug_dir / png.name)

            # Apply each augmentation
            for aug_name, aug_fn in AUGMENTATIONS:
                aug_img  = aug_fn(img)
                out_name = f"{png.stem}__{aug_name}.png"
                _save(aug_img, aug_dir / out_name)

        out_count = len(list(aug_dir.glob("*.png")))
        grand_total_in  += len(pngs)
        grand_total_out += out_count
        print(f"{out_count} images total (x{n_aug + 1})")

    print(f"\nDone. {grand_total_in} raw -> {grand_total_out} augmented images.")
    print("Run split_dataset.py next.")


if __name__ == "__main__":
    augment_dataset()
