"""
collect_synthetic.py

Generates synthetic 224x224 grayscale spectrograms for CNN training.
Three signal classes (Phase 2 initial scope):

  Key_Signal    — key fob / remote control (OOK, very narrow, bursty)
  Walkie_Talkie — narrowband FM radio (NFM, narrow, continuous PTT)
  LTE           — 4G cellular downlink (OFDM, wide flat block, always on)

Spectrogram orientation (matches interface contract):
  axis 0 (rows) = frequency  — row 0 = lowest frequency bin
  axis 1 (cols) = time       — col 0 = oldest sample, col 223 = newest

Resolution after 1024-bin STFT resized to 224px:
  ~9143 Hz per frequency pixel
  ~0.25 ms per time column  (hop=512 at 2.048 Msps)

Usage:
  python collect_synthetic.py           # 200 samples/class
  python collect_synthetic.py --n 500   # 500 samples/class
"""

import json
import random
import argparse
from pathlib import Path
from datetime import datetime

import numpy as np
from PIL import Image

# ─── CONSTANTS ────────────────────────────────────────────────────

IMG_H = 224          # frequency axis (rows)
IMG_W = 224          # time axis (columns)
SAMPLE_RATE = 2.048e6
NFFT = 1024
HOP  = 512

# After resizing 1024 FFT bins to 224 pixels:
#   raw: 2.048e6 / 1024 = 2000 Hz/bin
#   per pixel: 2000 * (1024/224) ≈ 9143 Hz/px
HZ_PER_PX = (SAMPLE_RATE / NFFT) * (NFFT / IMG_H)   # ~9143 Hz/px

# Per time column: 512 samples / 2.048e6 sps ≈ 0.25 ms
MS_PER_COL = (HOP / SAMPLE_RATE) * 1000               # ~0.25 ms/col

CLASS_LABELS = ["Key_Signal", "Walkie_Talkie", "LTE"]

DATASET_DIR = Path(__file__).parent / "dataset" / "raw"

# ─── HELPERS ──────────────────────────────────────────────────────

def _noise_floor(level: float, shape: tuple = (IMG_H, IMG_W)) -> np.ndarray:
    """
    Gaussian noise floor. level ~0.03-0.10 (fraction of full scale).
    Returns float32 in [0, 1].
    """
    noise = np.random.normal(level, level * 0.4, shape).astype(np.float32)
    return np.clip(noise, 0, 1)


def _gaussian_column(center: float, width_px: float, amplitude: float) -> np.ndarray:
    """
    Gaussian-shaped signal in the frequency axis for one time column.
    Returns 1D float32 array of length IMG_H.
    """
    f = np.arange(IMG_H, dtype=np.float32)
    sigma = max(width_px / 2.5, 0.5)
    return (amplitude * np.exp(-0.5 * ((f - center) / sigma) ** 2)).astype(np.float32)


def _flat_column(lo: int, hi: int, amplitude: float, texture_std: float = 0.02) -> np.ndarray:
    """
    Flat-top beam in the frequency axis for one time column (OFDM).
    Returns 1D float32 array of length IMG_H.
    """
    col = np.zeros(IMG_H, dtype=np.float32)
    if lo < hi:
        col[lo:hi] = amplitude + np.random.normal(0, texture_std, hi - lo).astype(np.float32)
    return np.clip(col, 0, 1)


def _to_uint8(arr: np.ndarray) -> np.ndarray:
    return (np.clip(arr, 0, 1) * 255).astype(np.uint8)


def _save(img: np.ndarray, class_name: str, idx: int, meta: dict) -> None:
    out_dir = DATASET_DIR / class_name
    out_dir.mkdir(parents=True, exist_ok=True)
    ts   = int(datetime.now().timestamp() * 1000) + idx
    freq = int(meta.get("center_freq_hz", 0))
    snr  = int(meta.get("snr_estimate_db", 0))
    stem = f"{class_name}_{ts}_{freq}_{snr}"
    Image.fromarray(img, mode="L").save(out_dir / f"{stem}.png")
    with open(out_dir / f"{stem}.json", "w") as f:
        json.dump({"class": class_name, "timestamp_ms": ts,
                   "source": "synthetic", **meta}, f, indent=2)


# ─── GENERATORS ───────────────────────────────────────────────────

def _gen_key_signal() -> tuple:
    """
    Key fob / remote control (OOK - On-Off Keying).

    Visual signature in spectrogram:
    - Extremely narrow: 1-3 frequency pixels (~9-27 kHz)
    - Short repeated bursts: 2-5 pulses, 20-55 columns each (~5-14 ms)
    - Pure noise floor between bursts
    - High peak amplitude during burst
    Typical frequency: 433 MHz ISM band
    """
    noise_lvl = random.uniform(0.03, 0.08)
    img = _noise_floor(noise_lvl)

    center    = random.randint(30, IMG_H - 30)
    width_px  = random.randint(1, 3)
    snr_db    = random.uniform(15, 40)
    amplitude = min(0.92, 0.30 + snr_db / 60.0)

    n_bursts  = random.randint(2, 5)
    burst_len = random.randint(20, 55)   # columns per burst
    gap_len   = random.randint(15, 50)   # columns of silence between bursts

    t = random.randint(10, 40)
    for _ in range(n_bursts):
        if t >= IMG_W:
            break
        t_end = min(t + burst_len, IMG_W)
        beam  = _gaussian_column(center, width_px, amplitude)
        for col in range(t, t_end):
            noise = np.random.normal(0, 0.018, IMG_H).astype(np.float32)
            img[:, col] = np.clip(img[:, col] + beam + noise, 0, 1)
        t += burst_len + gap_len + random.randint(-5, 5)

    return _to_uint8(img), {
        "center_freq_hz":  int(433e6 + random.uniform(-2e6, 2e6)),
        "sample_rate_hz":  int(SAMPLE_RATE),
        "snr_estimate_db": round(snr_db, 1),
        "nfft": NFFT, "overlap": 0.5, "window": "hanning",
        "p_min_dbfs": -100, "p_max_dbfs": 0,
        "notes": "OOK key fob / remote control at 433 MHz ISM",
    }


def _gen_walkie_talkie() -> tuple:
    """
    Walkie Talkie (NFM - Narrowband FM).

    Visual signature in spectrogram:
    - Narrow band: 2-5 frequency pixels (~18-45 kHz)
    - Continuous carrier while PTT is pressed
    - Slight sinusoidal frequency deviation (FM voice modulation wobble)
    - 30% chance of one short PTT silence gap
    Typical frequency: 462 MHz FRS/GMRS band
    """
    noise_lvl = random.uniform(0.03, 0.07)
    img = _noise_floor(noise_lvl)

    center    = random.randint(40, IMG_H - 40)
    width_px  = random.randint(2, 5)
    snr_db    = random.uniform(12, 35)
    amplitude = min(0.90, 0.35 + snr_db / 70.0)

    # FM voice modulation: sinusoidal frequency deviation
    dev_max  = random.uniform(0.5, 2.0)    # pixels of peak deviation
    dev_rate = random.uniform(0.03, 0.12)  # rad/column

    tx_start = random.randint(0, 30)
    tx_end   = random.randint(140, IMG_W)

    # 30% chance of a PTT release gap mid-capture
    has_gap   = random.random() < 0.30
    gap_start = random.randint(tx_start + 30, max(tx_start + 31, tx_end - 30)) if has_gap else -1
    gap_end   = gap_start + random.randint(10, 30) if has_gap else -1

    for col in range(tx_start, tx_end):
        if has_gap and gap_start <= col < gap_end:
            continue
        dev         = dev_max * np.sin(dev_rate * col)
        eff_center  = center + dev
        beam        = _gaussian_column(eff_center, width_px, amplitude)
        noise       = np.random.normal(0, 0.018, IMG_H).astype(np.float32)
        img[:, col] = np.clip(img[:, col] + beam + noise, 0, 1)

    return _to_uint8(img), {
        "center_freq_hz":  int(462e6 + random.uniform(-10e6, 10e6)),
        "sample_rate_hz":  int(SAMPLE_RATE),
        "snr_estimate_db": round(snr_db, 1),
        "nfft": NFFT, "overlap": 0.5, "window": "hanning",
        "p_min_dbfs": -100, "p_max_dbfs": 0,
        "notes": "Narrowband FM walkie talkie (NFM 12.5/25 kHz channel)",
    }


def _gen_lte() -> tuple:
    """
    LTE Mobile (OFDM - 4G cellular downlink).

    Visual signature in spectrogram:
    - Wide flat block: 120-200 frequency pixels
      (LTE channel 10-20 MHz >> RTL-SDR 2 MHz window, fills most of axis)
    - Almost always present and continuous (base station always on)
    - Flat top with slight OFDM subcarrier ripple texture
    - Attenuated guard bands at frequency edges (~6% of bandwidth)
    Typical frequency: 1800 MHz Taiwan LTE Band 3
    """
    noise_lvl = random.uniform(0.02, 0.06)
    img = _noise_floor(noise_lvl)

    # LTE channel is wider than our 2 MHz capture window
    bw_px  = random.randint(120, 200)
    center = random.randint(bw_px // 2 + 10, IMG_H - bw_px // 2 - 10)
    lo     = max(0, center - bw_px // 2)
    hi     = min(IMG_H, lo + bw_px)

    snr_db    = random.uniform(10, 28)
    amplitude = min(0.88, 0.40 + snr_db / 80.0)

    tx_start = random.randint(0, 10)
    tx_end   = random.randint(200, IMG_W)
    guard_px = max(3, bw_px // 15)

    # OFDM resource block ripple pattern (repeats every ~12 px)
    rb_ripple = (0.015 * np.sin(np.arange(IMG_H) * np.pi / 12)).astype(np.float32)

    for col in range(tx_start, tx_end):
        col_sig = _flat_column(lo, hi, amplitude, texture_std=0.022)

        # Guard band attenuation at edges
        col_sig[lo : lo + guard_px] *= 0.55
        col_sig[hi - guard_px : hi] *= 0.55

        # Apply resource block ripple
        col_sig = np.clip(col_sig + rb_ripple, 0, 1)

        noise       = np.random.normal(0, 0.014, IMG_H).astype(np.float32)
        img[:, col] = np.clip(img[:, col] + col_sig + noise, 0, 1)

    return _to_uint8(img), {
        "center_freq_hz":  int(1800e6 + random.uniform(-10e6, 10e6)),
        "sample_rate_hz":  int(SAMPLE_RATE),
        "snr_estimate_db": round(snr_db, 1),
        "nfft": NFFT, "overlap": 0.5, "window": "hanning",
        "p_min_dbfs": -100, "p_max_dbfs": 0,
        "notes": "LTE OFDM downlink — 2 MHz RTL-SDR slice of 10/20 MHz channel",
    }


# ─── REGISTRY & ENTRY POINT ───────────────────────────────────────

GENERATORS = {
    "Key_Signal":    _gen_key_signal,
    "Walkie_Talkie": _gen_walkie_talkie,
    "LTE":           _gen_lte,
}


def generate_dataset(n_per_class: int = 200, seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)

    total = n_per_class * len(GENERATORS)
    print(f"Generating {n_per_class} samples x {len(GENERATORS)} classes = {total} images\n")

    for class_name, generator in GENERATORS.items():
        print(f"  {class_name} ...", end=" ", flush=True)
        for i in range(n_per_class):
            arr, meta = generator()
            _save(arr, class_name, i, meta)
        count = len(list((DATASET_DIR / class_name).glob("*.png")))
        print(f"{count} images saved -> ml/dataset/raw/{class_name}/")

    print("\nDone. Run augment.py next.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic spectrogram dataset")
    parser.add_argument("--n",    type=int, default=200, help="Samples per class (default: 200)")
    parser.add_argument("--seed", type=int, default=42,  help="Random seed (default: 42)")
    args = parser.parse_args()
    generate_dataset(n_per_class=args.n, seed=args.seed)
