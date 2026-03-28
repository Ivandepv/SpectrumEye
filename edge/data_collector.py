"""
edge/data_collector.py — RTL-SDR training dataset builder

Tunes to one of 10 frequency bands, captures real IQ samples, and saves
224x224 spectrograms to disk for CNN training.

Usage:
    python edge/data_collector.py                          # interactive
    python edge/data_collector.py --category walkie_talkie
    python edge/data_collector.py --category aircraft_tracking --n-images 200 --format grayscale --output-dir ml/dataset/raw
"""

import argparse
import logging
import random
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from rtlsdr import RtlSdr

log = logging.getLogger(__name__)

FREQUENCY_DB = {
    "radio_fm":             (88e6,    108e6),   # FM commercial (WBFM)
    "air_traffic":          (108e6,   137e6),   # Aerial control tower
    "noaa":                 (137e6,   138e6),   # Meteorological satellites
    "local_repeaters":      (144e6,   148e6),   # Amateur radio (2 m band)
    "maritime":             (156e6,   174e6),   # Maritime VHF
    "short_range_devices":  (315e6,   433.05e6),# Car keys, gates, doorbells
    "wireless_controllers": (433.05e6,434.79e6),# Specific short-range devices
    "walkie_talkie":        (446e6,   462e6),   # Walkie-talkies (UHF)
    "cellular_network":     (850e6,   960e6),   # GSM/LTE/5G, LoRaWAN
    "aircraft_tracking":    (1090e6,  1090e6),  # ADS-B commercial aircraft
}


# ---------------------------------------------------------------------------
# Signal capture
# ---------------------------------------------------------------------------

def capture_real_signal(sdr: RtlSdr, num_samples: int) -> np.ndarray:
    return sdr.read_samples(num_samples)


# ---------------------------------------------------------------------------
# Spectrogram writers
# ---------------------------------------------------------------------------

def fd_gray(signal: np.ndarray, fs: float, filepath: Path) -> None:
    plt.figure(figsize=(2.24, 2.24), dpi=100)
    plt.specgram(signal, NFFT=256, Fs=fs, cmap="gray")
    plt.axis("off")
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.margins(0, 0)
    plt.savefig(filepath, dpi=100)
    plt.close()


def fd_color(signal: np.ndarray, fs: float, filepath: Path) -> None:
    plt.figure(figsize=(2.24, 2.24), dpi=100)
    plt.specgram(signal, NFFT=256, Fs=fs, cmap="jet")
    plt.axis("off")
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.margins(0, 0)
    plt.savefig(filepath, dpi=100)
    plt.close()


# ---------------------------------------------------------------------------
# Band selection
# ---------------------------------------------------------------------------

def select_category_interactive() -> tuple[str, float]:
    """Prompt the user to choose a frequency band. Returns (category, freq_hz)."""
    print("\nFREQUENCIES AVAILABLE:")
    for key in FREQUENCY_DB:
        print(f"  - {key}")

    user_selection = input("\nCategory: ").strip().lower()
    if user_selection not in FREQUENCY_DB:
        log.error("Unknown category '%s'. Available: %s", user_selection, ", ".join(FREQUENCY_DB))
        sys.exit(1)

    return _resolve_freq(user_selection)


def select_category_cli(category: str) -> tuple[str, float]:
    """Validate and resolve a category passed from the CLI."""
    if category not in FREQUENCY_DB:
        log.error("Unknown category '%s'. Available: %s", category, ", ".join(FREQUENCY_DB))
        sys.exit(1)
    return _resolve_freq(category)


def _resolve_freq(category: str) -> tuple[str, float]:
    min_freq, max_freq = FREQUENCY_DB[category]
    target_freq = random.uniform(min_freq, max_freq)
    log.info("Tuning to %s at %.2f MHz", category, target_freq / 1e6)
    return category, target_freq


# ---------------------------------------------------------------------------
# Dataset creation
# ---------------------------------------------------------------------------

def create_dataset(
    sdr: RtlSdr,
    fs: float,
    n_images: int,
    num_samples: int,
    dataset_name: str,
    fmt: str,
    output_dir: Path,
) -> None:
    """Capture `n_images` spectrograms and save them under `output_dir/dataset_name/`."""
    sub_dir = output_dir / dataset_name
    sub_dir.mkdir(parents=True, exist_ok=True)

    existing = len(list(sub_dir.glob("*.png")))
    count = existing

    writer = fd_gray if fmt == "grayscale" else fd_color

    try:
        for _ in range(n_images):
            signal = capture_real_signal(sdr, num_samples)
            filename = f"sample_{count:03d}.png"
            filepath = sub_dir / filename
            writer(signal, fs, filepath)
            log.info("Saved %s", filename)
            count += 1
    except Exception:
        log.exception("Capture failed")
    finally:
        sdr.close()
        log.info("Sampling complete. %d images saved to %s", count - existing, sub_dir)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="RTL-SDR training dataset builder — captures 224x224 spectrograms to disk.",
    )
    parser.add_argument(
        "--category",
        choices=list(FREQUENCY_DB),
        default=None,
        help="Frequency band to tune to (skips interactive menu).",
    )
    parser.add_argument(
        "--n-images",
        type=int,
        default=500,
        metavar="N",
        help="Number of spectrogram images to capture (default: 500).",
    )
    parser.add_argument(
        "--format",
        dest="fmt",
        choices=["grayscale", "color"],
        default=None,
        help="Image colour format (skips interactive prompt).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("dataset_rf"),
        metavar="DIR",
        help="Root output directory (default: dataset_rf/).",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    args = parse_args()

    # --- Category selection ---
    if args.category:
        dataset_name, target_freq = select_category_cli(args.category)
    else:
        dataset_name, target_freq = select_category_interactive()

    # --- Format selection ---
    if args.fmt:
        fmt = args.fmt
    else:
        raw = input("Output format — color or grayscale? ").strip().lower()
        if raw not in ("color", "grayscale"):
            log.error("Invalid format '%s'. Choose 'color' or 'grayscale'.", raw)
            sys.exit(1)
        fmt = raw

    # --- Output directory ---
    if fmt == "grayscale":
        output_dir = args.output_dir / "dataset_gray"
    else:
        output_dir = args.output_dir / "dataset_color"

    # --- SDR init ---
    try:
        sdr = RtlSdr()
    except Exception:
        log.exception("Failed to open RTL-SDR device")
        sys.exit(1)

    sdr.sample_rate = 2.048e6
    sdr.center_freq = target_freq
    sdr.gain = "auto"
    fs = sdr.sample_rate

    num_samples = 256 * 1024

    log.info(
        "Starting capture: category=%s  freq=%.2f MHz  images=%d  format=%s  output=%s",
        dataset_name, target_freq / 1e6, args.n_images, fmt, output_dir,
    )

    create_dataset(sdr, fs, args.n_images, num_samples, dataset_name, fmt, output_dir)


if __name__ == "__main__":
    main()
