"""
edge/rtlsdr_source.py — Real-time RTL-SDR frame source

Uses the system `rtl_sdr` binary instead of pyrtlsdr, so it works on
any Python version (3.11, 3.12, 3.13) without C-library symbol issues.

Requires the system package:
    sudo apt install rtl-sdr          # Debian / Raspberry Pi OS
    sudo pacman -S rtl-sdr            # Arch Linux (dev machine)

Each band capture calls `rtl_sdr` as a subprocess, reads raw uint8 IQ
from stdout, and converts to complex float — identical signal pipeline
to the original pyrtlsdr implementation.
"""

import io
import logging
import shutil
import subprocess
import time
from typing import Iterator, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
from PIL import Image

log = logging.getLogger(__name__)

# ─── HARDWARE CONSTANTS ───────────────────────────────────────────

_SAMPLE_RATE_HZ: int = 2_048_000          # stable for RTL-SDR Blog V4
_NUM_SAMPLES:    int = 256 * 1024         # ~128 ms at 2.048 MSPS
_CAPTURE_TIMEOUT = 15                     # subprocess timeout (seconds)

# ─── FREQUENCY BANDS ─────────────────────────────────────────────

_ALL_BANDS: list[dict] = [
    {"band_id": "radio_fm",             "center_freq_hz":    98_000_000},
    {"band_id": "air_traffic",          "center_freq_hz":   122_000_000},
    {"band_id": "noaa",                 "center_freq_hz":   137_500_000},
    {"band_id": "local_repeaters",      "center_freq_hz":   146_000_000},
    {"band_id": "maritime",             "center_freq_hz":   156_800_000},
    {"band_id": "short_range_devices",  "center_freq_hz":   315_000_000},
    {"band_id": "wireless_controllers", "center_freq_hz":   433_920_000},
    {"band_id": "walkie_talkie",        "center_freq_hz":   446_000_000},
    {"band_id": "cellular_network",     "center_freq_hz":   900_000_000},
    {"band_id": "aircraft_tracking",    "center_freq_hz": 1_090_000_000},
]

_MIN_FREQ_HZ = 500_000
_MAX_FREQ_HZ = 1_750_000_000
for _b in _ALL_BANDS:
    assert _MIN_FREQ_HZ <= _b["center_freq_hz"] <= _MAX_FREQ_HZ, (
        f"{_b['band_id']}: {_b['center_freq_hz']} out of RTL-SDR Blog V4 range"
    )


# ─── IQ CAPTURE ──────────────────────────────────────────────────

def _capture_iq(freq_hz: int, gain: float | str, device_index: int = 0) -> np.ndarray:
    """
    Capture IQ samples by calling the `rtl_sdr` system binary.

    rtl_sdr writes raw uint8 interleaved I/Q to stdout.
    We convert to complex float: value = (uint8 - 127.5) / 127.5

    Args:
        freq_hz:      center frequency in Hz
        gain:         tuner gain in dB, or "auto"
        device_index: RTL-SDR device index (0 for single dongle)

    Returns:
        complex64 numpy array of length _NUM_SAMPLES
    """
    cmd = [
        "rtl_sdr",
        "-d", str(device_index),
        "-f", str(freq_hz),
        "-s", str(_SAMPLE_RATE_HZ),
        "-n", str(_NUM_SAMPLES),
    ]
    if gain != "auto":
        cmd += ["-g", str(gain)]
    cmd.append("-")   # write to stdout

    result = subprocess.run(
        cmd,
        capture_output=True,
        check=True,
        timeout=_CAPTURE_TIMEOUT,
    )

    raw = np.frombuffer(result.stdout, dtype=np.uint8).astype(np.float32)
    i_samples = (raw[0::2] - 127.5) / 127.5
    q_samples = (raw[1::2] - 127.5) / 127.5
    return (i_samples + 1j * q_samples).astype(np.complex64)


# ─── SIGNAL PROCESSING ───────────────────────────────────────────

def _iq_to_spectrogram(samples: np.ndarray) -> np.ndarray:
    """
    Convert complex IQ to 224×224 uint8 grayscale spectrogram.
    Identical parameters to data_collector.py for training data match.
    """
    fig = plt.figure(figsize=(2.24, 2.24), dpi=100)
    plt.specgram(samples, NFFT=256, Fs=_SAMPLE_RATE_HZ, cmap="gray")
    plt.axis("off")
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.margins(0, 0)

    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=100)
    plt.close(fig)
    buf.seek(0)

    img = Image.open(buf).convert("L")
    img = img.resize((224, 224), Image.LANCZOS)
    return np.array(img, dtype=np.uint8)


def _compute_rssi(samples: np.ndarray) -> float:
    """Compute received signal strength in dBFS from IQ power."""
    power = float(np.mean(np.abs(samples) ** 2))
    return float(10.0 * np.log10(power + 1e-12))


# ─── FRAME SOURCE ─────────────────────────────────────────────────

class RTLSDRFrameSource:
    """
    Real-time sweep frame source using RTL-SDR Blog V4.

    Sweeps through all configured bands in round-robin order.
    Yields sweep_frame dicts compatible with EdgePipeline (Interface A/A+).

    Each band opens the device via `rtl_sdr` subprocess — no persistent
    device handle, no pyrtlsdr dependency.

    Args:
        bands:        list of band_id strings to scan (default: all 10)
        gain:         tuner gain in dB, or "auto" (recommended)
        device_index: RTL-SDR device index (0 for single dongle)
    """

    def __init__(
        self,
        bands: Optional[list[str]] = None,
        gain: float | str = "auto",
        device_index: int = 0,
    ) -> None:
        if bands is None:
            self._bands = list(_ALL_BANDS)
        else:
            band_map = {b["band_id"]: b for b in _ALL_BANDS}
            missing = [b for b in bands if b not in band_map]
            if missing:
                raise ValueError(f"Unknown band_id(s): {missing}. Available: {list(band_map)}")
            self._bands = [band_map[b] for b in bands]

        self._gain         = gain
        self._device_index = device_index
        self._frame_id     = 0

    def frames(self) -> Iterator[dict]:
        """Yield sweep_frames indefinitely, sweeping all bands round-robin."""
        log.info(
            "RTLSDRFrameSource: starting sweep — %d bands, gain=%s",
            len(self._bands), self._gain,
        )
        band_idx = 0
        while True:
            band     = self._bands[band_idx]
            band_idx = (band_idx + 1) % len(self._bands)
            frame    = self._capture_band(band)
            if frame is not None:
                yield frame

    def _capture_band(self, band: dict) -> Optional[dict]:
        """Capture one band, return sweep_frame or None on error."""
        freq_hz = band["center_freq_hz"]
        band_id = band["band_id"]

        try:
            samples   = _capture_iq(freq_hz, self._gain, self._device_index)
            rssi_dbfs = _compute_rssi(samples)
            spectrogram = _iq_to_spectrogram(samples)
        except subprocess.CalledProcessError as exc:
            log.warning(
                "rtl_sdr failed for %s @ %.1f MHz (exit %d): %s",
                band_id, freq_hz / 1e6, exc.returncode,
                exc.stderr.decode(errors="replace").strip(),
            )
            return None
        except Exception as exc:
            log.warning("capture failed for %s @ %.1f MHz: %s", band_id, freq_hz / 1e6, exc)
            return None

        self._frame_id += 1
        return {
            "frame_id":       self._frame_id,
            "timestamp_ms":   int(time.time() * 1000),
            "center_freq_hz": freq_hz,
            "sample_rate_hz": _SAMPLE_RATE_HZ,
            "gain_db":        self._gain,
            "spectrogram":    spectrogram,
            "rssi": {
                "band_id":        band_id,
                "center_freq_hz": freq_hz,
                "bandwidth_hz":   _SAMPLE_RATE_HZ,
                "rssi_dbfs":      round(rssi_dbfs, 1),
                "peak_dbfs":      round(rssi_dbfs + 2.0, 1),
                "occupied":       rssi_dbfs > -90.0,
            },
        }


# ─── SELF-TEST ────────────────────────────────────────────────────

def _run_hardware_test() -> None:
    """
    Scan all bands once and print an RSSI table.
    Verifies the rtl_sdr binary and dongle are working.

    Run with:
        python edge/rtlsdr_source.py --test
    """
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    if not shutil.which("rtl_sdr"):
        print("ERROR: 'rtl_sdr' binary not found.")
        print("Install with:  sudo apt install rtl-sdr")
        sys.exit(1)

    print("=" * 60)
    print("RTL-SDR Blog V4 — hardware self-test")
    print("=" * 60)
    print(f"Scanning {len(_ALL_BANDS)} bands (subprocess mode)...\n")

    source = RTLSDRFrameSource()

    print(f"  {'Band':<22} {'Freq (MHz)':>10}  {'RSSI dBFS':>10}  {'Occupied':>8}  Spec")
    print(f"  {'-'*22} {'-'*10}  {'-'*10}  {'-'*8}  ----")

    for band in _ALL_BANDS:
        frame = source._capture_band(band)
        if frame is None:
            print(f"  {band['band_id']:<22} {'ERROR':>10}")
            continue
        rssi     = frame["rssi"]["rssi_dbfs"]
        occupied = "YES" if frame["rssi"]["occupied"] else "no"
        spec     = frame["spectrogram"].shape
        print(
            f"  {band['band_id']:<22} {band['center_freq_hz']/1e6:>10.3f}"
            f"  {rssi:>10.1f}  {occupied:>8}  {spec}"
        )

    print("\nSelf-test complete.")


# ─── ENTRY POINT ──────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="RTL-SDR Blog V4 frame source")
    parser.add_argument("--test", action="store_true", help="Run hardware self-test")
    args = parser.parse_args()

    if args.test:
        _run_hardware_test()
    else:
        parser.print_help()
