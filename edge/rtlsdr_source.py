"""
edge/rtlsdr_source.py — Real-time RTL-SDR Blog V4 frame source

Sweeps 10 frequency bands in round-robin order.
For each band:
  1. Tunes the SDR to the band's center frequency
  2. Waits 25 ms for the R828D PLL to settle
  3. Captures 256×1024 IQ samples (~128 ms at 2.048 MSPS)
  4. Generates a 224×224 uint8 spectrogram using matplotlib Agg —
     identical preprocessing to training data (data_collector.py)
  5. Computes RSSI (dBFS) from IQ power
  6. Yields a sweep_frame dict (Interface A/A+) for EdgePipeline

RTL-SDR Blog V4 hardware notes
-------------------------------
- Receive-only passive device: cannot transmit, no RF damage possible.
- Sample rate: fixed at 2.048 MSPS (stable limit for R828D; DO NOT raise).
- Gain: "auto" by default — lets the tuner avoid ADC saturation automatically.
  Pass --gain <dB> (e.g. 30) only if "auto" gives poor sensitivity.
- PLL settling: 25 ms after each retune before samples are valid (not harmful
  if skipped, just gives invalid samples for one frame).
- Thermal: the dongle warms up during continuous use — this is normal.
  Ensure airflow; do not enclose in an airtight case.
- USB power: ~300 mA draw from the Pi 5 USB port (well within 900 mA limit).
- Cleanup: sdr.close() is always called via try/finally so the USB device
  is released cleanly on exit or error.

Usage:
    source = RTLSDRFrameSource()
    for frame in source.frames():
        pipeline.process(frame)

    # Custom band subset or gain:
    source = RTLSDRFrameSource(
        bands=["walkie_talkie", "aircraft_tracking", "cellular_network"],
        gain=30,
    )
"""

import io
import logging
import time
from typing import Iterator, Optional

# matplotlib MUST use Agg (non-interactive) backend before any pyplot import.
# This matches the training data pipeline in data_collector.py and avoids
# display errors on the headless Pi.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
from PIL import Image
from rtlsdr import RtlSdr

log = logging.getLogger(__name__)

# ─── HARDWARE CONSTANTS ───────────────────────────────────────────

# Stable sample rate for RTL-SDR Blog V4 (R828D tuner).
# 2.4 MSPS is also stable but 2.048 matches training data exactly.
_SAMPLE_RATE_HZ: float = 2.048e6

# Number of IQ samples per capture per band.
# 256 * 1024 = 262 144 samples → ~128 ms at 2.048 MSPS.
# This matches data_collector.py and produces a good spectrogram.
_NUM_SAMPLES: int = 256 * 1024

# PLL settling time (seconds) after each frequency change.
# The R828D needs ~20–25 ms to lock its synthesiser.
# Samples taken before this settle are meaningless but cause no damage.
_PLL_SETTLE_SEC: float = 0.025

# ─── FREQUENCY BANDS ─────────────────────────────────────────────

# One fixed center frequency per training class.
# Frequencies are chosen to land in the most active part of each band
# and are well within the RTL-SDR Blog V4 range (500 kHz – 1.75 GHz).
#
# Scan order is also the round-robin order: lower frequencies first
# so initial results appear quickly after startup.
_ALL_BANDS: list[dict] = [
    # band_id must match the CNN class label exactly
    {"band_id": "radio_fm",             "center_freq_hz":    98_000_000},  # FM broadcast center
    {"band_id": "air_traffic",          "center_freq_hz":   122_000_000},  # Int. emergency 121.5 area
    {"band_id": "noaa",                 "center_freq_hz":   137_500_000},  # NOAA APT center
    {"band_id": "local_repeaters",      "center_freq_hz":   146_000_000},  # 2m band center
    {"band_id": "maritime",             "center_freq_hz":   156_800_000},  # VHF Ch16 distress/calling
    {"band_id": "short_range_devices",  "center_freq_hz":   315_000_000},  # 315 MHz ISM (car keys)
    {"band_id": "wireless_controllers", "center_freq_hz":   433_920_000},  # 433.92 MHz ISM exact
    {"band_id": "walkie_talkie",        "center_freq_hz":   446_000_000},  # PMR446 center
    {"band_id": "cellular_network",     "center_freq_hz":   900_000_000},  # GSM-900 center
    {"band_id": "aircraft_tracking",    "center_freq_hz": 1_090_000_000},  # ADS-B exact frequency
]

# Validate all frequencies are within Blog V4 range
_MIN_FREQ_HZ = 500_000       # 500 kHz
_MAX_FREQ_HZ = 1_750_000_000 # 1.75 GHz
for _b in _ALL_BANDS:
    assert _MIN_FREQ_HZ <= _b["center_freq_hz"] <= _MAX_FREQ_HZ, (
        f"Frequency {_b['center_freq_hz']} for {_b['band_id']} out of RTL-SDR Blog V4 range"
    )


# ─── SPECTROGRAM GENERATION ───────────────────────────────────────

def _iq_to_spectrogram(samples: np.ndarray, fs: float) -> np.ndarray:
    """
    Convert complex IQ samples to a 224×224 uint8 grayscale spectrogram.

    Uses matplotlib Agg with the EXACT same parameters as data_collector.py
    so the output matches the training data distribution pixel-for-pixel.

    Args:
        samples: complex IQ array from sdr.read_samples()
        fs:      sample rate in Hz

    Returns:
        224×224 uint8 numpy array (grayscale, 0=noise floor, 255=strong signal)
    """
    fig = plt.figure(figsize=(2.24, 2.24), dpi=100)
    plt.specgram(samples, NFFT=256, Fs=fs, cmap="gray")
    plt.axis("off")
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.margins(0, 0)

    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=100)
    plt.close(fig)

    buf.seek(0)
    img = Image.open(buf).convert("L")   # ensure single-channel grayscale
    img = img.resize((224, 224), Image.LANCZOS)  # guarantee exact size
    return np.array(img, dtype=np.uint8)


def _compute_rssi(samples: np.ndarray) -> float:
    """
    Compute received signal strength in dBFS from complex IQ samples.

    rtlsdr normalises samples to approximately [-1+0j, 1+0j] complex range,
    so 0 dBFS = full ADC swing.

    Args:
        samples: complex IQ array from sdr.read_samples()

    Returns:
        RSSI in dBFS (typically -80 to -20 for real signals)
    """
    power = float(np.mean(np.abs(samples) ** 2))
    return float(10.0 * np.log10(power + 1e-12))


# ─── FRAME SOURCE ─────────────────────────────────────────────────

class RTLSDRFrameSource:
    """
    Real-time sweep frame source using RTL-SDR Blog V4.

    Sweeps through the configured bands in round-robin order.
    Yields sweep_frame dicts (Interface A/A+) compatible with EdgePipeline.

    Args:
        bands:    list of band_id strings to include in the sweep.
                  Defaults to all 10 bands. Pass a subset to limit scanning.
        gain:     SDR gain in dB, or "auto" (default).
                  "auto" is recommended — it prevents ADC saturation on
                  strong nearby signals without manual adjustment.
        device_index: RTL-SDR device index (0 for single dongle).
    """

    def __init__(
        self,
        bands: Optional[list[str]] = None,
        gain: float | str = "auto",
        device_index: int = 0,
    ) -> None:
        # Select bands
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
        self._sdr: Optional[RtlSdr] = None

    # ── Public API ────────────────────────────────────────────────

    def frames(self) -> Iterator[dict]:
        """
        Open the RTL-SDR, then yield sweep_frames indefinitely.

        Handles cleanup via try/finally — sdr.close() is always called,
        even on exception or KeyboardInterrupt, so the USB device is
        released cleanly.
        """
        log.info("RTLSDRFrameSource: opening device (index=%d)", self._device_index)
        try:
            self._sdr = RtlSdr(self._device_index)
            self._sdr.sample_rate = _SAMPLE_RATE_HZ
            self._sdr.gain        = self._gain
            log.info(
                "RTLSDRFrameSource: ready — sample_rate=%.3f MSPS  gain=%s  bands=%d",
                _SAMPLE_RATE_HZ / 1e6,
                self._gain,
                len(self._bands),
            )
            yield from self._sweep_loop()
        except Exception as exc:
            log.error("RTLSDRFrameSource: fatal error: %s", exc)
            raise
        finally:
            if self._sdr is not None:
                try:
                    self._sdr.close()
                    log.info("RTLSDRFrameSource: SDR device closed cleanly")
                except Exception:
                    pass  # device may already be gone
                self._sdr = None

    # ── Sweep loop ────────────────────────────────────────────────

    def _sweep_loop(self) -> Iterator[dict]:
        """Round-robin through all configured bands indefinitely."""
        band_idx = 0
        while True:
            band     = self._bands[band_idx]
            band_idx = (band_idx + 1) % len(self._bands)

            frame = self._capture_band(band)
            if frame is not None:
                yield frame

    def _capture_band(self, band: dict) -> Optional[dict]:
        """
        Tune to one band, capture IQ, generate spectrogram, return sweep_frame.

        Returns None if capture fails (logs the error, continues sweep).
        """
        freq_hz = band["center_freq_hz"]
        band_id = band["band_id"]

        try:
            # 1. Retune
            self._sdr.center_freq = freq_hz

            # 2. PLL settling — wait for R828D synthesiser to lock.
            #    Without this, the first samples after a retune are from
            #    the previous frequency (invalid but harmless to hardware).
            time.sleep(_PLL_SETTLE_SEC)

            # 3. Capture IQ
            samples = self._sdr.read_samples(_NUM_SAMPLES)

            # 4. Compute RSSI from raw power
            rssi_dbfs = _compute_rssi(samples)
            peak_dbfs = rssi_dbfs + 2.0  # headroom estimate

            # 5. Generate 224×224 spectrogram (same pipeline as training data)
            spectrogram = _iq_to_spectrogram(samples, _SAMPLE_RATE_HZ)

        except Exception as exc:
            log.warning("RTLSDRFrameSource: capture failed for %s @ %.1f MHz: %s",
                        band_id, freq_hz / 1e6, exc)
            return None

        # 6. Build sweep_frame (Interface A/A+)
        self._frame_id += 1
        return {
            "frame_id":       self._frame_id,
            "timestamp_ms":   int(time.time() * 1000),
            "center_freq_hz": freq_hz,
            "sample_rate_hz": int(_SAMPLE_RATE_HZ),
            "gain_db":        self._gain,
            "spectrogram":    spectrogram,
            "rssi": {
                "band_id":        band_id,
                "center_freq_hz": freq_hz,
                "bandwidth_hz":   int(_SAMPLE_RATE_HZ),
                "rssi_dbfs":      round(rssi_dbfs, 1),
                "peak_dbfs":      round(peak_dbfs, 1),
                "occupied":       rssi_dbfs > -90.0,
            },
            # No _sim_class / _sim_conf — pipeline will run real CNN inference
        }


# ─── SELF-TEST ────────────────────────────────────────────────────

def _run_hardware_test() -> None:
    """
    Open the SDR, capture one frame per band, print RSSI table.
    Use this to verify the dongle is connected and working before
    running the full pipeline.

    Run with:
        python edge/rtlsdr_source.py --test
    """
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    print("=" * 60)
    print("RTL-SDR Blog V4 — hardware self-test")
    print("=" * 60)
    print(f"Scanning {len(_ALL_BANDS)} bands once each...\n")

    source = RTLSDRFrameSource()

    try:
        source._sdr = RtlSdr(source._device_index)
        source._sdr.sample_rate = _SAMPLE_RATE_HZ
        source._sdr.gain        = source._gain

        print(f"  {'Band':<22} {'Freq (MHz)':>10}  {'RSSI dBFS':>10}  {'Occupied':>8}  {'Spec shape'}")
        print(f"  {'-'*22} {'-'*10}  {'-'*10}  {'-'*8}  {'-'*10}")

        for band in _ALL_BANDS:
            frame = source._capture_band(band)
            if frame is None:
                print(f"  {band['band_id']:<22} {'ERROR':>10}")
                continue
            rssi     = frame["rssi"]["rssi_dbfs"]
            occupied = "YES" if frame["rssi"]["occupied"] else "no"
            spec     = frame["spectrogram"].shape
            print(
                f"  {band['band_id']:<22} {band['center_freq_hz']/1e6:>10.3f}  "
                f"{rssi:>10.1f}  {occupied:>8}  {str(spec)}"
            )

    finally:
        if source._sdr is not None:
            source._sdr.close()
            print("\nSDR closed cleanly.")

    print("\nSelf-test complete.")


# ─── ENTRY POINT ──────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="RTL-SDR Blog V4 frame source")
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run hardware self-test: scan all bands once and print RSSI",
    )
    args = parser.parse_args()

    if args.test:
        _run_hardware_test()
    else:
        parser.print_help()
