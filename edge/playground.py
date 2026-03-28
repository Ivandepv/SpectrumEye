"""
edge/playground.py — RTL-SDR hardware diagnostic

One-shot script: connect RTL-SDR, capture IQ, plot PSD.
Run this to confirm the dongle is recognised and capturing before
running the full pipeline.

Usage:
    python edge/playground.py
"""

import logging
import sys

import matplotlib.pyplot as plt
from rtlsdr import RtlSdr


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    log = logging.getLogger(__name__)

    # 1. Connect to the SDR dongle
    try:
        sdr = RtlSdr()
    except Exception as e:
        log.error("Failed to connect to RTL-SDR: %s", e)
        log.error("Tip: check USB connection or run with sudo.")
        sys.exit(1)

    # 2. Configure receiver — tuned to 1090 MHz (ADS-B)
    sdr.sample_rate = 2.048e6
    sdr.center_freq = 1090e6
    sdr.gain = "auto"

    log.info("Hardware connected. Capturing IQ samples...")

    # 3. Capture ~1/8 s of IQ data
    samples = sdr.read_samples(256 * 1024)

    # 4. Release the USB device
    sdr.close()

    log.info("Captured %d IQ samples.", len(samples))
    log.info("First 4 samples: %s", samples[:4])

    # 5. Plot power spectral density
    plt.figure(figsize=(10, 5))
    plt.psd(
        samples,
        NFFT=1024,
        Fs=sdr.sample_rate / 1e6,
        Fc=sdr.center_freq / 1e6,
        color="cyan",
    )
    plt.title("RTL-SDR Diagnostic — 1090 MHz (ADS-B)")
    plt.xlabel("Frequency (MHz)")
    plt.ylabel("Signal strength (dB/Hz)")
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.style.use("dark_background")
    plt.show()


if __name__ == "__main__":
    main()
