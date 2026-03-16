# -*- coding: utf-8 -*-
"""simulation_final.py

Physics-based IQ spectrogram generator for SpectrumEye dataset.

Signal chain (same as real RTL-SDR hardware):
  IQ samples → AWGN noise (IQ level) → STFT → normalize → 224×224 PNG

Three classes:
  Key_Signal     — OOK-keyed carrier (key fob / remote)
  Walkie_Talkie  — FM-modulated carrier (narrowband voice radio)
  LTE            — OFDM wideband signal (4G cellular)

Output folder structure (matches ml/dataset/raw/ directly):
  dataset_rf/
  ├── Key_Signal/      key_{n:04d}.png
  ├── Walkie_Talkie/   walkie_{n:04d}.png
  └── LTE/             lte_{n:04d}.png

After running this script on Colab:
  1. Download dataset_rf/
  2. Copy its contents into ml/dataset/raw/
  3. Run ml/augment.py → ml/split_dataset.py → ml/train.py

──────────────────────────────────────────────────────────────────
CHANGES vs original simulation_final.py
──────────────────────────────────────────────────────────────────
1. fd_gray() REPLACED — matplotlib plt.specgram auto-scales the
   colormap per image, so each image has a different brightness
   mapping. On real hardware, gain is fixed, so the same noise
   floor always maps to the same pixel value. The new function
   spectrogram_to_image() applies the Interface Contract formula
   explicitly:  pixel = clip((P_dBFS - (-100)) / 100 * 255, 0, 255)
   This gives:  noise floor ≈ dark  |  strong signal = bright white

2. LTE class ADDED — lte_simulation() generates an OFDM-like
   wideband signal (sum of 50 subcarriers). Looks like a wide flat
   rectangular band in the spectrogram — very different from the
   narrow lines of Key_Signal and Walkie_Talkie.

3. N_IMAGES_PER_CLASS raised to 500 — after 7× augmentation this
   gives ~3500 images per class, enough for solid CNN training.

4. Output folders renamed to match training pipeline class labels:
   Key_Signal/  Walkie_Talkie/  LTE/  (instead of dataset_gray_key/ etc.)

5. Saves as mode 'L' true grayscale PNG (1 channel) — previous
   version saved RGBA (4 channels) because matplotlib renders even
   gray colormaps as RGBA. 1-channel PNG is 4× smaller and correct.

6. Fixed bug in key_signal_simulation: was using global t_sampling
   instead of the function parameter t.
"""

import os
import numpy as np
from scipy import signal as sp_signal
from PIL import Image

# ─── CONFIGURATION ────────────────────────────────────────────────

N_IMAGES_PER_CLASS = 500   # images generated per class
                           # (× 7 augmentations = 3500 per class for training)

FS       = 1000            # sampling frequency (Hz) — normalized baseband simulation
DURATION = 2               # capture duration (seconds) per sample
NFFT     = 256             # FFT window size (matches original)

# Interface Contract normalization constants
P_MIN = -100.0             # dBFS — noise floor maps to pixel 0  (black)
P_MAX =    0.0             # dBFS — full signal maps to pixel 255 (white)

# ─── TIME AXIS ────────────────────────────────────────────────────

def generate_t_sampling(fs, duration):
    """Time axis: evenly spaced samples at rate fs for `duration` seconds."""
    period = 1 / fs
    return np.arange(0, duration, period)

# ─── SIGNAL MODELS ────────────────────────────────────────────────

def key_signal_simulation(t, fs):
    """
    Key fob / remote control — OOK (On-Off Keying).

    Carrier at 50 Hz, amplitude keyed by a random binary code
    (simulates the burst pattern of a rolling-code key fob).
    In the spectrogram: short thin bright dashes at 50 Hz.
    """
    frequency = 50
    wave = np.exp(1j * 2 * np.pi * frequency * t)   # fixed: uses parameter t

    # Random binary code — controls which time slots are active
    code = np.random.randint(0, 2, 10)
    reps = int(np.ceil(len(t) / len(code)))
    pulse = np.repeat(code, reps)[:len(t)]

    return wave * pulse


def walkie_talkie_simulation(t, fs):
    """
    Walkie-talkie — narrowband FM (voice simulation).

    Carrier at 150 Hz, FM-modulated by a 5 Hz sine (simulates voice).
    Frequency deviation: ±50 Hz.
    In the spectrogram: a narrow continuous band that wobbles vertically
    as the voice modulation shifts the instantaneous frequency.
    """
    f_carrier   = 150
    f_message   = 5            # simulated voice at 5 Hz
    deviation   = 50           # FM deviation (Hz)

    message          = np.sin(2 * np.pi * f_message * t)
    phase_modulation = np.cumsum(message) / fs
    t_phase          = 2 * np.pi * f_carrier * t + 2 * np.pi * deviation * phase_modulation

    return np.exp(1j * t_phase)


def lte_simulation(t, fs):
    """
    LTE — OFDM wideband signal.

    Simulated as a sum of 50 equally-spaced subcarriers (simplified OFDM).
    Center at 300 Hz, total bandwidth 200 Hz (spans 200–400 Hz).
    Each subcarrier gets a random initial phase (simulates independent data
    on each OFDM resource element).

    In the spectrogram: a wide flat rectangular band — distinguishable from
    the narrow line of Key_Signal and the wobbling stripe of Walkie_Talkie.
    """
    n_subcarriers = 50
    f_center      = fs * 0.30   # 300 Hz center
    bandwidth     = fs * 0.20   # 200 Hz total bandwidth

    df     = bandwidth / n_subcarriers   # subcarrier spacing
    signal = np.zeros(len(t), dtype=complex)

    for k in range(n_subcarriers):
        f_k = f_center + (k - n_subcarriers // 2) * df
        phi = np.random.uniform(0, 2 * np.pi)   # random phase per subcarrier
        signal += np.exp(1j * (2 * np.pi * f_k * t + phi))

    # Normalize: total amplitude ≈ 1 (same scale as other signals)
    signal /= np.sqrt(n_subcarriers)
    return signal


def awgn(signal, noise_level=0.5):
    """
    Additive White Gaussian Noise — applied at IQ level (before FFT).

    This is physically correct: noise enters the signal chain before
    the spectrogram is computed, so it interacts with the signal in
    the frequency domain exactly as in real hardware.

    noise_level=0.5 gives SNR ≈ 6 dB (noisy but clearly classifiable).
    """
    real_noise    = np.random.normal(0, noise_level, len(signal))
    complex_noise = np.random.normal(0, noise_level, len(signal)) * 1j
    return signal + real_noise + complex_noise


# ─── SPECTROGRAM → IMAGE ──────────────────────────────────────────

def spectrogram_to_image(iq_signal, fs, nfft=NFFT):
    """
    Convert IQ samples to a 224×224 uint8 grayscale PIL Image.

    Processing chain:
      1. STFT (two-sided, hanning window, 50% overlap)
      2. fftshift — DC shifted to center row
      3. Interface Contract normalization:
           P_dBFS = 20·log10(|Zxx| + ε) − 20·log10(NFFT)
           pixel  = clip((P_dBFS − P_MIN) / (P_MAX − P_MIN), 0, 1) × 255
      4. Flip rows: row 0 = lowest frequency
      5. Resize to 224×224 (LANCZOS)
      6. Save as mode 'L' (1-channel grayscale, NOT RGBA)

    The normalization is fixed (not auto-scaled), so noise floor always
    maps to the same dark pixel value across all images. This matches
    what the RTL-SDR hardware produces with a fixed gain setting.
    """
    hop    = nfft // 2
    window = np.hanning(nfft)

    # Two-sided STFT for complex IQ — captures both sidebands
    _, _, Zxx = sp_signal.stft(
        iq_signal,
        fs=fs,
        window=window,
        nperseg=nfft,
        noverlap=nfft - hop,
        return_onesided=False,
    )

    # Shift DC bin to the center row
    Zxx = np.fft.fftshift(Zxx, axes=0)

    # Interface Contract normalization (amplitude spectrum in dBFS)
    power_dbfs = 20 * np.log10(np.abs(Zxx) + 1e-10) - 20 * np.log10(nfft)
    normalized = np.clip((power_dbfs - P_MIN) / (P_MAX - P_MIN), 0, 1)
    pixels     = (normalized * 255).astype(np.uint8)

    # Flip so row 0 = lowest frequency (Interface Contract axis convention)
    pixels = pixels[::-1, :]

    # Resize to 224×224 and save as true grayscale (mode L, 1 channel)
    img = Image.fromarray(pixels, mode='L')
    img = img.resize((224, 224), Image.LANCZOS)
    return img


# ─── DATASET GENERATION ───────────────────────────────────────────

def generate_dataset(n_per_class=N_IMAGES_PER_CLASS, fs=FS, duration=DURATION):
    """
    Generate N_IMAGES_PER_CLASS spectrograms for each of the three classes.

    Output structure:
      dataset_rf/Key_Signal/    key_{n:04d}.png
      dataset_rf/Walkie_Talkie/ walkie_{n:04d}.png
      dataset_rf/LTE/           lte_{n:04d}.png
    """
    base_dir  = "dataset_rf"
    class_dirs = {
        "Key_Signal":   os.path.join(base_dir, "Key_Signal"),
        "Walkie_Talkie":os.path.join(base_dir, "Walkie_Talkie"),
        "LTE":          os.path.join(base_dir, "LTE"),
    }

    for path in class_dirs.values():
        os.makedirs(path, exist_ok=True)

    signal_fns = {
        "Key_Signal":    (key_signal_simulation,    "key"),
        "Walkie_Talkie": (walkie_talkie_simulation,  "walkie"),
        "LTE":           (lte_simulation,            "lte"),
    }

    t = generate_t_sampling(fs, duration)

    print(f"Generating {n_per_class} images × 3 classes = {n_per_class * 3} total\n")

    for class_name, (sim_fn, prefix) in signal_fns.items():
        out_dir = class_dirs[class_name]
        print(f"  {class_name}: ", end="", flush=True)

        for n in range(n_per_class):
            pure_signal  = sim_fn(t, fs)
            noisy_signal = awgn(pure_signal, noise_level=0.5)
            img          = spectrogram_to_image(noisy_signal, fs)
            img.save(os.path.join(out_dir, f"{prefix}_{n:04d}.png"))

            if (n + 1) % 100 == 0:
                print(f"{n + 1}..", end="", flush=True)

        print(f" done ({n_per_class} images)")

    print(f"\nDone. Dataset saved to: {base_dir}/")
    print("Next steps:")
    print("  1. Copy dataset_rf/ contents into ml/dataset/raw/")
    print("  2. python ml/augment.py")
    print("  3. python ml/split_dataset.py")
    print("  4. python ml/train.py")


# ─── ENTRY POINT ──────────────────────────────────────────────────

if __name__ == "__main__":
    generate_dataset()
