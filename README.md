# SpectrumEye

**RF Situational Awareness System** — real-time detection and classification of radio frequency signals using a physics-based IQ simulation pipeline, a CNN trained on spectrograms, and a Behavioral Inference Engine that translates signal patterns into human-readable threat assessments.

Built as a defense-oriented project in Tainan, Taiwan. Current state: **fully working simulation/demo** with a trained CNN (100% test accuracy on 3 signal classes).

---

## What It Does

SpectrumEye listens to the RF spectrum, converts IQ samples into spectrograms, and classifies them with a convolutional neural network. A Behavioral Inference Engine (BIE) then tracks signal movement over time — is the source approaching? stationary? departing? — and generates an assessment sentence in plain language. Results appear on a real-time React dashboard.

```
IQ Samples → STFT → 224×224 Spectrogram → CNN → Signal Class + Confidence
                                                         ↓
                                               Behavioral Inference Engine
                                               (RSSI tracking, trend analysis)
                                                         ↓
                                          "Key fob signal approaching rapidly —
                                           possible remote device in perimeter"
                                                         ↓
                                               React Dashboard / AWS IoT
```

---

## Current State — Demo / Simulation Mode

Everything below is **fully implemented and working**:

| Component | Status | Notes |
|-----------|--------|-------|
| Physics-based dataset generator | ✅ Done | `simulation/simulation_final.py` |
| Data augmentation pipeline | ✅ Done | 7 transforms, 400 → 1 400 images/class |
| CNN training (Google Colab) | ✅ Done | MobileNetV2, 50 epochs, T4 GPU |
| CNN accuracy | ✅ 100% | Level 1 (seed=999) and Level 2 (300 unseen images) |
| Behavioral Inference Engine | ✅ Done | `edge/bie.py` — 8 behavioral states |
| Edge pipeline (Pi 5 ready) | ✅ Done | `edge/main.py` — sim / demo / socket modes |
| React dashboard | ✅ Done | Simulation mode with scripted key-fob scenario |
| Cloud pipeline (AWS) | 🔲 Stub | `edge/aws_publisher.py` — interface ready, not connected |
| Real RTL-SDR hardware | 🔲 Next | Edge code ready, hardware not yet connected |
| TFLite INT8 conversion | 🔲 Next | Needed for <50ms on Pi 5 |

---

## Signal Classes

The CNN currently classifies **3 signal types**. The architecture is designed to expand to more classes as the model is retrained.

| Class | Description | Frequency | Threat Level |
|-------|-------------|-----------|--------------|
| `Key_Signal` | OOK-keyed carrier — key fob, remote, possible detonator | 433 MHz | ELEVATED |
| `Walkie_Talkie` | Narrowband FM — voice radio, unauthorized comms | 162 MHz | MODERATE |
| `LTE` | OFDM wideband — 4G cellular, background | 2100 MHz | MONITOR |

---

## System Architecture

```
┌─────────────────────────────── EDGE (Raspberry Pi 5) ──────────────────────────────┐
│                                                                                      │
│  [RTL-SDR Blog V4] ──USB──► IQ Capture ──► STFT (nfft=256, hanning, 50% OVL)       │
│                                                    │                                 │
│                                              Interface A                             │
│                                          (224×224 uint8 PNG)                        │
│                                                    │                                 │
│                                       ┌────────────▼─────────────┐                  │
│                                       │  SpectrumClassifier       │                  │
│                                       │  MobileNetV2 (α=0.75)    │                  │
│                                       │  1.74M params             │                  │
│                                       └────────────┬─────────────┘                  │
│                                              Interface B                             │
│                                    {class, confidence, probs, ms}                   │
│                                                    │                                 │
│               ┌────────────────────────────────────▼───────────────────┐            │
│               │            Behavioral Inference Engine (BIE)           │            │
│               │  RSSI tracking · trend analysis · 8 behavioral states  │            │
│               │  APPEARED / APPROACHING_FAST / STATIONARY / DEPARTED…  │            │
│               └────────────────────────────────────┬───────────────────┘            │
│                                              Interface C                             │
│                            {threat_level, sentence, state, rssi_history}            │
│                    ┌───────────────────┬────────────┴──────────────┐                │
│                    ▼                   ▼                            ▼                │
│          [Local Display]      [Alert Controller]          [AWS Publisher]            │
│        (Flask / terminal)    (LED · buzzer · sound)       (IoT Core MQTT)           │
└─────────────────────────────────────────────────────────────────────────────────────┘
                                                                       │
                                                                  Interface D
                                                                       │
                                                            ┌──────────▼──────────┐
                                                            │   AWS IoT Core       │
                                                            │   → Kinesis          │
                                                            │   → Lambda           │
                                                            │   → DynamoDB         │
                                                            └──────────┬──────────┘
                                                                       │ WebSocket
                                                            ┌──────────▼──────────┐
                                                            │   React Dashboard    │
                                                            │   (live mode)        │
                                                            └─────────────────────┘

[ESP32S + Grove sensors] ──MQTT──► Pi 5  (Interface E — temperature, humidity, motion, sound)
```

---

## Project Structure

```
SpectrumEye/
│
├── simulation/
│   └── simulation_final.py          # Physics-based IQ → spectrogram generator
│                                    # (OOK key signal, narrowband FM, OFDM LTE)
│
├── ml/                              # CNN training pipeline
│   ├── augment.py                   # 7 augmentation transforms per image
│   ├── split_dataset.py             # Stratified train/val/test split (70/15/15)
│   ├── train.py                     # MobileNetV2 training (Keras/TF)
│   ├── evaluate.py                  # --quick / --folder / --image evaluation
│   ├── generate_test_batch.py       # Level 2 test set generator (seed=777)
│   ├── prepare_colab_zip.py         # Packages ml_training_v2.zip for Colab upload
│   ├── collect_synthetic.py         # Legacy: simple pixel-paint synthetic generator
│   ├── requirements.txt             # Python dependencies
│   ├── setup_env.sh                 # pyenv + venv setup helper
│   ├── notebooks/
│   │   └── SpectrumEye_Training.ipynb   # 9-cell Colab notebook (upload → train → download)
│   ├── dataset/
│   │   ├── raw/                     # 400 physics-based PNGs per class
│   │   ├── augmented/               # 1 400 PNGs per class (7× augmentation)
│   │   ├── train/                   # 980 per class  (70%)
│   │   ├── val/                     # 210 per class  (15%)
│   │   └── test/                    # 210 per class  (15%)
│   └── models/
│       ├── production/
│       │   └── spectromeye_best.keras   # ← evaluate.py default model
│       └── v2_colab/
│           ├── best_model.keras         # Same weights, with training artefacts
│           ├── config.json
│           ├── history.json
│           ├── classification_report.txt
│           ├── confusion_matrix.png
│           └── training_curves.png
│
├── edge/                            # Raspberry Pi 5 runtime
│   ├── main.py                      # Orchestration loop (--sim / --demo / --socket)
│   ├── classifier.py                # CNN inference wrapper (Interface B)
│   ├── bie.py                       # Behavioral Inference Engine (Interface C)
│   ├── rssi_tracker.py              # Re-exports RSSITracker from bie.py
│   ├── sensor_fusion.py             # ESP32 MQTT subscriber (Interface E)
│   ├── aws_publisher.py             # AWS IoT Core publisher stub (Interface D)
│   ├── alert_controller.py          # Threat alerts: terminal · GPIO · sound
│   └── local_display.py             # Terminal (ANSI) or Flask (HDMI) display
│
├── cloud/                           # AWS infrastructure (future)
│   ├── cdk/                         # AWS CDK stack (not yet written)
│   └── lambda/                      # Lambda functions (not yet written)
│
├── webapp/                          # React dashboard
│   ├── src/
│   │   ├── main.tsx                 # Entry point
│   │   ├── SpectrumEyeDashboard.jsx # Dashboard — simulation + future live mode
│   │   └── index.css                # Global styles
│   ├── index.html
│   ├── vite.config.ts
│   └── package.json
│
├── docs/                            # Design references (git-ignored)
│   ├── SpectrumEye_Interface_Contract.md
│   ├── SpectrumEye_CS_Roadmap.md
│   └── SpectrumEye_Dashboard.jsx    # Original full 9-class design reference
│
├── .gitignore
└── README.md
```

---

## Quick Start

### 1. Run the Dashboard (demo mode — no hardware needed)

```bash
cd webapp
npm install
npm run dev
# → http://localhost:5173
```

Click **▶ SIMULATE KEY FOB** to watch a scripted scenario: a key-fob signal appears, approaches, holds position, then departs — with real-time BIE assessment and alert log updates.

### 2. Set Up the ML Environment

Requires Python 3.12 (TensorFlow is not yet compatible with 3.13+).

```bash
# Install pyenv if not present (Arch Linux)
sudo pacman -S pyenv

# Add to ~/.zshrc / ~/.bashrc
export PYENV_ROOT="$HOME/.pyenv"
export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init -)"

# Set up the ML virtual environment
pyenv install 3.12.9
cd ml
bash setup_env.sh          # creates .venv and installs requirements.txt
source .venv/bin/activate
```

### 3. Evaluate the Trained Model

```bash
cd /path/to/SpectrumEye

# Level 1 — quick: 30 fresh simulation images, in-memory, no files needed
python ml/evaluate.py --quick

# Level 2 — batch: generate 300 unseen images to disk, then evaluate
python ml/generate_test_batch.py          # → test_batch/ (300 PNGs, seed=777)
python ml/evaluate.py --folder test_batch/

# Single image
python ml/evaluate.py --image path/to/spectrogram.png
```

### 4. Retrain the CNN (Google Colab — recommended)

The model trains best on a free Colab T4 GPU (~20–30 min for 50 epochs).

```bash
# Step 1: Generate the training dataset locally
python simulation/simulation_final.py
# → simulation/dataset_rf/  (1 200 raw PNGs)

# Step 2: Package everything for Colab
python ml/prepare_colab_zip.py
# → ml_training_v2.zip  (~25 MB)

# Step 3: Open Colab and run the notebook
# ml/notebooks/SpectrumEye_Training.ipynb
# Cell 3: upload ml_training_v2.zip
# Cell 4: augment (1 200 raw → 4 200 augmented)
# Cell 5: split into train/val/test
# Cell 7: train 50 epochs on T4 GPU
# Cell 9: download spectromeye_v2_colab.zip

# Step 4: Install the trained model
# Extract best_model.keras → ml/models/production/spectromeye_best.keras
```

### 5. Run the Edge Pipeline (simulation mode)

```bash
# Requires: ml/models/production/spectromeye_best.keras
source ml/.venv/bin/activate

# Simulation mode — synthetic frames, no hardware
python edge/main.py --sim

# Scripted demo — 4-phase scenario (appeared → approach → hover → depart)
python edge/main.py --demo

# Hardware mode — reads from Unix socket (real RTL-SDR pipeline)
python edge/main.py --socket /tmp/spectromeye_frames.sock
```

---

## Dataset

Generated with a physics-based IQ simulation (`simulation/simulation_final.py`) that replicates the real RTL-SDR hardware signal chain:

```
Random IQ signal → AWGN noise (IQ level, SNR ≈ 6 dB) → STFT → dBFS normalization → 224×224 PNG
```

The normalization is fixed (not auto-scaled), so the noise floor always maps to the same dark pixel value — consistent with a real receiver at fixed gain.

| Split | Per class | Total |
|-------|-----------|-------|
| Raw (generated) | 400 | 1 200 |
| Augmented | 1 400 | 4 200 |
| Train (70%) | 980 | 2 940 |
| Val (15%) | 210 | 630 |
| Test (15%) | 210 | 630 |

**Augmentation transforms** (applied in `ml/augment.py`):
`time_shift` · `freq_shift` · `awgn` · `amplitude_scale` · `noise_mix` · `time_flip` + original

**Signal models** (in `simulation/simulation_final.py`):
- **Key_Signal** — OOK keying: random 10-bit code, carrier at 50 Hz, burst pattern
- **Walkie_Talkie** — Narrowband FM: 5 Hz voice modulation, ±50 Hz deviation
- **LTE** — OFDM: 50 subcarriers, 200 Hz bandwidth, random phase per subcarrier

---

## CNN Model

| Parameter | Value |
|-----------|-------|
| Architecture | MobileNetV2 (α = 0.75) |
| Total parameters | 1 743 283 |
| Trainable parameters | 1 497 587 |
| Input | 224 × 224 × 1 (grayscale) |
| Output | 3 classes (softmax) |
| Optimizer | Adam, lr = 1e-4 |
| Batch size | 32 |
| Epochs | 50 (full run, no early stop) |
| Test accuracy | **100%** (630 images, seed=777) |
| Inference time | 75.6 ms / frame (dev CPU) |
| Training platform | Google Colab T4 GPU |

The confidence threshold in `evaluate.py` is 0.60 — predictions below this are reported as `Unknown`.

---

## Behavioral Inference Engine

`edge/bie.py` translates a stream of CNN outputs into behavioral assessments.

**8 behavioral states:**

| State | Meaning |
|-------|---------|
| `APPEARED` | Signal just became visible |
| `APPROACHING_SLOW` | RSSI rising slowly |
| `APPROACHING_FAST` | RSSI rising fast — high priority |
| `STATIONARY` | RSSI stable — source not moving |
| `DEPARTING_SLOW` | RSSI falling slowly |
| `DEPARTING_FAST` | RSSI falling fast |
| `ERRATIC` | RSSI fluctuating — source moving unpredictably |
| `DISAPPEARED` | Signal lost |

The BIE also fuses environmental data from ESP32S sensors (temperature, humidity, motion, sound) to produce a final context-aware output sentence displayed on the dashboard.

---

## Interface Contract

All data handoffs between components follow the Interface Contract defined in `docs/SpectrumEye_Interface_Contract.md`.

**Interface A — Spectrogram frame** (DSP → CNN):
- `numpy.ndarray`, shape `(224, 224)`, dtype `uint8`, grayscale
- Normalization: `pixel = clip((P_dBFS − (−100)) / 100 × 255, 0, 255)`
- STFT: `nfft=256`, Hanning window, 50% overlap, two-sided (complex IQ)

**Interface B — CNN output** (Classifier → BIE):
```json
{
  "frame_id": 42,
  "timestamp_ms": 1704067200000,
  "center_freq_hz": 433920000,
  "predicted_class": "Key_Signal",
  "confidence": 0.9741,
  "confidence_level": "HIGH",
  "all_probabilities": {"Key_Signal": 0.9741, "Walkie_Talkie": 0.0201, "LTE": 0.0058},
  "inference_time_ms": 75.6,
  "model_version": "v2_colab"
}
```

**Interface C — BIE output** (BIE → Dashboard/Cloud):
```json
{
  "timestamp_ms": 1704067200000,
  "signal_class": "Key_Signal",
  "behavioral_state": "APPROACHING_FAST",
  "threat_level": "CRITICAL",
  "rssi_dbm": -48,
  "rssi_trend_db_per_s": 4.2,
  "assessment": "Key fob signal approaching rapidly — possible remote device in perimeter",
  "env_context": {"temperature_c": 28.4, "humidity_pct": 72, "motion_detected": false}
}
```

---

## Roadmap

| Phase | Description | Status |
|-------|-------------|--------|
| 1 | Environment & project setup | ✅ Complete |
| 2 | Dataset collection & augmentation | ✅ Complete |
| 3 | Dashboard MVP (simulation) | ✅ Complete |
| 4 | CNN training pipeline | ✅ Complete — 100% accuracy |
| 5 | Behavioral Inference Engine | ✅ Complete |
| 6 | AWS cloud pipeline | 🔲 Stub ready — CDK to be written |
| 7 | Edge integration (Pi 5 + RTL-SDR) | 🔲 Code ready — hardware not yet connected |
| 8 | TFLite INT8 conversion | 🔲 Needed for <50ms on Pi 5 |
| 9 | Expand to more signal classes | 🔲 Future — DJI OcuSync, FPV, WiFi, ADS-B |
| 10 | Integration testing & demo | 🔲 After hardware is connected |

---

## Tech Stack

**Machine Learning**
- Python 3.12 · TensorFlow / Keras · NumPy · SciPy · Pillow · scikit-learn
- Training: Google Colab T4 GPU (free tier)

**Edge (Raspberry Pi 5)**
- Python 3.12 · pyrtlsdr · NumPy · SciPy
- Flask (local HDMI display) · paho-mqtt (ESP32 sensors) · gpiod (GPIO alerts)

**Dashboard**
- React 19 · TypeScript · Vite 7 · Tailwind CSS v4 · Recharts 3.7

**Cloud (planned)**
- AWS IoT Core · Kinesis Data Streams · Lambda · DynamoDB · API Gateway WebSocket

**Hardware**
- Raspberry Pi 5 (8 GB)
- RTL-SDR Blog V4 (24 MHz – 1.766 GHz, ~2 MHz instantaneous BW)
- Basys 3 FPGA (Artix-7) — optional DSP acceleration
- ESP32S + Grove sensors (temperature, humidity, PIR motion, sound)

---

## Authors

Jorge Coronado - Memo - Alan
