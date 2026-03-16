# SpectrumEye

**RF Situational Awareness System** — real-time detection and classification of radio frequency signals using a physics-based IQ simulation pipeline, a CNN trained on spectrograms, and a Behavioral Inference Engine that translates signal patterns into human-readable threat assessments.

Built as a defense-oriented project in Tainan, Taiwan. Current state: **fully working end-to-end simulation** — edge pipeline, CNN classifier, WebSocket bridge, and military radar dashboard all connected.

---

## What It Does

SpectrumEye listens to the RF spectrum, converts IQ samples into spectrograms, and classifies them with a convolutional neural network. A Behavioral Inference Engine (BIE) tracks signal movement over time — is the source approaching? stationary? departing? — and generates a plain-language assessment. Results stream in real time to a phosphor radar dashboard via WebSocket.

```
IQ Samples → STFT → 224×224 Spectrogram → CNN → Signal Class + Confidence
                                                         ↓
                                               Behavioral Inference Engine
                                               (RSSI tracking, trend analysis)
                                                         ↓
                                          "Key fob signal approaching rapidly —
                                           possible remote device in perimeter"
                                                         ↓
                                          WebSocket → React Radar Dashboard
```

---

## Current State

| Component | Status | Notes |
|-----------|--------|-------|
| Physics-based dataset generator | ✅ Done | `simulation/simulation_final.py` |
| Data augmentation pipeline | ✅ Done | 7 transforms, 400 → 1 400 images/class |
| CNN training (Google Colab) | ✅ Done | MobileNetV2, 50 epochs, T4 GPU |
| CNN accuracy | ✅ 100% | Level 1 (seed=999) and Level 2 (300 unseen images) |
| Behavioral Inference Engine | ✅ Done | `edge/bie.py` — 8 behavioral states |
| Edge pipeline | ✅ Done | `edge/main.py` — sim / demo / socket modes |
| WebSocket bridge | ✅ Done | `edge/ws_server.py` — BIE output → React dashboard |
| Radar dashboard | ✅ Done | Phosphor radar, live CNN data, JS fallback sim |
| Cloud pipeline (AWS) | 🔲 Stub | `edge/aws_publisher.py` — interface ready |
| Real RTL-SDR hardware | 🔲 Next | Edge code ready, hardware not yet connected |
| TFLite INT8 conversion | 🔲 Next | Needed for <50ms on Pi 5 |

---

## Signal Classes

The CNN classifies **3 signal types**. Designed to expand to more classes.

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
│               └────────────────────────────────────┬───────────────────┘            │
│                                              Interface C                             │
│                    ┌───────────────────┬────────────┴──────────────┐                │
│                    ▼                   ▼                            ▼                │
│          [Local Display]      [Alert Controller]          [AWS Publisher]            │
│          (terminal/Flask)    (LED · buzzer · sound)       (IoT Core stub)           │
│                                                                     │                │
│                                              ┌──────────────────────┘                │
│                                              ▼                                       │
│                                     [WsBroadcastServer]                              │
│                                     ws://localhost:8765                              │
└─────────────────────────────────────────────┬───────────────────────────────────────┘
                                              │ WebSocket (live CNN data)
                                   ┌──────────▼──────────┐
                                   │   React Dashboard    │
                                   │   Phosphor Radar     │
                                   │   Signal Cards       │
                                   │   Alert Log          │
                                   └─────────────────────┘

[ESP32S + Grove sensors] ──MQTT──► Pi 5  (temperature, humidity, motion, sound)
```

---

## Project Structure

```
SpectrumEye/
│
├── simulation/
│   └── simulation_final.py          # Physics-based IQ → spectrogram generator
│
├── ml/                              # CNN training pipeline
│   ├── augment.py                   # 7 augmentation transforms per image
│   ├── split_dataset.py             # Stratified train/val/test split (70/15/15)
│   ├── train.py                     # MobileNetV2 training (Keras/TF)
│   ├── evaluate.py                  # --quick / --folder / --image evaluation
│   ├── generate_test_batch.py       # Level 2 test set generator (seed=777)
│   ├── prepare_colab_zip.py         # Packages ml_training_v2.zip for Colab upload
│   ├── collect_synthetic.py         # Simple pixel-paint synthetic generator
│   ├── requirements.txt             # ML training dependencies (Python 3.12)
│   ├── setup_env.sh                 # pyenv + venv setup helper
│   ├── notebooks/
│   │   └── SpectrumEye_Training.ipynb   # 9-cell Colab training notebook
│   └── models/
│       └── v2_colab/
│           ├── config.json              # Training hyperparameters + accuracy
│           ├── history.json             # Per-epoch loss/accuracy
│           ├── classification_report.txt
│           ├── confusion_matrix.png
│           └── training_curves.png
│           # spectromeye_best.keras excluded from repo (large binary)
│           # retrain with ml/train.py or Colab notebook
│
├── edge/                            # Raspberry Pi 5 runtime
│   ├── main.py                      # Orchestration loop (--sim/--demo/--socket/--display ws)
│   ├── classifier.py                # CNN inference wrapper (Interface B)
│   ├── bie.py                       # Behavioral Inference Engine (Interface C)
│   ├── rssi_tracker.py              # Re-exports RSSITracker from bie.py
│   ├── ws_server.py                 # WebSocket broadcast server → React dashboard
│   ├── sensor_fusion.py             # ESP32 MQTT subscriber (Interface E)
│   ├── aws_publisher.py             # AWS IoT Core publisher stub (Interface D)
│   ├── alert_controller.py          # Threat alerts: terminal · GPIO · sound
│   └── local_display.py             # Terminal (ANSI) or Flask (HDMI) display
│
├── cloud/                           # AWS infrastructure (planned — Phase 6)
│   ├── cdk/                         # AWS CDK stack (not yet written)
│   ├── lambda/                      # Lambda functions (not yet written)
│   └── README.md
│
├── webapp/                          # React dashboard
│   ├── src/
│   │   ├── main.tsx                 # Entry point
│   │   ├── SpectrumEyeDashboard.jsx # Phosphor radar + signal cards
│   │   └── index.css                # Global styles + animations
│   ├── index.html
│   ├── vite.config.ts
│   └── package.json
│
├── requirements.txt                 # Edge + simulation dependencies (Python 3.12)
├── .gitignore
└── README.md
```

---

## Quick Start

### 1. React Dashboard (standalone — no hardware needed)

The dashboard runs with a built-in JavaScript simulation and switches to live CNN data automatically when the edge server is connected.

```bash
cd webapp
npm install
npm run dev
# → http://localhost:5173
```

The header badge shows **◌ SIMULATION** (amber) when running on JS data and **● LIVE · CNN** (green) when receiving real pipeline output.

---

### 2. Full End-to-End Pipeline (edge + dashboard)

**Step 1 — Install Python 3.12**

TensorFlow 2.18 does not support Python 3.13+.

```bash
# Arch Linux
sudo pacman -S python312
```

**Step 2 — Create the edge venv**

```bash
# From the project root
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**Step 3 — Run the pipeline**

```bash
# Terminal 1 — edge pipeline with WebSocket output (loops indefinitely)
source .venv/bin/activate
python edge/main.py --demo --display ws

# Terminal 2 — React dashboard
cd webapp && npm run dev
```

Open http://localhost:5173 — the dashboard connects automatically and switches to live CNN data.

**Stop with `Ctrl+C` in Terminal 1.**

#### Pipeline modes

| Flag | Behaviour |
|------|-----------|
| `--sim` | Random RSSI drift across 3 bands, CNN runs on synthetic spectrograms, runs forever |
| `--demo` | Scripted 4-phase scenario (see below), CNN bypassed with forced classes, loops forever |
| `--socket PATH` | Hardware mode — reads real frames from Unix socket written by DSP partner code |

#### `--demo` scenario (loops)

```
Phase 1 (15 frames, ~7s)   LTE background only
Phase 2 (20 frames, ~10s)  Walkie-talkie appears and approaches fast
Phase 3 (10 frames, ~5s)   Walkie-talkie holds, Key_Signal appears
Phase 4 (10 frames, ~5s)   Key_Signal departs
Pause  (10 frames, ~5s)    LTE cooldown
→ repeat
```

---

### 3. Evaluate the Trained Model

```bash
source .venv/bin/activate

# Level 1 — 30 fresh images, in-memory (fastest)
python ml/evaluate.py --quick

# Level 2 — 300 unseen images to disk
python ml/generate_test_batch.py   # → test_batch/ (seed=777)
python ml/evaluate.py --folder test_batch/

# Single image
python ml/evaluate.py --image path/to/spectrogram.png
```

---

### 4. Retrain the CNN (Google Colab)

```bash
# Step 1: Generate training dataset
python simulation/simulation_final.py
# → simulation/dataset_rf/  (1 500 raw PNGs)

# Step 2: Package for Colab
python ml/prepare_colab_zip.py
# → ml_training_v2.zip

# Step 3: Run ml/notebooks/SpectrumEye_Training.ipynb on Colab T4 GPU
# Step 4: Download best_model.keras → ml/models/production/spectromeye_best.keras
```

ML venv setup:

```bash
cd ml
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## Dataset

Generated with `simulation/simulation_final.py` — replicates the real RTL-SDR signal chain:

```
Random IQ signal → AWGN noise → STFT → dBFS normalization → 224×224 PNG
```

| Split | Per class | Total |
|-------|-----------|-------|
| Raw (generated) | 500 | 1 500 |
| Augmented | 1 400 | 4 200 |
| Train (70%) | 980 | 2 940 |
| Val (15%) | 210 | 630 |
| Test (15%) | 210 | 630 |

**Augmentation** (`ml/augment.py`): `time_shift` · `freq_shift` · `awgn` · `amplitude_scale` · `noise_mix` · `time_flip` + original

**Signal models:**
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
| Epochs | 50 |
| Test accuracy | **100%** (630 images, seed=777) |
| Inference time | 75.6 ms / frame (dev CPU) |
| Training platform | Google Colab T4 GPU |

---

## Behavioral Inference Engine

`edge/bie.py` translates a stream of CNN outputs into behavioral assessments.

| State | Meaning |
|-------|---------|
| `APPEARED` | Signal just became visible |
| `APPROACHING_SLOW` | RSSI rising slowly |
| `APPROACHING_FAST` | RSSI rising fast — high priority |
| `STATIONARY` | RSSI stable |
| `DEPARTING_SLOW` | RSSI falling slowly |
| `DEPARTING_FAST` | RSSI falling fast |
| `ERRATIC` | RSSI fluctuating unpredictably |
| `DISAPPEARED` | Signal lost |

---

## WebSocket Protocol

`edge/ws_server.py` broadcasts on `ws://localhost:8765`. Each message is JSON:

```json
{
  "threat_level": "CRITICAL",
  "threat_score": 9,
  "timestamp_ms": 1704067200000,
  "signals": [
    {
      "id":        "Key_Signal",
      "cls":       "Key_Signal",
      "state":     "APPROACHING_FAST",
      "rssi":      -48.0,
      "conf":      0.94,
      "bearing":   52,
      "trend":     4.2,
      "activeFor": 23
    }
  ],
  "alert": {
    "level":   "CRITICAL",
    "message": "Key fob approaching rapidly — possible remote device"
  }
}
```

The dashboard connects on load, auto-reconnects every 3 seconds on disconnect, and falls back to the JS simulation while the server is offline.

---

## Roadmap

| Phase | Description | Status |
|-------|-------------|--------|
| 1 | Environment & project setup | ✅ Complete |
| 2 | Dataset collection & augmentation | ✅ Complete |
| 3 | CNN training pipeline | ✅ Complete — 100% accuracy |
| 4 | Behavioral Inference Engine | ✅ Complete |
| 5 | Radar dashboard + live WebSocket pipeline | ✅ Complete |
| 6 | AWS cloud pipeline | 🔲 Stub ready — CDK to be written |
| 7 | Edge integration (Pi 5 + RTL-SDR) | 🔲 Code ready — hardware not yet connected |
| 8 | TFLite INT8 conversion | 🔲 Needed for <50ms on Pi 5 |
| 9 | Expand signal classes | 🔲 DJI OcuSync, FPV, WiFi, ADS-B |

---

## Tech Stack

**Machine Learning**
Python 3.12 · TensorFlow / Keras · NumPy · SciPy · Pillow · scikit-learn · Google Colab T4

**Edge Pipeline**
Python 3.12 · websockets · NumPy · SciPy · Pillow · Flask · psutil

**Dashboard**
React 19 · Vite 7 · Tailwind CSS v4 · Canvas API (phosphor radar)

**Cloud (planned)**
AWS IoT Core · Kinesis · Lambda · DynamoDB · API Gateway WebSocket

**Hardware**
Raspberry Pi 5 (8 GB) · RTL-SDR Blog V4 · Basys 3 FPGA (Artix-7) · ESP32S + Grove sensors

---

## Authors

Jorge Coronado - Memo - Alan
