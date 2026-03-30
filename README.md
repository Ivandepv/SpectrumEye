# SpectrumEye

**RF Situational Awareness System** — real-time radio frequency detection and classification using a CNN trained on real IQ captures, with a Behavioral Inference Engine that translates signal patterns into human-readable threat assessments.

Built on a Raspberry Pi 5 with an RTL-SDR Blog V4 dongle. Streams results live to a phosphor radar dashboard in the browser.

---

## What It Does

SpectrumEye sweeps 10 RF bands with an RTL-SDR dongle, converts each capture into a spectrogram image, classifies it with a MobileNetV2 CNN, and tracks behavioral state over time. Results stream to a React radar dashboard via WebSocket.

```
RTL-SDR Blog V4 (500 kHz – 1.75 GHz)
        ↓  IQ samples (2.048 MSPS, ~128 ms per band)
  224×224 grayscale spectrogram (matplotlib Agg — identical to training)
        ↓
  CNN — MobileNetV2 α=0.75  (97.45% accuracy, 10 classes, real RF data)
        ↓
  Behavioral Inference Engine  (RSSI tracking, 8 states, threat scoring)
        ↓
  WebSocket → React Phosphor Radar Dashboard
```

---

## Signal Classes

| Class | Frequency | Description |
|-------|-----------|-------------|
| `radio_fm` | 98 MHz | FM broadcast |
| `air_traffic` | 122 MHz | ATC voice (VHF air band) |
| `noaa` | 137.5 MHz | NOAA weather satellites |
| `local_repeaters` | 146 MHz | Amateur radio 2m band |
| `maritime` | 156.8 MHz | Maritime VHF Ch16 |
| `short_range_devices` | 315 MHz | Car keys, gate remotes |
| `wireless_controllers` | 433.92 MHz | RC controllers, IoT devices |
| `walkie_talkie` | 446 MHz | PMR446 handheld radios |
| `cellular_network` | 900 MHz | GSM / LTE / 5G |
| `aircraft_tracking` | 1090 MHz | ADS-B transponders |

---

## System Architecture

```
┌───────────────────────── EDGE (Raspberry Pi 5) ─────────────────────────┐
│                                                                           │
│  [RTL-SDR Blog V4] ──USB──► IQ Capture ──► 224×224 spectrogram (Agg)    │
│                              2.048 MSPS      NFFT=256, Hanning, 50% OVL  │
│                              10-band sweep                                │
│                                                   │                      │
│                                    ┌──────────────▼──────────────┐       │
│                                    │  SpectrumClassifier          │       │
│                                    │  MobileNetV2 (α=0.75)        │       │
│                                    │  97.45% · 10 classes         │       │
│                                    └──────────────┬──────────────┘       │
│                                                   │                      │
│                    ┌──────────────────────────────▼──────────────┐       │
│                    │       Behavioral Inference Engine (BIE)      │       │
│                    │  RSSI slope · EMA smoothing · 8 states       │       │
│                    │  Threat scoring (CLEAR/MODERATE/ELEVATED/    │       │
│                    │                  CRITICAL)                   │       │
│                    └──────────────────────────────┬──────────────┘       │
│                                                   │                      │
│                 ┌─────────────────────────────────┤                      │
│                 ▼                                 ▼                      │
│       [AlertController]              [WsBroadcastServer]                 │
│       terminal + GPIO/buzzer         ws://localhost:8765                 │
└─────────────────────────────────────────────────┬────────────────────────┘
                                                  │ WebSocket
                                       ┌──────────▼──────────┐
                                       │   React Dashboard    │
                                       │   Phosphor Radar     │
                                       │   Signal Cards       │
                                       │   Alert Log          │
                                       └─────────────────────┘
```

---

## Hardware

| Component | Model | Notes |
|-----------|-------|-------|
| SBC | Raspberry Pi 5 (8 GB) | Debian Trixie, Python 3.13 |
| SDR dongle | RTL-SDR Blog V4 | R828D tuner, 500 kHz – 1.75 GHz |

The RTL-SDR Blog V4 is a passive receive-only device. No RF output, no transmission.

---

## Quick Start

### Dev Machine (no hardware)

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Scripted demo scenario (CNN bypassed, good for presentations)
python edge/main.py --demo --display ws

# Dashboard (separate terminal)
cd webapp && npm install && npm run dev
# → http://localhost:5173
```

### Raspberry Pi 5 — One-Time Setup

```bash
# System packages
sudo apt install rtl-sdr nodejs npm

# Prevent the DVB kernel module from claiming the dongle
echo 'blacklist dvb_usb_rtl28xxu' | sudo tee /etc/modprobe.d/rtlsdr-blacklist.conf
echo 'blacklist dvb_usb_rtl2832u' | sudo tee -a /etc/modprobe.d/rtlsdr-blacklist.conf
sudo rmmod dvb_usb_rtl28xxu 2>/dev/null || true

# Python virtual environment
python3 -m venv .venv && source .venv/bin/activate
pip install --prefer-binary -r requirements.txt

# Copy the trained model (not in git — large binary)
# ml/models/production/best_model.keras   (~19 MB)
# ml/models/production/best_model.tflite  (~2 MB, optional but 4–8× faster)

# Web app
cd webapp && npm install
```

### Running the Live Pipeline

```bash
# Terminal 1 — edge pipeline
source .venv/bin/activate
python edge/main.py --hardware --display ws

# Terminal 2 — web dashboard
cd webapp && npm run dev -- --host
# Access from any device on the same network: http://<pi-ip>:5173
```

**Stop:** `Ctrl+C` — dongle is released cleanly.
**Unlock stuck dongle:** `pkill -f "edge/main.py"`

---

## Pipeline Modes

| Flag | Behaviour |
|------|-----------|
| `--hardware` | RTL-SDR Blog V4 — real-time 10-band sweep with full CNN inference |
| `--demo` | Scripted scenario — CNN bypassed, deterministic signal sequence |
| `--sim` | Random RSSI drift — no hardware, no CNN |
| `--socket PATH` | Reads frames from a Unix socket (DSP partner integration) |
| `--display ws` | Stream to React dashboard via WebSocket (default for Pi) |
| `--display terminal` | Print BIE output to console |
| `--display flask` | HTTP display backend |

---

## Dashboard Modes

The React dashboard (`webapp/`) has two operating modes that switch automatically based on whether the WebSocket pipeline is connected.

### ◌ SIMULATION (amber badge)

Active when the edge pipeline is **not running** or unreachable. The dashboard runs a fully self-contained JavaScript simulation with no hardware or Python required.

- Scripted signal scenario loops continuously: a key fob appears, approaches fast, goes stationary, then departs — mimicking a real detection lifecycle
- Background signals (FM broadcast, cellular) are always present with slow RSSI drift
- Periodic ADS-B aircraft flyovers and RC/drone controller appearances
- Signal positions on the radar are driven by simulated RSSI values (stronger = closer to center, weaker = outer rings)
- Bearings drift slowly to simulate movement
- Threat level, signal cards, and alert log all update in real time

This mode is useful to demonstrate the dashboard interface without any hardware. Open `http://localhost:5173` with no pipeline running and you will see it immediately.

<img width="1894" height="682" alt="image" src="https://github.com/user-attachments/assets/57fe5061-fa49-434d-bf62-2c86bb0fb7b7" />


### ● LIVE · CNN (green badge)

Active when the edge pipeline is running with `--display ws`. The dashboard connects to `ws://localhost:8765` and renders real data from the RTL-SDR dongle.

- Every signal card shows a real CNN classification (class + confidence %) from the MobileNetV2 model
- RSSI values are real measurements from the dongle (reported in **dBFS** — dB relative to full scale)
- Radar dot positions are derived from RSSI: the raw dBFS range (strong ≈ −5, weak ≈ −55) is normalized to the radar's display range so signals spread naturally across the rings
- Behavioral state (APPROACHING\_FAST, STATIONARY, etc.) and trend arrows come from the BIE's real RSSI slope computation
- Threat level is calculated by the BIE based on signal class and behavioral state
- The alert log shows real pipeline events (connections, threat transitions)

If the connection drops the dashboard automatically switches back to SIMULATION and retries every 3 seconds. When the pipeline reconnects the badge turns green and live data resumes immediately.

| | SIMULATION | LIVE · CNN |
|-|-----------|------------|
| Hardware needed | No | RTL-SDR + Pi 5 |
| CNN running | No (scripted) | Yes (MobileNetV2) |
| RSSI source | Simulated (dBm) | Real dongle (dBFS) |
| Bearing | Simulated drift | Simulated drift |
| Threat scoring | JS-derived | BIE (Python) |
| Badge color | Amber | Green |

---

<img width="1917" height="679" alt="image" src="https://github.com/user-attachments/assets/a1ff2cf8-1098-4622-ba73-6ac5d5eaa6c5" />


## Project Structure

```
SpectrumEye/
│
├── edge/
│   ├── main.py              # Pipeline orchestration (all modes)
│   ├── rtlsdr_source.py     # RTL-SDR Blog V4 frame source (pyrtlsdr)
│   ├── classifier.py        # CNN inference wrapper (Keras + TFLite)
│   ├── bie.py               # Behavioral Inference Engine (8 states)
│   ├── ws_server.py         # WebSocket broadcast → React dashboard
│   ├── alert_controller.py  # Threat alerts (terminal / GPIO)
│   └── local_display.py     # Terminal / Flask display backend
│
├── ml/
│   ├── train.py             # MobileNetV2 training (Google Colab / GPU)
│   ├── evaluate.py          # Model evaluation & metrics
│   ├── augment.py           # 7× data augmentation pipeline
│   ├── split_dataset.py     # train/val/test stratification
│   ├── convert_tflite.py    # INT8 TFLite conversion for Pi 5
│   ├── generate_test_batch.py
│   ├── notebooks/
│   │   └── SpectrumEye_Training.ipynb
│   ├── requirements.txt     # Training deps (TF 2.18, jupyter, sklearn)
│   └── models/production/
│       ├── best_model.keras   # v3_colab — 97.45%, 10 classes (git-ignored)
│       └── best_model.tflite  # INT8 quantized — ~60–120 ms on Pi 5
│
├── webapp/
│   └── src/
│       ├── SpectrumEyeDashboard.jsx  # Phosphor radar + signal cards
│       ├── main.tsx
│       └── index.css
│
├── simulation/
│   └── simulation_final.py  # Original physics-based IQ generator (archive)
│
├── requirements.txt         # Edge deps (Python 3.12 dev / 3.13 Pi)
└── README.md
```

---

## CNN Model

| Parameter | Value |
|-----------|-------|
| Architecture | MobileNetV2 (α = 0.75) |
| Input | 224 × 224 × 1 (grayscale spectrogram) |
| Output | 10 classes (softmax) |
| Training data | Real IQ captures from RTL-SDR Blog V4 |
| Accuracy | **97.45%** on held-out test set |
| Inference (Keras) | ~500 ms / frame on Pi 5 |
| Inference (TFLite INT8) | ~60–120 ms / frame on Pi 5 |

Generate the TFLite model from the Keras checkpoint (run on dev machine):

```bash
python ml/convert_tflite.py
# Output: ml/models/production/best_model.tflite
# Copy to Pi 5 — classifier.py auto-detects it
```

---

## Training the CNN — Full Pipeline

The model is trained on real RF spectrograms captured with the RTL-SDR dongle. Training runs on **Google Colab** (free T4 GPU). The pipeline is manual and follows these steps:

```
1. Collect real RF data (Pi 5 + RTL-SDR)  ──┐
                                             ├──► ml/dataset/raw/<class>/
2. Generate synthetic data (dev machine)  ──┘

3. Package for Colab
   ml/prepare_colab_zip.py  →  ml_training_v3.zip

4. Upload to Google Colab & train
   ml/notebooks/SpectrumEye_Training.ipynb
   augment.py        5 000 raw  →  35 000 augmented
   split_dataset.py             →  train / val / test
   train.py          50 epochs, early stopping, T4 GPU
   → download best_model.keras

5. Convert to TFLite (dev machine)
   ml/convert_tflite.py  →  best_model.tflite

6. Deploy to Pi 5
   scp best_model.keras best_model.tflite  pi5:~/SpectrumEye/ml/models/production/
```

### Step 1a — Collect Real RF Data (recommended)

Run on the **Raspberry Pi 5** with the RTL-SDR dongle connected. Point the antenna at the sky / outdoors for best signal variety.

```bash
source .venv/bin/activate

# Interactive mode — prompts for class and format
python edge/data_collector.py

# Non-interactive (recommended for scripting)
python edge/data_collector.py \
    --category walkie_talkie \
    --n-images 500 \
    --format grayscale \
    --output-dir ml/dataset/raw

# Repeat for all 10 classes:
for CLASS in radio_fm air_traffic noaa local_repeaters maritime \
             short_range_devices wireless_controllers walkie_talkie \
             cellular_network aircraft_tracking; do
    python edge/data_collector.py \
        --category $CLASS \
        --n-images 500 \
        --format grayscale \
        --output-dir ml/dataset/raw
done
```

Images are saved as `ml/dataset/raw/<class>/sample_NNN.png` (224×224 grayscale PNG). Aim for **at least 200 images per class**; 500 is recommended. The script appends to existing captures so you can run it in multiple sessions.

### Step 1b — Generate Synthetic Data (optional / bootstrap)

If hardware is not available, generate physics-based synthetic spectrograms for 5 signal classes (Key_Signal, Walkie_Talkie, LTE, ADS_B, DJI_Drone):

```bash
# Run on any machine (no hardware needed)
python ml/collect_synthetic.py --n 500
# → ml/dataset/raw/<class>/ (5 classes × 500 images)
```

Synthetic data is useful to bootstrap training before real captures are available, but the model's accuracy on real hardware will be lower than with a fully real dataset.

### Step 2 — Package for Google Colab

Run on the **dev machine** from the project root. This bundles the raw dataset, augmentation scripts, and training script into a single zip ready to upload to Colab.

```bash
python ml/prepare_colab_zip.py
# → ml_training_v3.zip  (~50–200 MB depending on dataset size)
```

The zip contains:
```
ml/
├── dataset/raw/<class>/*.png   (all collected spectrograms)
├── augment.py
├── split_dataset.py
└── train.py
```

### Step 3 — Train on Google Colab

1. Open `ml/notebooks/SpectrumEye_Training.ipynb` in Google Colab
2. Select **Runtime → Change runtime type → T4 GPU**
3. Run Cell 1 (install deps) and Cell 2 (mount Drive or upload)
4. Upload `ml_training_v3.zip` when prompted (Cell 3)
5. Run all remaining cells — the notebook will:
   - Unzip and validate the dataset
   - Run `augment.py` (7× augmentation → ~35,000 images)
   - Run `split_dataset.py` (70% train / 15% val / 15% test)
   - Run `train.py` (MobileNetV2, 50 epochs, early stopping)
   - Print accuracy metrics and save `best_model.keras`
6. Download `best_model.keras` from the Colab file browser

Expected training time: **~15–30 minutes** on a T4 GPU.

### Step 4 — Convert to TFLite and Deploy

```bash
# On dev machine — convert Keras model to INT8 TFLite
cp ~/Downloads/best_model.keras ml/models/production/best_model.keras
python ml/convert_tflite.py
# → ml/models/production/best_model.tflite

# Copy both to Pi 5
scp ml/models/production/best_model.keras \
    ml/models/production/best_model.tflite \
    porphyras@192.168.0.156:~/Desktop/SpectrumEye/ml/models/production/
```

`edge/classifier.py` auto-detects the `.tflite` file and uses it instead of the `.keras` model — no code changes needed.

---

## Behavioral Inference Engine

`edge/bie.py` converts raw CNN + RSSI streams into behavioral assessments using EMA-smoothed RSSI and linear regression over a 20-sample sliding window.

| State | Trigger |
|-------|---------|
| `APPEARED` | First detection (< 5 samples) |
| `APPROACHING_SLOW` | RSSI slope > +0.5 dBFS/s |
| `APPROACHING_FAST` | RSSI slope > +2.0 dBFS/s |
| `STATIONARY` | \|slope\| ≤ 0.5 dBFS/s |
| `DEPARTING_SLOW` | RSSI slope < −0.5 dBFS/s |
| `DEPARTING_FAST` | RSSI slope < −2.0 dBFS/s |
| `ERRATIC` | High variance in consecutive RSSI differences |
| `DISAPPEARED` | No update for 10 seconds |

Threat levels: **CLEAR → MODERATE → ELEVATED → CRITICAL**, scored per signal class and behavioral state. Walkie-talkie and RC controllers score highest; FM/cellular score zero.

---

## WebSocket Protocol

`ws://localhost:8765` — JSON pushed after each sweep frame:

```json
{
  "threat_level": "ELEVATED",
  "threat_score": 6,
  "timestamp_ms": 1704067200000,
  "signals": [
    {
      "id":        "walkie_talkie",
      "cls":       "walkie_talkie",
      "state":     "APPROACHING_FAST",
      "rssi":      -52.0,
      "conf":      0.91,
      "bearing":   315,
      "trend":     3.2,
      "activeFor": 12
    }
  ]
}
```

The dashboard auto-reconnects every 3 seconds and falls back to a scripted JS simulation when the pipeline is offline.

---

## Tech Stack

**Edge** — Python 3.13 · pyrtlsdr 0.2.92 · TensorFlow 2.21 · websockets · NumPy · matplotlib

**Dashboard** — React 19 · Vite 7 · Tailwind CSS v4 · Canvas API

**Hardware** — Raspberry Pi 5 (8 GB) · RTL-SDR Blog V4 (R828D, 500 kHz – 1.75 GHz)

---

## Authors

Alan Romo · Guillermo Portillo · Jorge Coronado
