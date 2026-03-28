# SpectrumEye

**RF Situational Awareness System** — real-time detection and classification of radio frequency signals using a CNN trained on real IQ captures, and a Behavioral Inference Engine that translates signal patterns into human-readable threat assessments.

Built in Tainan, Taiwan. Current state: **fully working end-to-end on Raspberry Pi 5** — RTL-SDR Blog V4 → CNN → BIE → WebSocket → React radar dashboard, confirmed live.

---

## What It Does

SpectrumEye sweeps 10 RF bands with an RTL-SDR dongle, converts IQ samples into spectrograms, and classifies them with a CNN. A Behavioral Inference Engine (BIE) tracks signal behavior over time and generates plain-language threat assessments. Results stream live to a phosphor radar dashboard via WebSocket.

```
RTL-SDR Blog V4 (500 kHz – 1.75 GHz)
        ↓  IQ samples (real RF, 2.048 MSPS)
  224×224 spectrogram (matplotlib Agg, identical to training)
        ↓
  CNN — MobileNetV2 (97.45% accuracy, 10 classes, real RF data)
        ↓
  Behavioral Inference Engine (RSSI tracking, 8 states)
        ↓
  WebSocket → React Phosphor Radar Dashboard
```

---

## Current State

| Component | Status | Notes |
|-----------|--------|-------|
| RTL-SDR Blog V4 hardware | ✅ Working | Pi 5, real-time 10-band sweep |
| CNN model (v3_colab) | ✅ Working | 97.45% accuracy, 10 classes, real RF data |
| Edge pipeline | ✅ Working | `--hardware`, `--demo`, `--sim` modes |
| WebSocket bridge | ✅ Working | BIE output → React dashboard |
| Radar dashboard | ✅ Working | Live CNN badge, JS fallback when offline |
| Graceful shutdown | ✅ Fixed | Ctrl+C properly closes SDR dongle |
| Cloud pipeline (AWS) | 🔲 Stub | Interface ready, CDK not written |
| TFLite INT8 conversion | 🔲 Next | Target <50ms inference on Pi 5 |
| Expand signal classes | 🔲 Next | DJI OcuSync, FPV, WiFi, ADS-B |

---

## Signal Classes (10 real RF bands)

| Class | Center Frequency | Description |
|-------|-----------------|-------------|
| `radio_fm` | 98 MHz | FM broadcast |
| `air_traffic` | 122 MHz | Aerial control tower |
| `noaa` | 137.5 MHz | Meteorological satellites |
| `local_repeaters` | 146 MHz | Amateur radio (2m band) |
| `maritime` | 156.8 MHz | Maritime VHF Ch16 |
| `short_range_devices` | 315 MHz | Car keys, gates, doorbells |
| `wireless_controllers` | 433.92 MHz | ISM short-range devices |
| `walkie_talkie` | 446 MHz | PMR446 handheld radios |
| `cellular_network` | 900 MHz | GSM/LTE/5G |
| `aircraft_tracking` | 1090 MHz | ADS-B commercial aircraft |

---

## System Architecture

```
┌─────────────────────────── EDGE (Raspberry Pi 5) ───────────────────────────┐
│                                                                               │
│  [RTL-SDR Blog V4] ──USB──► IQ Capture ──► 224×224 spectrogram (Agg)        │
│                              2.048 MSPS      NFFT=256, Hanning, 50% OVL     │
│                              10-band sweep                                    │
│                                                   │                          │
│                                    ┌──────────────▼─────────────┐            │
│                                    │  SpectrumClassifier         │            │
│                                    │  MobileNetV2 (α=0.75)      │            │
│                                    │  97.45% · 10 classes        │            │
│                                    └──────────────┬─────────────┘            │
│                                                   │                          │
│                    ┌──────────────────────────────▼──────────────────┐       │
│                    │         Behavioral Inference Engine (BIE)        │       │
│                    │  RSSI tracking · trend analysis · 8 states       │       │
│                    └──────────────────────────────┬──────────────────┘       │
│                                                   │                          │
│              ┌────────────────┬───────────────────┤                          │
│              ▼                ▼                   ▼                          │
│    [Local Display]  [Alert Controller]   [WsBroadcastServer]                 │
│    (terminal/Flask)  (LED/sound stub)    ws://localhost:8765                 │
└──────────────────────────────────────────────────┬──────────────────────────┘
                                                   │ WebSocket
                                        ┌──────────▼──────────┐
                                        │   React Dashboard    │
                                        │   Phosphor Radar     │
                                        │   Signal Cards       │
                                        │   Alert Log          │
                                        └─────────────────────┘
```

---

## Quick Start

### Dev Machine (Arch Linux, Python 3.12)

```bash
# Edge pipeline — simulation mode
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python edge/main.py --demo --display ws

# Dashboard (separate terminal)
cd webapp && npm install && npm run dev
# → http://localhost:5173
```

---

### Raspberry Pi 5 (Debian Trixie, Python 3.13)

**One-time setup:**

```bash
# System packages
sudo apt install rtl-sdr

# Blacklist DVB kernel module (prevents it from claiming the dongle)
echo 'blacklist dvb_usb_rtl28xxu' | sudo tee /etc/modprobe.d/rtlsdr-blacklist.conf
echo 'blacklist dvb_usb_rtl2832u' | sudo tee -a /etc/modprobe.d/rtlsdr-blacklist.conf

# Python venv
cd ~/Desktop/SpectrumEye
python3 -m venv .venv && source .venv/bin/activate
pip install --prefer-binary -r requirements.txt

# Node.js for webapp
sudo apt install nodejs npm
cd webapp && npm install
```

**Deploy model:**

```bash
# Copy best_model.keras to ml/models/production/ (excluded from git — large binary)
# File: ml/models/production/best_model.keras
```

**Run:**

```bash
# Terminal 1 — edge pipeline
source .venv/bin/activate
python edge/main.py --hardware --display ws

# Terminal 2 — webapp
cd webapp && npm run dev -- --host
# → http://<pi-ip>:5173 from any device on the same network
```

**Stop:** `Ctrl+C` — dongle is released cleanly.

**If dongle locks:** `pkill -f "edge/main.py"`

---

### Pipeline Modes

| Flag | Behaviour |
|------|-----------|
| `--hardware` | Real RTL-SDR Blog V4 — sweeps 10 bands, full CNN inference |
| `--demo` | Scripted scenario — CNN bypassed, good for presentations |
| `--sim` | Random RSSI drift — no hardware, no CNN |
| `--display ws` | Stream BIE output to React dashboard via WebSocket (default) |
| `--display terminal` | Print to console |

---

## Project Structure

```
SpectrumEye/
├── edge/
│   ├── main.py                  # Pipeline orchestration (--hardware/--demo/--sim)
│   ├── rtlsdr_source.py         # RTL-SDR Blog V4 frame source (pyrtlsdr)
│   ├── classifier.py            # CNN inference wrapper
│   ├── bie.py                   # Behavioral Inference Engine (8 states)
│   ├── ws_server.py             # WebSocket broadcast → React dashboard
│   ├── data_collector.py        # Training data collection tool
│   ├── alert_controller.py      # Threat alerts (terminal / GPIO)
│   ├── local_display.py         # Terminal / Flask display backend
│   ├── sensor_fusion.py         # ESP32 MQTT subscriber
│   └── aws_publisher.py         # AWS IoT Core stub
│
├── ml/
│   ├── train.py                 # MobileNetV2 training
│   ├── evaluate.py              # Model evaluation
│   ├── notebooks/
│   │   └── SpectrumEye_Training.ipynb
│   └── models/production/
│       └── best_model.keras     # v3_colab — 97.45%, 10 classes (git-ignored)
│
├── webapp/
│   └── src/
│       └── SpectrumEyeDashboard.jsx   # Phosphor radar + signal cards
│
├── simulation/
│   └── simulation_final.py      # Physics-based IQ → spectrogram generator
│
├── requirements.txt             # Edge deps (Python 3.12 dev / 3.13 Pi)
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
| Accuracy | **97.45%** |
| Model file | `ml/models/production/best_model.keras` |

---

## Behavioral Inference Engine

`edge/bie.py` translates CNN output streams into behavioral assessments.

| State | Meaning |
|-------|---------|
| `APPEARED` | Signal just became visible |
| `APPROACHING_SLOW` | RSSI rising slowly |
| `APPROACHING_FAST` | RSSI rising fast — elevated priority |
| `STATIONARY` | RSSI stable |
| `DEPARTING_SLOW` | RSSI falling slowly |
| `DEPARTING_FAST` | RSSI falling fast |
| `ERRATIC` | RSSI fluctuating unpredictably |
| `DISAPPEARED` | Signal lost |

---

## WebSocket Protocol

`ws://localhost:8765` — JSON messages from BIE:

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

Dashboard auto-reconnects every 3 seconds and falls back to JS simulation when offline.

---

## Roadmap

| Phase | Description | Status |
|-------|-------------|--------|
| 1–5 | Dataset, CNN, BIE, dashboard, WebSocket pipeline | ✅ Complete |
| 6 | Pi 5 + RTL-SDR Blog V4 real-time integration | ✅ Complete |
| 7 | TFLite INT8 conversion (<50ms inference on Pi 5) | 🔲 Next |
| 8 | AWS CDK cloud pipeline | 🔲 Stub ready |
| 9 | Expand signal classes (DJI OcuSync, FPV, WiFi) | 🔲 Planned |

---

## Tech Stack

**Edge** — Python 3.13 · pyrtlsdr 0.2.92 · TensorFlow 2.21 · websockets · NumPy · matplotlib

**Dashboard** — React 19 · Vite 7 · Tailwind CSS v4 · Canvas API

**Hardware** — Raspberry Pi 5 (8 GB) · RTL-SDR Blog V4 (R828D, 500 kHz – 1.75 GHz)

**Cloud (planned)** — AWS IoT Core · Kinesis · Lambda · DynamoDB

---

## Authors

Jorge Coronado · Memo · Alan
