"""
edge/main.py — SpectrumEye Edge Main Loop

Orchestrates the full edge pipeline on Raspberry Pi 5:

    sweep_frame → CNN Classifier → BIE → Local Display
                                       → AWS Publisher
                                       → Alert Controller

Modes:
  --sim     Simulation mode: generates synthetic sweep_frames internally.
            No hardware or DSP partner code needed. Good for development.

  --demo    Demo mode: runs a scripted 3-signal scenario (Key_Signal
            approaches, Walkie_Talkie appears and is CRITICAL, LTE stays).
            Pauses between phases so you can observe the dashboard.

  --socket  Hardware mode: reads sweep_frames from a Unix domain socket
            written by the partner DSP code. Default socket path:
            /tmp/spectromeye_frames.sock

  --display flask  Use Flask HTTP display instead of terminal.
  --display ws     Push BIE output to the React dashboard via WebSocket.
  --port    5000   Flask port (default 5000).
  --ws-port 8765   WebSocket port (default 8765, only used with --display ws).

Data flow per frame:
  1. Receive sweep_frame dict (Interface A + A+)
  2. Run CNN inference → Interface B classification dict
  3. Run BIE → Interface C BIE output dict
  4. Attach env_context from SensorFusion
  5. Send to LocalDisplay, AWSPublisher, AlertController

Usage:
  # Development / no hardware:
  python edge/main.py --sim
  python edge/main.py --demo

  # Real hardware (DSP partner writes frames to socket):
  python edge/main.py --socket /tmp/spectromeye_frames.sock
"""

import sys
import json
import time
import signal
import socket
import logging
import argparse
import threading
from pathlib import Path
from typing import Optional, Iterator

import numpy as np

# Add project root to path so edge.* imports work
sys.path.insert(0, str(Path(__file__).parent.parent))

from edge.classifier       import SpectrumClassifier
from edge.bie              import BIE
from edge.sensor_fusion    import SensorFusion
from edge.aws_publisher    import AWSPublisher
from edge.alert_controller import AlertController
from edge.local_display    import LocalDisplay
from edge.ws_server        import WsBroadcastServer

# ─── LOGGING ──────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(name)-20s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("main")

# ─── CONFIGURATION ────────────────────────────────────────────────

SENSOR_ID      = "spectromeye-pi5-001"
DEFAULT_SOCKET = "/tmp/spectromeye_frames.sock"

# Simulation parameters
SIM_FRAME_INTERVAL = 0.5    # seconds between synthetic frames
SIM_BANDS = [               # swept bands for simulation
    {"band_id": "iot_433",  "center_freq_hz": 433_000_000,   "rssi_dbfs": -78.0},
    {"band_id": "lte_band3","center_freq_hz": 1_800_000_000, "rssi_dbfs": -75.0},
    {"band_id": "wifi_24",  "center_freq_hz": 2_437_000_000, "rssi_dbfs": -80.0},
]

# Status heartbeat every N frames
HEARTBEAT_EVERY = 60


# ─── SYNTHETIC FRAME GENERATOR ────────────────────────────────────

class SimulationFrameSource:
    """
    Generates synthetic sweep_frames for development / demo mode.

    In --sim mode:  cycles through bands with slow random RSSI drift.
    In --demo mode: runs a scripted multi-signal scenario.
    """

    # Class → typical center_freq_hz used in simulation
    _CLASS_FREQ = {
        "Key_Signal":   433_000_000,
        "Walkie_Talkie":462_000_000,
        "LTE":        1_800_000_000,
    }

    def __init__(self, mode: str = "sim", interval: float = SIM_FRAME_INTERVAL) -> None:
        self._mode     = mode
        self._interval = interval
        self._frame_id = 0
        self._rng      = np.random.default_rng(seed=42)

    def frames(self) -> Iterator[dict]:
        """Yield synthetic sweep_frames indefinitely (or until demo ends)."""
        if self._mode == "demo":
            yield from self._demo_sequence()
        else:
            yield from self._random_sim()

    # ── Random simulation ─────────────────────────────────────────

    def _random_sim(self) -> Iterator[dict]:
        """
        Cycle through SIM_BANDS.
        Each band has slow RSSI drift and a synthetic 224×224 spectrogram.
        """
        rssi_state = {b["band_id"]: b["rssi_dbfs"] for b in SIM_BANDS}

        while True:
            for band in SIM_BANDS:
                bid   = band["band_id"]
                freq  = band["center_freq_hz"]

                # Slow random walk on RSSI
                rssi_state[bid] += float(self._rng.uniform(-0.3, 0.3))
                rssi_state[bid]  = float(np.clip(rssi_state[bid], -95.0, -40.0))

                yield self._make_frame(
                    freq=freq,
                    rssi_dbfs=rssi_state[bid],
                    band_id=bid,
                )
                time.sleep(self._interval)

    # ── Demo scenario ─────────────────────────────────────────────

    def _demo_sequence(self) -> Iterator[dict]:
        """
        Scripted 3-phase demo:
          Phase 1 (15 frames): LTE background only
          Phase 2 (20 frames): Walkie_Talkie appears and approaches fast
          Phase 3 (10 frames): Walkie_Talkie stabilises then Key_Signal appears
          Phase 4 (10 frames): Key_Signal departs, Walkie_Talkie holds
        """
        log.info("Demo: Phase 1 — LTE background only")
        for i in range(15):
            yield self._make_frame(
                freq=1_800_000_000, rssi_dbfs=-75.0, band_id="lte_band3",
                force_class="LTE", force_conf=0.99,
            )
            time.sleep(self._interval)

        log.info("Demo: Phase 2 — Walkie-talkie approaching fast")
        for i in range(20):
            rssi = -85.0 + i * 2.0   # +4 dBFS/s at 0.5s intervals → slope ≈ +4
            yield self._make_frame(
                freq=462_000_000, rssi_dbfs=rssi, band_id="wifi_24",
                force_class="Walkie_Talkie", force_conf=0.93,
            )
            time.sleep(self._interval)

        log.info("Demo: Phase 3 — Key_Signal appears, Walkie_Talkie stationary")
        for i in range(10):
            yield self._make_frame(
                freq=462_000_000, rssi_dbfs=-45.0, band_id="wifi_24",
                force_class="Walkie_Talkie", force_conf=0.91,
            )
            time.sleep(self._interval / 2)
            yield self._make_frame(
                freq=433_000_000, rssi_dbfs=-65.0 + i * 0.5, band_id="iot_433",
                force_class="Key_Signal", force_conf=0.88,
            )
            time.sleep(self._interval / 2)

        log.info("Demo: Phase 4 — Key_Signal departing")
        for i in range(10):
            rssi = -65.0 - i * 1.5
            yield self._make_frame(
                freq=433_000_000, rssi_dbfs=rssi, band_id="iot_433",
                force_class="Key_Signal", force_conf=0.87,
            )
            time.sleep(self._interval)

        log.info("Demo scenario complete.")

    # ── Synthetic spectrogram builder ─────────────────────────────

    def _make_frame(
        self,
        freq:        int,
        rssi_dbfs:   float,
        band_id:     str,
        force_class: Optional[str] = None,
        force_conf:  Optional[float] = None,
    ) -> dict:
        """
        Build a sweep_frame dict with a synthetic 224×224 spectrogram.

        The spectrogram is noise-only — the CNN will classify based on visual
        patterns from the actual model, not forced output. Use force_class only
        when you want to bypass the CNN entirely (e.g. demo scripting).

        Interface A + A+ sweep_frame format.
        """
        self._frame_id += 1
        ts = int(time.time() * 1000)

        # Minimal noise spectrogram (CNN will classify it; don't expect perfect results)
        # For demo accuracy, we inject _force_class as a metadata override, not a real spectrogram.
        spectrogram = self._make_spectrogram(force_class)

        frame = {
            "frame_id":      self._frame_id,
            "timestamp_ms":  ts,
            "center_freq_hz": freq,
            "sample_rate_hz": 2_048_000,
            "gain_db":        30.0,
            "spectrogram":    spectrogram,
            "rssi": {
                "band_id":       band_id,
                "center_freq_hz": freq,
                "bandwidth_hz":  2_048_000,
                "rssi_dbfs":     round(rssi_dbfs, 1),
                "peak_dbfs":     round(rssi_dbfs + 2.0, 1),
                "occupied":      rssi_dbfs > -90.0,
            },
            # Optional override for demo mode — bypasses CNN inference
            "_sim_class": force_class,
            "_sim_conf":  force_conf,
        }
        return frame

    def _make_spectrogram(self, signal_class: Optional[str]) -> np.ndarray:
        """
        Generate a 224×224 uint8 spectrogram that resembles the given class.
        Used only in simulation — not as perfect training-quality images.
        """
        img = np.full((224, 224), 8, dtype=np.uint8)   # noise floor

        if signal_class == "Key_Signal":
            # Short narrow bursts
            for _ in range(self._rng.integers(2, 5)):
                col  = int(self._rng.integers(20, 200))
                row  = int(self._rng.integers(80, 140))
                width = int(self._rng.integers(1, 3))
                height= int(self._rng.integers(5, 15))
                img[row:row+height, col:col+width] = 220

        elif signal_class == "Walkie_Talkie":
            # Narrow continuous vertical stripe with FM wobble
            row0  = int(self._rng.integers(90, 120))
            width = int(self._rng.integers(3, 6))
            for col in range(224):
                wobble = int(self._rng.integers(-2, 3))
                r = max(0, min(223, row0 + wobble))
                img[r:r+width, col] = 200

        elif signal_class == "LTE":
            # Wide flat-top block
            row0 = 62
            img[row0:row0+100, :] = 160

        # Add noise
        noise = self._rng.integers(0, 20, (224, 224), dtype=np.uint8)
        img   = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        return img


# ─── SOCKET FRAME SOURCE ──────────────────────────────────────────

class SocketFrameSource:
    """
    Reads sweep_frames from a Unix domain socket.

    Partner DSP code writes JSON-encoded frames (one per line).
    The spectrogram is transmitted as a base64-encoded PNG or flat uint8 list.

    Protocol:
        Each message is a newline-terminated JSON object.
        The "spectrogram" field is a flat list of 224*224 uint8 integers
        (row-major, shape 224×224).
    """

    def __init__(self, socket_path: str = DEFAULT_SOCKET) -> None:
        self._path = socket_path

    def frames(self) -> Iterator[dict]:
        """Block until the socket is available, then yield frames."""
        import base64

        log.info("SocketFrameSource: waiting for socket %s", self._path)
        while not Path(self._path).exists():
            time.sleep(0.5)

        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
            sock.connect(self._path)
            log.info("SocketFrameSource: connected")
            buf = b""

            while True:
                chunk = sock.recv(65536)
                if not chunk:
                    log.warning("SocketFrameSource: connection closed")
                    break
                buf += chunk

                while b"\n" in buf:
                    line, buf = buf.split(b"\n", 1)
                    try:
                        frame = json.loads(line.decode())
                        # Decode spectrogram
                        raw = frame["spectrogram"]
                        if isinstance(raw, str):
                            # base64 PNG
                            import io
                            from PIL import Image
                            data   = base64.b64decode(raw)
                            img    = Image.open(io.BytesIO(data)).convert("L")
                            arr    = np.array(img, dtype=np.uint8)
                        else:
                            # flat uint8 list
                            arr = np.array(raw, dtype=np.uint8).reshape(224, 224)
                        frame["spectrogram"] = arr
                        yield frame
                    except Exception as exc:
                        log.warning("SocketFrameSource: bad frame: %s", exc)


# ─── PIPELINE ─────────────────────────────────────────────────────

class EdgePipeline:
    """
    Wires together all edge components and runs the main processing loop.
    """

    def __init__(
        self,
        frame_source,
        model_path:      Optional[Path] = None,
        display_backend: str = "terminal",
        flask_port:      int = 5000,
        enable_aws:      bool = False,
        ws_port:         int = 8765,
    ) -> None:
        self._source = frame_source

        log.info("EdgePipeline: loading CNN classifier...")
        self._clf     = SpectrumClassifier(model_path=model_path) if model_path else SpectrumClassifier()
        self._bie     = BIE(sensor_id=SENSOR_ID)
        self._sensors = SensorFusion()
        self._alerts  = AlertController(sensor_id=SENSOR_ID)
        self._aws     = AWSPublisher(sensor_id=SENSOR_ID)

        # WebSocket broadcast server (started only with --display ws)
        self._ws: Optional[WsBroadcastServer] = None
        if display_backend == "ws":
            self._ws = WsBroadcastServer(host="localhost", port=ws_port)
            self._display = LocalDisplay(backend="terminal")  # keep terminal log too
        else:
            self._display = LocalDisplay(backend=display_backend, flask_port=flask_port)

        self._running      = False
        self._frame_count  = 0
        self._start_time   = 0.0

    def run(self) -> None:
        """Start the pipeline and block until interrupted."""
        self._running    = True
        self._start_time = time.monotonic()

        # Graceful shutdown on Ctrl-C / SIGTERM
        signal.signal(signal.SIGINT,  self._handle_shutdown)
        signal.signal(signal.SIGTERM, self._handle_shutdown)

        self._display.start()
        if self._ws:
            self._ws.start()
        log.info("EdgePipeline: running — Ctrl+C to stop")

        try:
            for frame in self._source.frames():
                if not self._running:
                    break
                self._process_frame(frame)
        except StopIteration:
            pass
        except Exception as exc:
            log.exception("EdgePipeline: fatal error: %s", exc)
        finally:
            self._shutdown()

    def _process_frame(self, frame: dict) -> None:
        """Process one sweep_frame through the full pipeline."""
        self._frame_count += 1

        spectrogram   = frame["spectrogram"]          # (224, 224) uint8
        frame_id      = frame["frame_id"]
        timestamp_ms  = frame["timestamp_ms"]
        center_freq   = frame["center_freq_hz"]
        rssi_dbfs     = frame["rssi"]["rssi_dbfs"]

        # ── Step 1: CNN Inference ─────────────────────────────────
        # In demo/sim mode, _sim_class overrides CNN output for scripted scenarios
        sim_class = frame.get("_sim_class")
        sim_conf  = frame.get("_sim_conf")

        if sim_class is not None and sim_conf is not None:
            # Demo mode: skip CNN, use scripted class
            classification = {
                "frame_id":          frame_id,
                "timestamp_ms":      timestamp_ms,
                "center_freq_hz":    center_freq,
                "predicted_class":   sim_class,
                "confidence":        sim_conf,
                "confidence_level":  "HIGH" if sim_conf >= 0.85 else "MEDIUM",
                "all_probabilities": {},
                "inference_time_ms": 0.0,
                "model_version":     "sim",
            }
        else:
            classification = self._clf.classify(
                spectrogram=spectrogram,
                frame_id=frame_id,
                timestamp_ms=timestamp_ms,
                center_freq_hz=center_freq,
            )

        # ── Step 2: BIE ───────────────────────────────────────────
        bie_output = self._bie.process(
            signal_class=classification["predicted_class"],
            confidence=classification["confidence"],
            rssi_dbfs=rssi_dbfs,
            center_freq_hz=center_freq,
            timestamp_ms=timestamp_ms,
            frame_id=frame_id,
        )

        # ── Step 3: Attach env context ────────────────────────────
        bie_output["env_context"] = self._sensors.get_env_context()

        # ── Step 4: Output ────────────────────────────────────────
        self._display.update(bie_output)
        self._alerts.evaluate(bie_output)
        self._aws.publish_detection(bie_output)
        if self._ws:
            self._ws.send(bie_output)

        # RSSI time-series data point
        self._aws.publish_rssi(
            signal_class=bie_output["signal_class"],
            rssi_dbfs=rssi_dbfs,
            confidence=classification["confidence"],
            threat_score=bie_output["threat_score"],
            timestamp_ms=timestamp_ms,
        )

        # Periodic status heartbeat
        if self._frame_count % HEARTBEAT_EVERY == 0:
            self._publish_status()

    def _publish_status(self) -> None:
        import psutil  # optional — skip if not installed
        uptime = int(time.monotonic() - self._start_time)

        status: dict = {
            "sensor_id":          SENSOR_ID,
            "timestamp_ms":       int(time.time() * 1000),
            "status":             "online",
            "uptime_sec":         uptime,
            "frames_processed":   self._frame_count,
            "cnn_model_version":  self._clf.model_version,
            "signals_tracked":    len(self._bie.get_all_states()),
        }

        try:
            status["cpu_temp_c"]    = psutil.sensors_temperatures()["cpu_thermal"][0].current
            status["cpu_usage_pct"] = psutil.cpu_percent()
            status["ram_usage_pct"] = psutil.virtual_memory().percent
        except Exception:
            pass   # psutil not available or no CPU temp sensor

        self._aws.publish_status(status)

    def _handle_shutdown(self, signum, frame) -> None:
        log.info("EdgePipeline: shutdown signal received")
        self._running = False

    def _shutdown(self) -> None:
        log.info("EdgePipeline: shutting down...")
        self._display.stop()
        if self._ws:
            self._ws.stop()
        self._sensors.close()
        self._alerts.close()
        self._aws.close()
        uptime = int(time.monotonic() - self._start_time)
        log.info(
            "EdgePipeline: stopped after %d frames in %d seconds",
            self._frame_count, uptime,
        )


# ─── ENTRY POINT ──────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="SpectrumEye Edge — main processing loop",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument(
        "--sim",
        action="store_true",
        help="Simulation mode: synthetic sweep_frames, no hardware needed",
    )
    mode_group.add_argument(
        "--demo",
        action="store_true",
        help="Demo mode: scripted 3-signal scenario (good for presentations)",
    )
    mode_group.add_argument(
        "--socket",
        metavar="PATH",
        default=None,
        help=f"Hardware mode: Unix socket path written by DSP partner code\n"
             f"(default: {DEFAULT_SOCKET})",
    )

    parser.add_argument(
        "--model",
        metavar="PATH",
        default=None,
        help="Path to .keras model file (default: ml/models/production/spectromeye_best.keras)",
    )
    parser.add_argument(
        "--display",
        choices=["terminal", "flask", "ws"],
        default="terminal",
        help="Display backend: terminal (default) | flask | ws (React dashboard)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=5000,
        help="Flask display port (default: 5000, only used with --display flask)",
    )
    parser.add_argument(
        "--ws-port",
        type=int,
        default=8765,
        help="WebSocket port for the React dashboard (default: 8765)",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=SIM_FRAME_INTERVAL,
        help=f"Seconds between synthetic frames in --sim/--demo mode (default: {SIM_FRAME_INTERVAL})",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable DEBUG-level logging",
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Select frame source
    if args.sim:
        source = SimulationFrameSource(mode="sim", interval=args.interval)
        log.info("Mode: simulation (synthetic frames, %.1fs interval)", args.interval)
    elif args.demo:
        source = SimulationFrameSource(mode="demo", interval=args.interval)
        log.info("Mode: demo (scripted scenario)")
    else:
        socket_path = args.socket if args.socket != "None" else DEFAULT_SOCKET
        source = SocketFrameSource(socket_path=socket_path)
        log.info("Mode: hardware (socket: %s)", socket_path)

    # Select model path
    model_path = Path(args.model) if args.model else None

    # Build and run pipeline
    pipeline = EdgePipeline(
        frame_source=source,
        model_path=model_path,
        display_backend=args.display,
        flask_port=args.port,
        ws_port=args.ws_port,
    )
    pipeline.run()


if __name__ == "__main__":
    main()
