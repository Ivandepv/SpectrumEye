"""
edge/sensor_fusion.py — ESP32 Environmental Sensor Fusion

Subscribes to MQTT topic:  spectromeye/sensors/{esp32_id}
Maintains the last known environmental context dict.

Stale detection: if no message is received for ENV_STALE_TIMEOUT seconds,
  data_stale is set to True. Callers should note this in the BIE output.

If paho-mqtt is not installed or no MQTT broker is reachable, the module
falls back silently — get_env_context() returns null values with data_stale=True.

Interface E payload (from ESP32):
    {
        "esp32_id":      str,
        "timestamp_ms":  int,
        "temperature_c": float,
        "humidity_pct":  float,
        "sound_level":   int,     # 0–1023 ADC raw
        "pir_motion":    bool,
        "vibration":     bool,
        "light_level":   int,     # 0–1023 ADC raw
    }

Usage:
    sf = SensorFusion(broker_host="localhost", esp32_id="grove-hub-001")
    ctx = sf.get_env_context()
    sf.close()
"""

import json
import time
import logging
import threading

log = logging.getLogger(__name__)

# Seconds before env data is considered stale
ENV_STALE_TIMEOUT = 30.0

# Default/null env context returned when no data is available
_NULL_ENV: dict = {
    "temperature_c": None,
    "humidity_pct":  None,
    "pir_motion":    False,
    "sound_level":   None,
    "vibration":     False,
    "light_level":   None,
    "data_stale":    True,
}


class SensorFusion:
    """
    Subscribes to ESP32 MQTT topic and caches the latest environmental context.

    Thread-safe: MQTT callbacks run in a background thread; get_env_context()
    acquires a lock before reading.
    """

    def __init__(
        self,
        broker_host: str = "localhost",
        broker_port: int = 1883,
        esp32_id:    str = "grove-hub-001",
    ) -> None:
        self._esp32_id    = esp32_id
        self._env         = dict(_NULL_ENV)
        self._last_update = 0.0
        self._lock        = threading.Lock()
        self._client      = None

        self._connect(broker_host, broker_port)

    # ── MQTT setup ────────────────────────────────────────────────

    def _connect(self, host: str, port: int) -> None:
        try:
            import paho.mqtt.client as mqtt
        except ImportError:
            log.warning("paho-mqtt not installed — running without environmental sensors")
            return

        client = mqtt.Client()
        client.on_connect = self._on_connect
        client.on_message = self._on_message

        try:
            client.connect(host, port, keepalive=60)
            client.loop_start()
            self._client = client
            log.info("SensorFusion: connected to MQTT broker %s:%d", host, port)
        except Exception as exc:
            log.warning("SensorFusion: could not connect to MQTT broker (%s) — no env data", exc)

    def _on_connect(self, client, userdata, flags, rc) -> None:
        if rc == 0:
            topic = f"spectromeye/sensors/{self._esp32_id}"
            client.subscribe(topic)
            log.info("SensorFusion: subscribed to %s", topic)
        else:
            log.warning("SensorFusion: MQTT connect failed, rc=%d", rc)

    def _on_message(self, client, userdata, msg) -> None:
        try:
            data = json.loads(msg.payload.decode())
            with self._lock:
                self._env = {
                    "temperature_c": data.get("temperature_c"),
                    "humidity_pct":  data.get("humidity_pct"),
                    "pir_motion":    bool(data.get("pir_motion", False)),
                    "sound_level":   data.get("sound_level"),
                    "vibration":     bool(data.get("vibration", False)),
                    "light_level":   data.get("light_level"),
                    "data_stale":    False,
                }
                self._last_update = time.monotonic()
        except Exception as exc:
            log.warning("SensorFusion: bad MQTT payload: %s", exc)

    # ── Public API ────────────────────────────────────────────────

    def get_env_context(self) -> dict:
        """
        Return the latest environmental context dict.

        Always returns a complete dict — None values mean sensor not available.
        data_stale=True means no update in the last ENV_STALE_TIMEOUT seconds.
        """
        with self._lock:
            result = dict(self._env)
            elapsed = time.monotonic() - self._last_update
            result["data_stale"] = (self._last_update == 0.0) or (elapsed > ENV_STALE_TIMEOUT)
            return result

    def inject(self, payload: dict) -> None:
        """
        Inject a sensor reading directly (for testing / simulation mode).

        Accepts the same dict structure as the MQTT payload.
        """
        with self._lock:
            self._env = {
                "temperature_c": payload.get("temperature_c"),
                "humidity_pct":  payload.get("humidity_pct"),
                "pir_motion":    bool(payload.get("pir_motion", False)),
                "sound_level":   payload.get("sound_level"),
                "vibration":     bool(payload.get("vibration", False)),
                "light_level":   payload.get("light_level"),
                "data_stale":    False,
            }
            self._last_update = time.monotonic()

    def close(self) -> None:
        """Stop MQTT background thread."""
        if self._client:
            self._client.loop_stop()
            self._client.disconnect()
            self._client = None
