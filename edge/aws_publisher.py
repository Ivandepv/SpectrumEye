"""
edge/aws_publisher.py — AWS IoT MQTT Publisher (STUB)

AWS IoT Core integration is deferred until cloud budget is available.

This module provides the AWSPublisher interface so edge/main.py does not
need to change when AWS is added. All methods return False (not published)
and log at DEBUG level.

When implemented, this will publish to:
    spectromeye/{sensor_id}/detection   ← BIE output (main event)
    spectromeye/{sensor_id}/rssi        ← RSSI time-series data point
    spectromeye/{sensor_id}/status      ← heartbeat every 30 seconds

Size constraints:
    AWS IoT Core MQTT payload: max 128 KB
    Spectrogram images go to S3 (not MQTT) — only S3 key in detection event

Usage:
    pub = AWSPublisher(sensor_id="spectromeye-pi5-001")
    pub.publish_detection(bie_output)   # no-op for now
"""

import logging

log = logging.getLogger(__name__)


class AWSPublisher:
    """
    No-op publisher. Drop-in replacement for real AWS IoT publisher.

    When AWS integration is added, replace this class body with the real
    implementation. The interface (method signatures, return types) must
    stay the same so main.py requires no changes.
    """

    def __init__(self, sensor_id: str = "spectromeye-pi5-001", **kwargs) -> None:
        self._sensor_id  = sensor_id
        self._connected  = False
        log.info("AWSPublisher: AWS deferred — detection events will not be uploaded")

    # ── Publications ──────────────────────────────────────────────

    def publish_detection(self, bie_output: dict) -> bool:
        """
        Publish a BIE output dict to spectromeye/{sensor_id}/detection.

        Returns:
            True if published successfully, False otherwise.
        """
        log.debug(
            "AWSPublisher [STUB]: detection dropped — event_id=%s  threat=%s",
            bie_output.get("event_id", "?"),
            bie_output.get("threat_level", "?"),
        )
        return False

    def publish_rssi(
        self,
        signal_class: str,
        rssi_dbfs:    float,
        confidence:   float,
        threat_score: int,
        timestamp_ms: int,
    ) -> bool:
        """
        Publish an RSSI data point to spectromeye/{sensor_id}/rssi.

        Returns:
            True if published successfully, False otherwise.
        """
        log.debug(
            "AWSPublisher [STUB]: RSSI dropped — class=%s  rssi=%.1f  score=%d",
            signal_class, rssi_dbfs, threat_score,
        )
        return False

    def publish_status(self, status_dict: dict) -> bool:
        """
        Publish a heartbeat to spectromeye/{sensor_id}/status.

        Returns:
            True if published successfully, False otherwise.
        """
        log.debug("AWSPublisher [STUB]: status dropped — uptime=%s s", status_dict.get("uptime_sec"))
        return False

    # ── State ─────────────────────────────────────────────────────

    @property
    def connected(self) -> bool:
        """True if connected to AWS IoT Core."""
        return self._connected

    def close(self) -> None:
        """Disconnect (no-op for stub)."""
        pass
