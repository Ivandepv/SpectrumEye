"""
edge/alert_controller.py — Alert Controller

Evaluates BIE output threat levels and triggers alerts.

Alert channels (in priority order):
  1. Terminal bell + colored ANSI text (always available)
  2. GPIO LED / buzzer on Raspberry Pi 5 (when gpiod is available)
  3. System sound via aplay/paplay (when available on Pi)

Cooldown: alerts are rate-limited per threat level to avoid flooding.

Usage:
    ac = AlertController(sensor_id="spectromeye-pi5-001")
    ac.evaluate(bie_output)   # call after every BIE.process()
    ac.close()

Run standalone for a visual test:
    python alert_controller.py --test
"""

import time
import logging
import argparse
import subprocess
from typing import Optional

log = logging.getLogger(__name__)

# ─── CONFIGURATION ────────────────────────────────────────────────

# Minimum seconds between alerts of the same level (prevents spam)
_COOLDOWN: dict[str, float] = {
    "CLEAR":    60.0,
    "MODERATE": 30.0,
    "ELEVATED": 15.0,
    "CRITICAL":  5.0,
}

# ANSI colors for terminal output
_COLORS = {
    "CLEAR":    "\033[32m",   # green
    "MODERATE": "\033[33m",   # yellow
    "ELEVATED": "\033[91m",   # bright red
    "CRITICAL": "\033[1;31m", # bold red
    "RESET":    "\033[0m",
}

# Threat levels that fire alerts
_ALERT_LEVELS = {"ELEVATED", "CRITICAL"}

# GPIO pin numbers (BCM) — set to None to disable
_GPIO_LED_PIN    = None   # e.g. 17 for a red LED
_GPIO_BUZZER_PIN = None   # e.g. 27 for a buzzer


# ─── ALERT CONTROLLER ─────────────────────────────────────────────

class AlertController:
    """
    Evaluates BIE threat level and triggers appropriate alerts.
    """

    def __init__(self, sensor_id: str = "spectromeye-pi5-001") -> None:
        self._sensor_id  = sensor_id
        self._last_alert: dict[str, float] = {}   # level → last alert time
        self._gpio       = self._init_gpio()

    # ── GPIO setup ────────────────────────────────────────────────

    def _init_gpio(self) -> Optional[object]:
        """Try to initialise GPIO. Returns chip handle or None if unavailable."""
        if _GPIO_LED_PIN is None and _GPIO_BUZZER_PIN is None:
            return None
        try:
            import gpiod
            chip = gpiod.Chip("gpiochip0")
            log.info("AlertController: GPIO initialised")
            return chip
        except Exception:
            log.debug("AlertController: gpiod not available — GPIO alerts disabled")
            return None

    # ── Public API ────────────────────────────────────────────────

    def evaluate(self, bie_output: dict) -> None:
        """
        Check the BIE output and trigger alerts if warranted.

        Args:
            bie_output: BIE output dict from BIE.process()
        """
        level = bie_output.get("threat_level", "CLEAR")
        score = bie_output.get("threat_score", 0)
        cls   = bie_output.get("signal_class", "Unknown")
        state = bie_output.get("behavioral_state", "")

        # Always print a colored status line
        self._print_status(level, score, cls, state)

        # Fire alert only for ELEVATED / CRITICAL with cooldown check
        if level in _ALERT_LEVELS and self._cooldown_ok(level):
            self._fire_alert(level, bie_output)
            self._last_alert[level] = time.monotonic()

    # ── Internal ──────────────────────────────────────────────────

    def _cooldown_ok(self, level: str) -> bool:
        """Return True if enough time has passed since the last alert of this level."""
        last = self._last_alert.get(level, 0.0)
        return (time.monotonic() - last) >= _COOLDOWN.get(level, 30.0)

    def _print_status(self, level: str, score: int, cls: str, state: str) -> None:
        """Print a one-line colored status to the terminal."""
        color = _COLORS.get(level, "")
        reset = _COLORS["RESET"]
        print(
            f"{color}[{level:8s}] score={score:2d}  {cls:<16} {state}{reset}",
            flush=True,
        )

    def _fire_alert(self, level: str, bie_output: dict) -> None:
        """Trigger all available alert channels."""
        interp = bie_output.get("interpretation", "")
        color  = _COLORS.get(level, "")
        reset  = _COLORS["RESET"]

        # Terminal alert (always)
        print(f"\n{color}{'!'*60}")
        print(f"  ALERT — {level}")
        print(f"  {interp}")
        print(f"{'!'*60}{reset}\n", flush=True)

        # Terminal bell
        print("\a", end="", flush=True)

        # GPIO (if configured)
        if self._gpio and (_GPIO_LED_PIN or _GPIO_BUZZER_PIN):
            self._pulse_gpio(level)

        # System sound (best-effort)
        self._play_sound(level)

        log.warning("ALERT fired: %s | %s | %s", level, bie_output.get("signal_class"), interp[:60])

    def _pulse_gpio(self, level: str) -> None:
        """Pulse LED/buzzer GPIO pin. Silently no-ops if GPIO unavailable."""
        try:
            import gpiod
            pulses = 3 if level == "CRITICAL" else 1
            for pin in filter(None, [_GPIO_LED_PIN, _GPIO_BUZZER_PIN]):
                line = self._gpio.get_line(pin)
                line.request(consumer="spectromeye", type=gpiod.LINE_REQ_DIR_OUT)
                for _ in range(pulses):
                    line.set_value(1)
                    time.sleep(0.15)
                    line.set_value(0)
                    time.sleep(0.1)
                line.release()
        except Exception as exc:
            log.debug("AlertController: GPIO pulse failed: %s", exc)

    def _play_sound(self, level: str) -> None:
        """Play an alert sound using aplay (ALSA) or paplay (PulseAudio), if available."""
        # Use different beep frequencies for threat levels via speaker-test or beep
        # This is best-effort — silently skip if no audio tools are present
        try:
            freq = 880 if level == "CRITICAL" else 440
            subprocess.run(
                ["speaker-test", "-t", "sine", "-f", str(freq), "-l", "1"],
                capture_output=True,
                timeout=2.0,
            )
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass   # speaker-test not installed or no audio device

    def close(self) -> None:
        """Release GPIO resources."""
        if self._gpio:
            try:
                self._gpio.close()
            except Exception:
                pass


# ─── SELF-TEST ────────────────────────────────────────────────────

def _run_test() -> None:
    """Visual test: cycle through all threat levels."""
    print("AlertController visual test — cycling through threat levels\n")
    ac = AlertController()

    mock_outputs = [
        {"threat_level": "CLEAR",    "threat_score": 0,  "signal_class": "LTE",          "behavioral_state": "STATIONARY", "interpretation": "LTE background signal, no threat."},
        {"threat_level": "MODERATE", "threat_score": 3,  "signal_class": "Key_Signal",   "behavioral_state": "APPEARED",   "interpretation": "A short-range remote control signal has been detected nearby."},
        {"threat_level": "ELEVATED", "threat_score": 6,  "signal_class": "Walkie_Talkie","behavioral_state": "STATIONARY", "interpretation": "A walkie-talkie signal is active and stable nearby."},
        {"threat_level": "CRITICAL", "threat_score": 10, "signal_class": "Walkie_Talkie","behavioral_state": "APPROACHING_FAST", "interpretation": "A walkie-talkie signal is getting significantly stronger."},
    ]

    for out in mock_outputs:
        ac.evaluate(out)
        time.sleep(0.3)

    ac.close()
    print("\nVisual test complete.")


# ─── ENTRY POINT ──────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SpectrumEye Alert Controller")
    parser.add_argument("--test", action="store_true", help="Run visual test")
    args = parser.parse_args()

    if args.test:
        _run_test()
    else:
        parser.print_help()
