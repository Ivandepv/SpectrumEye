"""
edge/bie.py — Behavioral Inference Engine (BIE)

Converts raw CNN classification results + RSSI values into:
  - Behavioral state  (APPEARED / APPROACHING_FAST / STATIONARY / etc.)
  - Plain-English interpretation sentence
  - Threat level      (CLEAR / MODERATE / ELEVATED / CRITICAL)
  - Threat score      (0–10)

The BIE is pure Python logic — no ML, no GPU, no external dependencies.
It runs identically with simulated data or real RTL-SDR data.

Classes supported (Phase 2 scope):
  Key_Signal    — key fob / remote control
  Walkie_Talkie — narrowband FM radio
  LTE           — 4G cellular (always-on background)

Usage:
    bie = BIE(sensor_id="spectromeye-pi5-001")

    # Feed one classification result per sweep step
    result = bie.process(
        signal_class="Walkie_Talkie",
        confidence=0.94,
        rssi_dbfs=-58.3,
        center_freq_hz=462000000,
        timestamp_ms=1709847362000,
    )
    print(result["interpretation"])
    print(result["threat_level"])

Run standalone for unit tests:
    python bie.py --test
"""

import time
import uuid
import argparse
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

# ─── CONSTANTS ────────────────────────────────────────────────────

CLASS_LABELS = ["Key_Signal", "Walkie_Talkie", "LTE"]

BEHAVIORAL_STATES = [
    "APPEARED",
    "APPROACHING_FAST",
    "APPROACHING_SLOW",
    "STATIONARY",
    "DEPARTING_SLOW",
    "DEPARTING_FAST",
    "ERRATIC",
    "DISAPPEARED",
]

THREAT_LEVELS = ["CLEAR", "MODERATE", "ELEVATED", "CRITICAL"]

# ─── THRESHOLDS (tunable) ─────────────────────────────────────────

# RSSI slope thresholds (dBFS per second)
SLOPE_FAST      =  2.0   # > +2.0  → APPROACHING_FAST
SLOPE_SLOW      =  0.5   # > +0.5  → APPROACHING_SLOW
SLOPE_STABLE    =  0.5   # |slope| ≤ 0.5 → STATIONARY
SLOPE_DEP_SLOW  = -0.5   # < -0.5  → DEPARTING_SLOW
SLOPE_DEP_FAST  = -2.0   # < -2.0  → DEPARTING_FAST

# Variance threshold for ERRATIC classification.
# Computed on consecutive raw-RSSI differences (not EMA values):
#   trending signal  → diffs are nearly constant → variance ≈ 0
#   erratic signal   → diffs jump randomly       → variance >> 4
VARIANCE_ERRATIC = 4.0

# Minimum RSSI samples before computing trend (avoids noisy initial transitions)
MIN_SAMPLES_FOR_TREND = 5

# Samples kept in RSSI history window for slope/variance computation
HISTORY_WINDOW = 20

# Seconds a signal must be absent before marking DISAPPEARED
DISAPPEAR_TIMEOUT_SEC = 10.0

# CNN confidence below this → treated as Unknown, not classified
CONFIDENCE_THRESHOLD = 0.60

# State hysteresis: new state must persist this many consecutive
# samples before the displayed state changes (prevents flickering)
STATE_HYSTERESIS = 3

# EMA smoothing factor (0 = no smoothing, 1 = no update)
EMA_ALPHA = 0.25

# ─── RSSI TRACKER ─────────────────────────────────────────────────

@dataclass
class RSSITracker:
    """
    Tracks RSSI history for a single signal over time.

    Computes:
    - EMA-smoothed RSSI
    - Linear regression slope (dBFS/sec) over the last HISTORY_WINDOW samples
    - Variance of the sliding window (detects erratic behaviour)
    - Active duration since first detection
    """

    signal_class:   str
    center_freq_hz: int
    first_seen_ms:  int  = field(default_factory=lambda: int(time.time() * 1000))

    # Internal state
    _rssi_ema:      float            = field(default=None, init=False)
    _history:       deque            = field(default_factory=lambda: deque(maxlen=HISTORY_WINDOW), init=False)
    _raw_history:   deque            = field(default_factory=lambda: deque(maxlen=HISTORY_WINDOW), init=False)
    _timestamps:    deque            = field(default_factory=lambda: deque(maxlen=HISTORY_WINDOW), init=False)
    _last_seen_ms:  int              = field(default_factory=lambda: int(time.time() * 1000), init=False)

    def update(self, rssi_dbfs: float, timestamp_ms: int) -> None:
        """Add a new RSSI observation."""
        self._last_seen_ms = timestamp_ms
        self._raw_history.append(rssi_dbfs)   # unsmoothed — used for variance

        # EMA smoothing (used for slope)
        if self._rssi_ema is None:
            self._rssi_ema = rssi_dbfs
        else:
            self._rssi_ema = EMA_ALPHA * rssi_dbfs + (1 - EMA_ALPHA) * self._rssi_ema

        self._history.append(self._rssi_ema)
        self._timestamps.append(timestamp_ms / 1000.0)  # store in seconds

    @property
    def rssi(self) -> float:
        """Current EMA-smoothed RSSI."""
        return self._rssi_ema if self._rssi_ema is not None else -100.0

    @property
    def n_samples(self) -> int:
        return len(self._history)

    @property
    def slope(self) -> float:
        """
        RSSI trend in dBFS/second via linear regression over the history window.
        Positive = signal getting stronger (approaching).
        Negative = signal getting weaker (departing).
        Returns 0.0 if not enough samples.
        """
        if self.n_samples < MIN_SAMPLES_FOR_TREND:
            return 0.0
        t = np.array(self._timestamps)
        r = np.array(self._history)
        t_norm = t - t[0]   # start from 0 for numerical stability
        # slope from polyfit degree-1 (linear regression)
        coeffs = np.polyfit(t_norm, r, 1)
        return float(coeffs[0])

    @property
    def variance(self) -> float:
        """
        Variance of consecutive raw-RSSI differences.

        Why differences, not raw values:
          - A trending signal (clean approach/depart) has nearly constant diffs
            → variance ≈ 0, correctly NOT classified as erratic.
          - An erratic signal has wildly varying diffs
            → variance >> threshold, correctly classified as erratic.
          Using raw EMA window variance instead would falsely flag any trending
          signal as erratic (because windowed EMA values span a wide range).
        """
        if len(self._raw_history) < 3:
            return 0.0
        diffs = np.diff(np.array(self._raw_history))
        return float(np.var(diffs))

    @property
    def active_duration_sec(self) -> float:
        return (self._last_seen_ms - self.first_seen_ms) / 1000.0

    def is_lost(self, now_ms: int) -> bool:
        """True if no update received within DISAPPEAR_TIMEOUT_SEC."""
        return (now_ms - self._last_seen_ms) / 1000.0 > DISAPPEAR_TIMEOUT_SEC


# ─── BEHAVIORAL CLASSIFIER ────────────────────────────────────────

class BehavioralClassifier:
    """
    Determines behavioral state from RSSITracker metrics.
    Applies hysteresis to prevent rapid state flickering.
    """

    def __init__(self) -> None:
        self._pending_state: Optional[str] = None
        self._pending_count: int = 0
        self._current_state: str = "APPEARED"

    def classify(self, tracker: RSSITracker) -> str:
        """
        Compute behavioral state from tracker metrics.
        New state must hold for STATE_HYSTERESIS consecutive calls before committing.
        """
        if tracker.n_samples < MIN_SAMPLES_FOR_TREND:
            raw_state = "APPEARED"
        elif tracker.variance > VARIANCE_ERRATIC:
            raw_state = "ERRATIC"
        elif tracker.slope > SLOPE_FAST:
            raw_state = "APPROACHING_FAST"
        elif tracker.slope > SLOPE_SLOW:
            raw_state = "APPROACHING_SLOW"
        elif tracker.slope < SLOPE_DEP_FAST:
            raw_state = "DEPARTING_FAST"
        elif tracker.slope < SLOPE_DEP_SLOW:
            raw_state = "DEPARTING_SLOW"
        else:
            raw_state = "STATIONARY"

        # Hysteresis: only commit if raw_state held for STATE_HYSTERESIS steps
        if raw_state == self._pending_state:
            self._pending_count += 1
        else:
            self._pending_state = raw_state
            self._pending_count = 1

        if self._pending_count >= STATE_HYSTERESIS:
            self._current_state = raw_state

        return self._current_state

    def mark_disappeared(self) -> str:
        self._current_state = "DISAPPEARED"
        self._pending_state = None
        self._pending_count = 0
        return "DISAPPEARED"


# ─── NATURAL LANGUAGE SENTENCE LIBRARY ───────────────────────────

# Maps (signal_class, behavioral_state) → plain-English interpretation sentence.
# Every combination must have an entry. Fallback is defined in get_sentence().

_SENTENCES: dict[tuple, str] = {

    # ── Key Signal (key fob / remote control) ─────────────────────
    ("Key_Signal", "APPEARED"): (
        "A short-range remote control signal has been detected nearby. "
        "This could be a key fob, garage door opener, or similar device."
    ),
    ("Key_Signal", "APPROACHING_FAST"): (
        "A remote control signal is getting significantly stronger. "
        "Someone carrying a key fob or remote is moving toward this location quickly."
    ),
    ("Key_Signal", "APPROACHING_SLOW"): (
        "A remote control signal is gradually getting stronger. "
        "Someone carrying a key fob or remote may be walking toward this location."
    ),
    ("Key_Signal", "STATIONARY"): (
        "A remote control signal is active and stable nearby. "
        "A key fob or remote device is transmitting at a fixed location."
    ),
    ("Key_Signal", "DEPARTING_SLOW"): (
        "A remote control signal is weakening. "
        "The device or person carrying it appears to be moving away."
    ),
    ("Key_Signal", "DEPARTING_FAST"): (
        "A remote control signal has dropped sharply. "
        "The device is departing this area quickly."
    ),
    ("Key_Signal", "ERRATIC"): (
        "A remote control signal is fluctuating erratically. "
        "The device may be in motion or the user is repeatedly activating it."
    ),
    ("Key_Signal", "DISAPPEARED"): (
        "The remote control signal has been lost. "
        "The device is no longer in range."
    ),

    # ── Walkie Talkie (NFM radio) ──────────────────────────────────
    ("Walkie_Talkie", "APPEARED"): (
        "A walkie-talkie radio signal has been detected nearby. "
        "Someone in the area is using a two-way radio."
    ),
    ("Walkie_Talkie", "APPROACHING_FAST"): (
        "A walkie-talkie signal is getting much stronger rapidly. "
        "A person using a two-way radio is approaching this location at speed."
    ),
    ("Walkie_Talkie", "APPROACHING_SLOW"): (
        "A walkie-talkie signal is gradually getting stronger. "
        "A person using a two-way radio is moving closer to this location."
    ),
    ("Walkie_Talkie", "STATIONARY"): (
        "A walkie-talkie signal is active and stable. "
        "A person with a two-way radio is nearby and not moving."
    ),
    ("Walkie_Talkie", "DEPARTING_SLOW"): (
        "A walkie-talkie signal is weakening. "
        "The radio operator appears to be moving away from this location."
    ),
    ("Walkie_Talkie", "DEPARTING_FAST"): (
        "A walkie-talkie signal has dropped sharply. "
        "The radio operator is departing this area quickly."
    ),
    ("Walkie_Talkie", "ERRATIC"): (
        "A walkie-talkie signal is fluctuating erratically. "
        "The operator may be moving between locations or the signal is being blocked and unblocked."
    ),
    ("Walkie_Talkie", "DISAPPEARED"): (
        "The walkie-talkie signal has been lost. "
        "The radio is no longer transmitting or has moved out of range."
    ),

    # ── LTE (4G cellular — always-on background) ───────────────────
    ("LTE", "APPEARED"): (
        "A 4G LTE cellular signal has been detected. "
        "This is normal background activity from a nearby mobile tower or device."
    ),
    ("LTE", "APPROACHING_FAST"): (
        "A cellular signal is getting much stronger. "
        "A person carrying a phone or mobile device is moving toward this location quickly."
    ),
    ("LTE", "APPROACHING_SLOW"): (
        "A cellular signal is gradually getting stronger. "
        "A person carrying a phone is likely walking toward this location."
    ),
    ("LTE", "STATIONARY"): (
        "Normal 4G LTE cellular activity. "
        "A mobile tower or nearby device is transmitting at a stable level."
    ),
    ("LTE", "DEPARTING_SLOW"): (
        "A cellular signal is weakening. "
        "A person with a mobile device is moving away."
    ),
    ("LTE", "DEPARTING_FAST"): (
        "A cellular signal has dropped sharply. "
        "A person with a mobile device is leaving the area quickly."
    ),
    ("LTE", "ERRATIC"): (
        "A cellular signal is fluctuating unusually. "
        "This may indicate a device moving in and out of coverage, or unusual interference."
    ),
    ("LTE", "DISAPPEARED"): (
        "The cellular signal has been lost. "
        "The device or tower is no longer detectable."
    ),
}


def get_sentence(signal_class: str, state: str) -> str:
    """Return the interpretation sentence for a (class, state) pair."""
    return _SENTENCES.get(
        (signal_class, state),
        f"{signal_class} signal detected — state: {state}. Monitoring."
    )


# ─── THREAT CALCULATOR ────────────────────────────────────────────

# Per-class base threat score when signal is active
_CLASS_BASE_SCORE: dict[str, int] = {
    "Key_Signal":    2,    # low — remote controls are usually benign
    "Walkie_Talkie": 4,    # moderate — could indicate coordinated activity
    "LTE":           0,    # background — always present, not a threat
}

# State multipliers applied to base score
_STATE_MULTIPLIER: dict[str, float] = {
    "APPEARED":        1.0,
    "APPROACHING_FAST":2.0,
    "APPROACHING_SLOW":1.5,
    "STATIONARY":      1.0,
    "DEPARTING_SLOW":  0.5,
    "DEPARTING_FAST":  0.3,
    "ERRATIC":         1.8,
    "DISAPPEARED":     0.0,
}


def calculate_threat(active_signals: list[dict]) -> tuple[str, int]:
    """
    Compute overall threat level and score from all currently active signals.

    Args:
        active_signals: list of dicts, each with keys:
            signal_class, behavioral_state, rssi_dbfs, confidence

    Returns:
        (threat_level: str, threat_score: int)  where score is 0–10
    """
    if not active_signals:
        return "CLEAR", 0

    total_score = 0.0
    for sig in active_signals:
        cls   = sig.get("signal_class", "")
        state = sig.get("behavioral_state", "STATIONARY")
        conf  = sig.get("confidence", 1.0)

        base  = _CLASS_BASE_SCORE.get(cls, 1)
        mult  = _STATE_MULTIPLIER.get(state, 1.0)
        total_score += base * mult * conf

    # Clamp to 0–10
    score = int(min(10, round(total_score)))

    if score == 0:
        level = "CLEAR"
    elif score <= 4:
        level = "MODERATE"
    elif score <= 8:
        level = "ELEVATED"
    else:
        level = "CRITICAL"

    return level, score


# ─── MAIN BIE ORCHESTRATOR ────────────────────────────────────────

class BIE:
    """
    Behavioral Inference Engine.

    Maintains state for all tracked signals across sweep cycles.
    Call process() once per sweep frame received from the CNN.

    Example:
        bie = BIE(sensor_id="spectromeye-pi5-001")
        output = bie.process(
            signal_class="Walkie_Talkie",
            confidence=0.91,
            rssi_dbfs=-62.0,
            center_freq_hz=462000000,
            timestamp_ms=int(time.time() * 1000),
        )
    """

    def __init__(self, sensor_id: str = "spectromeye-pi5-001") -> None:
        self.sensor_id = sensor_id
        # Active trackers: key = signal_class (one per class in 3-class scope)
        self._trackers:    dict[str, RSSITracker]          = {}
        self._classifiers: dict[str, BehavioralClassifier] = {}

    def process(
        self,
        signal_class:   str,
        confidence:     float,
        rssi_dbfs:      float,
        center_freq_hz: int,
        timestamp_ms:   Optional[int] = None,
        frame_id:       int = 0,
    ) -> dict:
        """
        Process one CNN classification result and produce a BIE output dict.

        Args:
            signal_class:   one of CLASS_LABELS
            confidence:     CNN softmax confidence (0.0–1.0)
            rssi_dbfs:      measured RSSI for this band (dBFS)
            center_freq_hz: center frequency of the captured band (Hz)
            timestamp_ms:   Unix timestamp in milliseconds (defaults to now)
            frame_id:       monotonic frame counter from sweep pipeline

        Returns:
            BIE output dict matching the interface contract (Interface C)
        """
        if timestamp_ms is None:
            timestamp_ms = int(time.time() * 1000)

        # Low-confidence detections → treat as noise, don't track
        if confidence < CONFIDENCE_THRESHOLD:
            signal_class = "Unknown"

        # ── Check for disappeared signals ────────────────────────
        for cls in list(self._trackers.keys()):
            if cls != signal_class and self._trackers[cls].is_lost(timestamp_ms):
                self._classifiers[cls].mark_disappeared()
                # Keep tracker briefly for the DISAPPEARED state, then remove
                del self._trackers[cls]
                del self._classifiers[cls]

        # ── Update or create tracker for this signal ──────────────
        if signal_class not in ("Unknown",) and signal_class in CLASS_LABELS:
            if signal_class not in self._trackers:
                self._trackers[signal_class] = RSSITracker(
                    signal_class=signal_class,
                    center_freq_hz=center_freq_hz,
                    first_seen_ms=timestamp_ms,
                )
                self._classifiers[signal_class] = BehavioralClassifier()

            tracker    = self._trackers[signal_class]
            classifier = self._classifiers[signal_class]
            tracker.update(rssi_dbfs, timestamp_ms)
            state = classifier.classify(tracker)
        else:
            # Unknown or below threshold — report but don't track
            state   = "APPEARED"
            tracker = None

        # ── Build active signals snapshot ────────────────────────
        active_signals = []
        for cls, trk in self._trackers.items():
            clf   = self._classifiers[cls]
            s     = clf._current_state
            active_signals.append({
                "signal_class":      cls,
                "behavioral_state":  s,
                "rssi_dbfs":         round(trk.rssi, 1),
                "confidence":        confidence if cls == signal_class else 1.0,
                "active_duration_sec": round(trk.active_duration_sec, 1),
            })

        # ── Threat assessment ─────────────────────────────────────
        threat_level, threat_score = calculate_threat(active_signals)

        # ── Natural language output ───────────────────────────────
        interpretation = get_sentence(signal_class, state)
        technical = (
            f"{signal_class} | state: {state} | "
            f"RSSI: {rssi_dbfs:.1f} dBFS | "
            f"slope: {tracker.slope:+.2f} dBFS/s | "
            f"confidence: {confidence*100:.0f}%"
            if tracker else
            f"{signal_class} | confidence: {confidence*100:.0f}% (below threshold)"
        )

        return {
            # Identification
            "event_id":         f"evt_{uuid.uuid4().hex[:8]}",
            "sensor_id":        self.sensor_id,
            "timestamp_ms":     timestamp_ms,
            "frame_id":         frame_id,

            # Classification
            "signal_class":     signal_class,
            "confidence":       round(confidence, 4),
            "center_freq_hz":   center_freq_hz,

            # Behavioral
            "behavioral_state":      state,
            "rssi_dbfs":             round(tracker.rssi if tracker else rssi_dbfs, 1),
            "rssi_raw_dbfs":         round(rssi_dbfs, 1),
            "rssi_slope_dbfs_per_s": round(tracker.slope if tracker else 0.0, 3),
            "rssi_variance":         round(tracker.variance if tracker else 0.0, 3),
            "active_duration_sec":   round(tracker.active_duration_sec if tracker else 0.0, 1),
            "n_samples":             tracker.n_samples if tracker else 0,

            # Human-readable
            "interpretation":  interpretation,
            "technical":       technical,

            # Threat
            "threat_level":    threat_level,
            "threat_score":    threat_score,
            "alert_fire":      threat_level in ("ELEVATED", "CRITICAL"),

            # Active signals snapshot
            "active_signals":  active_signals,
        }

    def get_all_states(self) -> dict[str, str]:
        """Return current behavioral state for every tracked signal."""
        return {
            cls: clf._current_state
            for cls, clf in self._classifiers.items()
        }

    def reset(self) -> None:
        """Clear all tracked signals (e.g. between test scenarios)."""
        self._trackers.clear()
        self._classifiers.clear()


# ─── UNIT TESTS ───────────────────────────────────────────────────

def _run_tests() -> None:
    print("Running BIE unit tests...\n")
    passed = 0
    failed = 0

    def check(label: str, condition: bool) -> None:
        nonlocal passed, failed
        status = "PASS" if condition else "FAIL"
        print(f"  [{status}] {label}")
        if condition:
            passed += 1
        else:
            failed += 1

    bie = BIE(sensor_id="test-sensor")
    now = int(time.time() * 1000)

    # ── Test 1: APPEARED on first detection ────────────────────────
    print("Test 1 — APPEARED on first detection:")
    out = bie.process("Walkie_Talkie", 0.95, -65.0, 462000000, now)
    check("state is APPEARED (not enough history)", out["behavioral_state"] == "APPEARED")
    check("threat_level not CLEAR (walkie talkie active)", out["threat_level"] != "CLEAR")
    check("interpretation is a string", isinstance(out["interpretation"], str))
    check("event_id starts with evt_", out["event_id"].startswith("evt_"))

    # ── Test 2: APPROACHING_FAST after rising RSSI ─────────────────
    print("\nTest 2 — APPROACHING_FAST after rising RSSI sequence:")
    bie.reset()
    t = now
    # Feed 15 samples with rapidly increasing RSSI (+4 dBFS/s)
    for i in range(15):
        rssi = -80.0 + i * 2.0     # +2 dBFS per step
        t   += 500                  # 0.5 seconds between samples → slope ≈ +4 dBFS/s
        out  = bie.process("Walkie_Talkie", 0.93, rssi, 462000000, t)
    check("state is APPROACHING_FAST", out["behavioral_state"] == "APPROACHING_FAST")
    check("slope is positive", out["rssi_slope_dbfs_per_s"] > 0)
    check("threat not CLEAR", out["threat_level"] != "CLEAR")

    # ── Test 3: STATIONARY after stable RSSI ──────────────────────
    print("\nTest 3 — STATIONARY after stable RSSI:")
    bie.reset()
    t = now
    for i in range(15):
        t   += 500
        rssi = -60.0 + np.random.normal(0, 0.3)   # tight noise around -60
        out  = bie.process("Walkie_Talkie", 0.91, rssi, 462000000, t)
    check("state is STATIONARY", out["behavioral_state"] == "STATIONARY")
    check("slope near zero", abs(out["rssi_slope_dbfs_per_s"]) < SLOPE_SLOW)

    # ── Test 4: DEPARTING_FAST after falling RSSI ─────────────────
    print("\nTest 4 — DEPARTING_FAST after falling RSSI:")
    bie.reset()
    t = now
    for i in range(15):
        rssi = -40.0 - i * 2.5    # -2.5 dBFS per step
        t   += 500
        out  = bie.process("Key_Signal", 0.88, rssi, 433000000, t)
    check("state is DEPARTING_FAST", out["behavioral_state"] == "DEPARTING_FAST")
    check("slope is negative", out["rssi_slope_dbfs_per_s"] < 0)

    # ── Test 5: ERRATIC with high variance ────────────────────────
    print("\nTest 5 — ERRATIC with high-variance RSSI:")
    bie.reset()
    t = now
    rng = np.random.default_rng(99)
    for i in range(15):
        rssi = -60.0 + rng.uniform(-8, 8)   # ±8 dBFS random jumps
        t   += 500
        out  = bie.process("Walkie_Talkie", 0.89, rssi, 462000000, t)
    check("state is ERRATIC", out["behavioral_state"] == "ERRATIC")
    check("variance > threshold", out["rssi_variance"] > VARIANCE_ERRATIC)

    # ── Test 6: Low confidence → Unknown ──────────────────────────
    print("\nTest 6 — Low confidence below threshold:")
    bie.reset()
    out = bie.process("Walkie_Talkie", 0.45, -70.0, 462000000, now)
    check("class becomes Unknown", out["signal_class"] == "Unknown")

    # ── Test 7: LTE → CLEAR threat ────────────────────────────────
    print("\nTest 7 — LTE alone produces CLEAR threat:")
    bie.reset()
    t = now
    for i in range(10):
        t  += 500
        out = bie.process("LTE", 0.99, -75.0, 1800000000, t)
    check("LTE is STATIONARY", out["behavioral_state"] == "STATIONARY")
    check("threat is CLEAR for LTE only", out["threat_level"] == "CLEAR")

    # ── Test 8: Sentence library covers all combinations ──────────
    print("\nTest 8 — Sentence library completeness:")
    missing = []
    for cls in CLASS_LABELS:
        for state in BEHAVIORAL_STATES:
            key = (cls, state)
            if key not in _SENTENCES:
                missing.append(key)
    check(f"all {len(CLASS_LABELS) * len(BEHAVIORAL_STATES)} sentences defined", len(missing) == 0)
    if missing:
        for m in missing:
            print(f"    Missing: {m}")

    # ── Test 9: Active signals snapshot ───────────────────────────
    print("\nTest 9 — Active signals snapshot contains correct classes:")
    bie.reset()
    t = now
    for i in range(6):
        t += 500
        bie.process("Walkie_Talkie", 0.91, -65.0, 462000000, t)
        bie.process("LTE", 0.99, -75.0, 1800000000, t)
    out = bie.process("Key_Signal", 0.85, -70.0, 433000000, t + 500)
    classes_in_snapshot = {s["signal_class"] for s in out["active_signals"]}
    check("snapshot contains Walkie_Talkie", "Walkie_Talkie" in classes_in_snapshot)
    check("snapshot contains LTE", "LTE" in classes_in_snapshot)
    check("snapshot contains Key_Signal", "Key_Signal" in classes_in_snapshot)

    # ── Summary ───────────────────────────────────────────────────
    print(f"\n{'─'*40}")
    print(f"Results: {passed} passed / {failed} failed / {passed + failed} total")
    if failed == 0:
        print("All tests passed.")
    else:
        print(f"{failed} test(s) FAILED — review output above.")


# ─── ENTRY POINT ──────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Behavioral Inference Engine")
    parser.add_argument("--test", action="store_true", help="Run unit tests")
    args = parser.parse_args()

    if args.test:
        _run_tests()
    else:
        # Quick interactive demo
        print("BIE interactive demo — feeding simulated Walkie_Talkie approach sequence\n")
        bie = BIE()
        t   = int(time.time() * 1000)
        rssi_sequence = [-85, -82, -79, -76, -73, -70, -67, -64, -61, -58,
                         -56, -55, -55, -55, -54, -55, -56, -60, -65, -72]
        for i, rssi in enumerate(rssi_sequence):
            t  += 600
            out = bie.process("Walkie_Talkie", 0.92, float(rssi), 462000000, t, frame_id=i)
            print(
                f"  [{i+1:02d}] RSSI={rssi:4.0f} dBFS  "
                f"state={out['behavioral_state']:<16}  "
                f"slope={out['rssi_slope_dbfs_per_s']:+.2f}  "
                f"threat={out['threat_level']}"
            )
        print(f"\nFinal interpretation:\n  {out['interpretation']}")
