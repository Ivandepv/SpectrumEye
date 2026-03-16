"""
edge/local_display.py — Local Dashboard Output

Renders BIE output to a local display. Two backends are supported:

  1. TERMINAL (default) — formatted ANSI table output.
     Works everywhere: development, SSH, headless Pi.

  2. FLASK (optional) — lightweight HTTP server serving a minimal HTML page
     that auto-refreshes every second. Connect a browser on the Pi's 7"
     HDMI touchscreen to http://localhost:5000.
     Requires: pip install flask

The two backends share the same update() interface, so main.py does not
need to know which one is active.

Usage:
    display = LocalDisplay(backend="terminal")   # or "flask"
    display.start()
    display.update(bie_output)
    display.stop()
"""

import time
import logging
import argparse
import threading
from typing import Optional

log = logging.getLogger(__name__)

# ─── ANSI COLORS ──────────────────────────────────────────────────

_C = {
    "CLEAR":    "\033[32m",
    "MODERATE": "\033[33m",
    "ELEVATED": "\033[91m",
    "CRITICAL": "\033[1;31m",
    "header":   "\033[1;36m",   # bold cyan
    "dim":      "\033[2m",
    "reset":    "\033[0m",
}


# ─── TERMINAL BACKEND ─────────────────────────────────────────────

class TerminalDisplay:
    """
    Renders BIE output as a compact ANSI status block in the terminal.
    Overwrites the previous output in-place using ANSI cursor controls.
    """

    _LINES = 12   # number of lines to overwrite per update

    def __init__(self) -> None:
        self._first = True

    def start(self) -> None:
        print(f"\n{_C['header']}{'─'*60}")
        print("  SpectrumEye — Edge Monitor")
        print(f"{'─'*60}{_C['reset']}\n")

    def update(self, bie: dict) -> None:
        level = bie.get("threat_level", "CLEAR")
        color = _C.get(level, "")
        rst   = _C["reset"]
        dim   = _C["dim"]

        if not self._first:
            # Move cursor up to overwrite previous block
            print(f"\033[{self._LINES}A", end="")
        self._first = False

        ts    = bie.get("timestamp_ms", 0) / 1000.0
        ts_s  = time.strftime("%H:%M:%S", time.localtime(ts))
        score = bie.get("threat_score", 0)
        cls   = bie.get("signal_class", "Unknown")
        state = bie.get("behavioral_state", "")
        rssi  = bie.get("rssi_dbfs", 0.0)
        slope = bie.get("rssi_slope_dbfs_per_s", 0.0)
        conf  = bie.get("confidence", 0.0)
        dur   = bie.get("active_duration_sec", 0.0)
        interp= bie.get("interpretation", "")[:55]

        env   = bie.get("env_context", {})
        temp  = env.get("temperature_c")
        hum   = env.get("humidity_pct")
        pir   = env.get("pir_motion", False)
        stale = env.get("data_stale", True)
        env_s = (f"{temp:.1f}°C  {hum:.0f}%  PIR={'Y' if pir else 'N'}"
                 if temp is not None and not stale else "no sensor data")

        n_sig = len(bie.get("active_signals", []))

        print(f"  {dim}{ts_s}{rst}  "
              f"{color}[{level:8s}  score={score:2d}]{rst}  "
              f"{cls:<16}  {state}")
        print(f"  RSSI: {rssi:5.1f} dBFS   slope: {slope:+.2f} dBFS/s   "
              f"conf: {conf*100:.0f}%   dur: {dur:.0f}s")
        print(f"  Signals tracked: {n_sig}   Env: {env_s}")
        print(f"  {dim}{interp}{rst}")
        print()

        # Active signals table
        signals = bie.get("active_signals", [])
        print(f"  {_C['header']}{'Class':<16}  {'State':<18}  {'RSSI':>8}  {'Dur':>6}{rst}")
        if signals:
            for s in signals:
                sc = s['signal_class']
                ss = s['behavioral_state']
                sr = s['rssi_dbfs']
                sd = s['active_duration_sec']
                print(f"  {sc:<16}  {ss:<18}  {sr:>5.1f} dBF  {sd:>4.0f}s")
        else:
            print(f"  {dim}(no active signals){rst}")

        # Pad to fixed height
        rows_used = 6 + max(1, len(signals))
        for _ in range(self._LINES - rows_used):
            print()

    def stop(self) -> None:
        print(f"\n{_C['dim']}Display stopped.{_C['reset']}")


# ─── FLASK BACKEND ────────────────────────────────────────────────

class FlaskDisplay:
    """
    Serves a minimal HTML page at http://localhost:5000.
    The page auto-refreshes every 2 seconds via meta refresh.
    Intended for the Pi's 7" HDMI screen in full-screen browser mode.
    """

    _HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta http-equiv="refresh" content="2">
<title>SpectrumEye Monitor</title>
<style>
  body {{ background:#0d1117; color:#e6edf3; font-family:monospace; padding:20px; }}
  .header {{ color:#58a6ff; font-size:1.4em; margin-bottom:12px; }}
  .threat-CLEAR    {{ color:#3fb950; }}
  .threat-MODERATE {{ color:#d29922; }}
  .threat-ELEVATED {{ color:#f85149; }}
  .threat-CRITICAL {{ color:#ff0000; font-weight:bold; animation:blink 0.5s step-start infinite; }}
  @keyframes blink {{ 50% {{ opacity:0; }} }}
  .label {{ color:#8b949e; }}
  table {{ border-collapse:collapse; width:100%; margin-top:12px; }}
  th {{ color:#58a6ff; text-align:left; padding:4px 12px; border-bottom:1px solid #30363d; }}
  td {{ padding:4px 12px; }}
  tr:hover {{ background:#161b22; }}
</style>
</head>
<body>
<div class="header">SpectrumEye — Edge Monitor</div>
<div>
  <span class="label">Time: </span>{ts} &nbsp;
  <span class="label">Threat: </span>
  <span class="threat-{level}">{level} (score {score}/10)</span>
</div>
<div><span class="label">Signal: </span>{cls} &nbsp; <span class="label">State: </span>{state}</div>
<div><span class="label">RSSI: </span>{rssi:.1f} dBFS &nbsp;
     <span class="label">Slope: </span>{slope:+.2f} dBFS/s &nbsp;
     <span class="label">Confidence: </span>{conf:.0f}%</div>
<div><span class="label">Interpretation: </span>{interp}</div>
{env_row}
<table>
  <tr><th>Class</th><th>State</th><th>RSSI (dBFS)</th><th>Duration (s)</th></tr>
  {rows}
</table>
</body>
</html>"""

    def __init__(self, host: str = "0.0.0.0", port: int = 5000) -> None:
        self._host    = host
        self._port    = port
        self._app     = None
        self._thread  = None
        self._latest  = {}
        self._lock    = threading.Lock()

    def start(self) -> None:
        try:
            from flask import Flask, Response
        except ImportError:
            log.warning("Flask not installed — falling back to terminal display")
            raise

        app = Flask("spectromeye_display")
        app.logger.disabled = True
        import logging as _lg
        _lg.getLogger("werkzeug").setLevel(_lg.ERROR)

        display = self   # capture for closure

        @app.route("/")
        def index():
            with display._lock:
                bie = dict(display._latest)
            html = display._render(bie) if bie else "<p>Waiting for data...</p>"
            return Response(html, content_type="text/html")

        self._app    = app
        self._thread = threading.Thread(
            target=lambda: app.run(host=self._host, port=self._port, debug=False),
            daemon=True,
        )
        self._thread.start()
        log.info("FlaskDisplay: http://%s:%d", self._host, self._port)
        print(f"[LocalDisplay] Flask server: http://{self._host}:{self._port}")

    def update(self, bie: dict) -> None:
        with self._lock:
            self._latest = bie

    def stop(self) -> None:
        pass   # daemon thread exits with process

    def _render(self, bie: dict) -> str:
        level = bie.get("threat_level", "CLEAR")
        score = bie.get("threat_score", 0)
        cls   = bie.get("signal_class", "Unknown")
        state = bie.get("behavioral_state", "")
        rssi  = bie.get("rssi_dbfs", 0.0)
        slope = bie.get("rssi_slope_dbfs_per_s", 0.0)
        conf  = bie.get("confidence", 0.0) * 100
        interp= bie.get("interpretation", "")
        ts    = time.strftime(
            "%H:%M:%S",
            time.localtime(bie.get("timestamp_ms", 0) / 1000.0),
        )

        env   = bie.get("env_context", {})
        if env and not env.get("data_stale", True) and env.get("temperature_c") is not None:
            env_row = (
                f'<div><span class="label">Env: </span>'
                f'{env["temperature_c"]:.1f}°C &nbsp; {env["humidity_pct"]:.0f}% RH &nbsp; '
                f'PIR={"YES" if env.get("pir_motion") else "no"}</div>'
            )
        else:
            env_row = ""

        rows = ""
        for s in bie.get("active_signals", []):
            rows += (
                f"<tr><td>{s['signal_class']}</td><td>{s['behavioral_state']}</td>"
                f"<td>{s['rssi_dbfs']:.1f}</td><td>{s['active_duration_sec']:.0f}</td></tr>"
            )
        if not rows:
            rows = "<tr><td colspan='4'>no active signals</td></tr>"

        return self._HTML_TEMPLATE.format(
            ts=ts, level=level, score=score, cls=cls, state=state,
            rssi=rssi, slope=slope, conf=conf, interp=interp,
            env_row=env_row, rows=rows,
        )


# ─── FACTORY ──────────────────────────────────────────────────────

class LocalDisplay:
    """
    Unified display facade. Selects backend based on the `backend` argument.

    backend="terminal" — always works, prints to stdout
    backend="flask"    — HTTP server at http://localhost:5000, requires Flask
    """

    def __init__(self, backend: str = "terminal", flask_port: int = 5000) -> None:
        if backend == "flask":
            try:
                self._impl = FlaskDisplay(port=flask_port)
            except ImportError:
                log.warning("Falling back to terminal display")
                self._impl = TerminalDisplay()
        else:
            self._impl = TerminalDisplay()

    def start(self) -> None:
        self._impl.start()

    def update(self, bie_output: dict) -> None:
        self._impl.update(bie_output)

    def stop(self) -> None:
        self._impl.stop()


# ─── SELF-TEST ────────────────────────────────────────────────────

def _run_test() -> None:
    """Feed mock BIE outputs to the terminal display."""
    import time as _time
    display = LocalDisplay(backend="terminal")
    display.start()

    mock_signals = [
        {"signal_class": "LTE",           "behavioral_state": "STATIONARY",      "rssi_dbfs": -75.0, "active_duration_sec": 300},
        {"signal_class": "Walkie_Talkie", "behavioral_state": "APPROACHING_FAST","rssi_dbfs": -52.3, "active_duration_sec": 47},
    ]
    levels = ["CLEAR", "MODERATE", "ELEVATED", "CRITICAL"]

    for i, level in enumerate(levels):
        bie = {
            "threat_level":          level,
            "threat_score":          i * 3,
            "signal_class":          "Walkie_Talkie",
            "behavioral_state":      "APPROACHING_FAST",
            "rssi_dbfs":             -60.0 + i * 3,
            "rssi_slope_dbfs_per_s": 2.1,
            "confidence":            0.93,
            "active_duration_sec":   10 + i * 5,
            "timestamp_ms":          int(_time.time() * 1000),
            "interpretation":        f"Test scenario {i+1}: threat level is {level}.",
            "env_context":           {"temperature_c": 28.4, "humidity_pct": 72.0, "pir_motion": i == 3, "data_stale": False},
            "active_signals":        mock_signals,
        }
        display.update(bie)
        _time.sleep(1.5)

    display.stop()


# ─── ENTRY POINT ──────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SpectrumEye Local Display")
    parser.add_argument("--test",    action="store_true",   help="Run visual self-test")
    parser.add_argument("--backend", default="terminal",    help="terminal | flask")
    args = parser.parse_args()

    if args.test:
        _run_test()
    else:
        parser.print_help()
