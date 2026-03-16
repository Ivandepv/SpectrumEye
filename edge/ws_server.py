"""
edge/ws_server.py — WebSocket broadcast server for the React dashboard.

Runs an asyncio WebSocket server in a background thread.
The EdgePipeline calls send(bie_output) after each frame; this server
pushes the enriched payload to every connected React client.

Protocol
--------
Each message is a JSON object with the following shape (sent to client):

{
  "threat_level":  "CLEAR" | "MODERATE" | "ELEVATED" | "CRITICAL",
  "threat_score":  0–10,
  "timestamp_ms":  int,
  "signals": [
    {
      "id":           "Key_Signal",
      "cls":          "Key_Signal",
      "state":        "APPROACHING_FAST",
      "rssi":         -62.0,
      "conf":         0.94,
      "bearing":      52,
      "trend":        3.2,          # dBm/s (positive = getting stronger)
      "activeFor":    45            # seconds
    },
    ...
  ],
  "alert": {                        # only present when alert_fire is True
    "level":   "CRITICAL",
    "message": "..."
  }
}

Usage
-----
    server = WsBroadcastServer(host="localhost", port=8765)
    server.start()          # non-blocking — background thread
    server.send(bie_output) # call from pipeline thread
    server.stop()
"""

import asyncio
import json
import logging
import threading
from typing import Optional

import websockets
import websockets.server

log = logging.getLogger("ws_server")

# ─── DEFAULT BEARINGS (degrees) ──────────────────────────────────
# Used when no real direction-finding hardware is available.
# LTE tower is roughly SW, walkie-talkie is NW, key fob starts NE.
_DEFAULT_BEARING = {
    "LTE":          210,
    "Walkie_Talkie": 305,
    "Key_Signal":    52,
}

# Small per-tick drift (degrees) for realism in sim/demo mode
_BEARING_DRIFT = {
    "LTE":           0.0,   # fixed — tower
    "Walkie_Talkie": 0.5,   # slight movement
    "Key_Signal":    1.5,   # moving device
}


class WsBroadcastServer:
    """Thread-safe WebSocket broadcast server."""

    def __init__(self, host: str = "localhost", port: int = 8765) -> None:
        self._host = host
        self._port = port

        self._loop:    Optional[asyncio.AbstractEventLoop] = None
        self._thread:  Optional[threading.Thread]          = None
        self._clients: set = set()
        self._queue:   Optional[asyncio.Queue]             = None
        self._running  = False

        # Per-class bearing state (bearing drifts over time)
        self._bearings = dict(_DEFAULT_BEARING)

        # Last RSSI per class — used to compute trend on the React side
        # (trend is already in bie_output as rssi_slope_dbfs_per_s)
        self._last_slope: dict[str, float] = {}

    # ── Public API ────────────────────────────────────────────────

    def start(self) -> None:
        """Start the WS server in a background daemon thread."""
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        log.info("WsBroadcastServer: started on ws://%s:%d", self._host, self._port)

    def stop(self) -> None:
        """Signal the server to shut down."""
        self._running = False
        if self._loop and self._loop.is_running():
            self._loop.call_soon_threadsafe(self._loop.stop)

    def send(self, bie_output: dict) -> None:
        """
        Enrich a BIE output dict and broadcast it to all connected clients.
        Safe to call from any thread.
        """
        if not self._loop or not self._queue:
            return
        try:
            payload = self._build_payload(bie_output)
            self._loop.call_soon_threadsafe(self._queue.put_nowait, payload)
        except Exception as exc:
            log.warning("WsBroadcastServer.send error: %s", exc)

    # ── Payload builder ───────────────────────────────────────────

    def _build_payload(self, bie: dict) -> str:
        """Convert a BIE output dict → JSON string for the dashboard."""
        # Update bearing drift for the current signal
        cls = bie.get("signal_class", "")
        if cls in self._bearings:
            drift = _BEARING_DRIFT.get(cls, 0.0)
            self._bearings[cls] = (self._bearings[cls] + drift) % 360

        # Store latest slope per class
        slope = bie.get("rssi_slope_dbfs_per_s", 0.0)
        if cls:
            self._last_slope[cls] = slope

        # Build signals list from active_signals snapshot
        signals = []
        for sig in bie.get("active_signals", []):
            sc = sig["signal_class"]
            signals.append({
                "id":        sc,
                "cls":       sc,
                "state":     sig["behavioral_state"],
                "rssi":      sig["rssi_dbfs"],
                "conf":      sig["confidence"],
                "bearing":   round(self._bearings.get(sc, _DEFAULT_BEARING.get(sc, 0))),
                "trend":     round(self._last_slope.get(sc, 0.0) * 10, 1),  # per 10s
                "activeFor": int(sig["active_duration_sec"]),
            })

        payload: dict = {
            "threat_level":  bie.get("threat_level", "CLEAR"),
            "threat_score":  bie.get("threat_score", 0),
            "timestamp_ms":  bie.get("timestamp_ms", 0),
            "signals":       signals,
        }

        # Include alert when the BIE fires one
        if bie.get("alert_fire"):
            payload["alert"] = {
                "level":   bie["threat_level"],
                "message": bie.get("interpretation", "Signal detected"),
            }

        return json.dumps(payload)

    # ── Asyncio internals ─────────────────────────────────────────

    def _run_loop(self) -> None:
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        self._queue = asyncio.Queue()
        self._running = True
        self._loop.run_until_complete(self._serve())

    async def _serve(self) -> None:
        async with websockets.server.serve(
            self._handle_client,
            self._host,
            self._port,
            ping_interval=20,
            ping_timeout=10,
        ):
            log.info("WsBroadcastServer: listening on ws://%s:%d", self._host, self._port)
            # Broadcast loop
            while self._running:
                try:
                    msg = await asyncio.wait_for(self._queue.get(), timeout=1.0)
                    await self._broadcast(msg)
                except asyncio.TimeoutError:
                    pass
                except Exception as exc:
                    log.warning("WsBroadcastServer broadcast error: %s", exc)

    async def _handle_client(self, ws) -> None:
        log.info("WsBroadcastServer: client connected from %s", ws.remote_address)
        self._clients.add(ws)
        try:
            await ws.wait_closed()
        finally:
            self._clients.discard(ws)
            log.info("WsBroadcastServer: client disconnected")

    async def _broadcast(self, msg: str) -> None:
        if not self._clients:
            return
        dead = set()
        for ws in self._clients:
            try:
                await ws.send(msg)
            except Exception:
                dead.add(ws)
        self._clients -= dead
