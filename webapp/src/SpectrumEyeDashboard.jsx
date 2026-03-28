import { useState, useEffect, useRef, useCallback } from "react";

const WS_URL = "ws://localhost:8765";

// ─── DESIGN SYSTEM ───────────────────────────────────────────────
const C = {
  bg:          "#07111f",
  panel:       "#0b1a2e",
  surface:     "#0f2038",
  border:      "#1e3a58",
  radarGreen:  "#00ff41",
  radarDim:    "#003d14",
  red:         "#ff3355",
  amber:       "#ffbb00",
  purple:      "#b79dfc",
  cyan:        "#00d4ff",
  text:        "#e0eeff",
  textDim:     "#7aaacf",
  textMuted:   "#4a7090",
  clear:       "#2edb6e",
  critical:    "#ff3355",
  elevated:    "#f97316",
  moderate:    "#ffbb00",
};

// ─── SIGNAL DEFINITIONS ──────────────────────────────────────────
// Keys match the CNN class labels exactly (10-class v3 model)
const SIGNAL_DEFS = {
  aircraft_tracking: {
    label: "ADS-B AIRCRAFT",
    shortLabel: "ADS-B",
    color: C.cyan,
    rgb: "0,212,255",
    category: "AIRSPACE",
    freq: "1090.000 MHz",
    freqNum: 1090.0,
  },
  air_traffic: {
    label: "AIR TRAFFIC CONTROL",
    shortLabel: "ATC",
    color: "#00bbff",
    rgb: "0,187,255",
    category: "AIRSPACE",
    freq: "122.000 MHz",
    freqNum: 122.0,
  },
  cellular_network: {
    label: "LTE / 5G CELLULAR",
    shortLabel: "Cellular",
    color: C.purple,
    rgb: "183,157,252",
    category: "COMMS",
    freq: "900.000 MHz",
    freqNum: 900.0,
  },
  local_repeaters: {
    label: "AMATEUR RADIO 2m",
    shortLabel: "Ham Radio",
    color: "#88cc88",
    rgb: "136,204,136",
    category: "COMMS",
    freq: "146.000 MHz",
    freqNum: 146.0,
  },
  maritime: {
    label: "MARITIME VHF",
    shortLabel: "Maritime",
    color: "#00aadd",
    rgb: "0,170,221",
    category: "COMMS",
    freq: "156.800 MHz",
    freqNum: 156.8,
  },
  noaa: {
    label: "NOAA WEATHER SAT",
    shortLabel: "NOAA",
    color: "#66ddff",
    rgb: "102,221,255",
    category: "AIRSPACE",
    freq: "137.500 MHz",
    freqNum: 137.5,
  },
  radio_fm: {
    label: "FM BROADCAST",
    shortLabel: "FM Radio",
    color: C.textDim,
    rgb: "122,170,207",
    category: "COMMS",
    freq: "98.000 MHz",
    freqNum: 98.0,
  },
  short_range_devices: {
    label: "KEY FOB / REMOTE",
    shortLabel: "Short Range",
    color: C.red,
    rgb: "255,51,85",
    category: "ALERT",
    freq: "315.000 MHz",
    freqNum: 315.0,
  },
  walkie_talkie: {
    label: "WALKIE-TALKIE UHF",
    shortLabel: "Walkie-Talkie",
    color: C.amber,
    rgb: "255,187,0",
    category: "ALERT",
    freq: "446.000 MHz",
    freqNum: 446.0,
  },
  wireless_controllers: {
    label: "RC / IOT CONTROLLER",
    shortLabel: "RC/IoT",
    color: C.elevated,
    rgb: "249,115,22",
    category: "ALERT",
    freq: "433.920 MHz",
    freqNum: 433.92,
  },
};

// ─── BIE SENTENCES ───────────────────────────────────────────────
// Covers the three ALERT classes in full; background classes use fallback.
const SENTENCES = {
  short_range_devices: {
    APPEARED:         "A short-range RF device has been detected nearby — key fob, gate remote, or similar transmitter.",
    APPROACHING_SLOW: "Short-range device signal is gradually getting stronger — a person is moving toward this location.",
    APPROACHING_FAST: "Short-range device approaching rapidly — someone carrying a key fob or remote is closing distance fast.",
    STATIONARY:       "Short-range device signal is stable. A key fob or remote is transmitting at a fixed location nearby.",
    DEPARTING_SLOW:   "Short-range device signal is weakening — the person or device is moving away.",
    DEPARTING_FAST:   "Short-range device signal has dropped sharply — the device is departing this area quickly.",
    ERRATIC:          "Short-range device signal is erratic — the device may be in motion or repeatedly activated.",
    DISAPPEARED:      "Short-range device signal lost — the device is no longer in range.",
  },
  walkie_talkie: {
    APPEARED:         "A walkie-talkie transmission detected on UHF — someone in the area is using a two-way radio.",
    APPROACHING_SLOW: "Walkie-talkie signal gradually strengthening — a radio operator is moving toward this location.",
    APPROACHING_FAST: "Walkie-talkie signal approaching rapidly — radio operator closing distance at speed.",
    STATIONARY:       "Walkie-talkie signal active and stable — a person with a two-way radio is nearby.",
    DEPARTING_SLOW:   "Walkie-talkie signal weakening — the radio operator is moving away.",
    DEPARTING_FAST:   "Walkie-talkie signal has dropped sharply — the operator is departing quickly.",
    ERRATIC:          "Walkie-talkie signal fluctuating — operator may be moving between locations.",
    DISAPPEARED:      "Walkie-talkie signal lost — radio no longer transmitting or out of range.",
  },
  wireless_controllers: {
    APPEARED:         "Wireless controller signal detected at 433 MHz — RC transmitter, drone controller, or IoT device active.",
    APPROACHING_SLOW: "Wireless controller signal gradually strengthening — RC operator or drone moving toward this location.",
    APPROACHING_FAST: "Wireless controller approaching rapidly — RC operator closing distance at speed.",
    STATIONARY:       "Wireless controller signal stable — RC transmitter or IoT device operating at a fixed position.",
    DEPARTING_SLOW:   "Wireless controller signal weakening — RC operator moving away.",
    DEPARTING_FAST:   "Wireless controller signal dropped sharply — RC operator departed quickly.",
    ERRATIC:          "Wireless controller signal erratic — RC device may be maneuvering or at range limits.",
    DISAPPEARED:      "Wireless controller signal lost — transmitter no longer active or in range.",
  },
  aircraft_tracking: {
    APPEARED:    "ADS-B transponder detected — aircraft within range overhead at 1090 MHz.",
    STATIONARY:  "ADS-B signal stable — aircraft circling or holding position overhead.",
    DISAPPEARED: "ADS-B signal lost — aircraft has passed out of transponder range.",
  },
  air_traffic: {
    APPEARED:   "ATC VHF communications detected — aircraft or control tower transmitting nearby.",
    STATIONARY: "Steady air traffic control communications — ongoing aircraft-tower exchange.",
  },
  cellular_network: {
    STATIONARY: "Normal LTE/5G cellular activity — stable background signal from tower or device.",
    APPEARED:   "Cellular network signal detected — normal LTE or 5G activity from a nearby tower.",
  },
  local_repeaters: {
    APPEARED:   "Amateur radio signal on 2m band — licensed ham radio operator or repeater active nearby.",
    STATIONARY: "Steady amateur radio transmission — operator active at a fixed location.",
  },
  maritime: {
    APPEARED:   "Maritime VHF radio detected — vessel or shore station communicating.",
    STATIONARY: "Steady maritime VHF transmission — vessel or shore station at stable position.",
  },
  noaa: {
    APPEARED:   "NOAA weather satellite signal detected at 137 MHz — meteorological satellite overhead.",
    STATIONARY: "NOAA satellite signal at peak — satellite near zenith in current orbital pass.",
  },
  radio_fm: {
    STATIONARY: "Stable FM broadcast signal — normal commercial radio transmission from a fixed tower.",
    APPEARED:   "FM broadcast station detected — commercial radio in the 88–108 MHz band is present.",
  },
};

function getSentence(cls, state) {
  return SENTENCES[cls]?.[state] || SENTENCES[cls]?.STATIONARY || `${cls} signal detected. Monitoring.`;
}

// ─── SCENARIO PHASES ─────────────────────────────────────────────
// [phaseName, ticks]
const KEY_PHASES = [
  ["APPEARED",         5],
  ["APPROACHING_SLOW", 10],
  ["APPROACHING_FAST", 10],
  ["STATIONARY",       15],
  ["DEPARTING_SLOW",   8],
  ["DEPARTING_FAST",   6],
  ["DISAPPEARED",      5],
];
const KEY_RSSI_BY_PHASE = {
  APPEARED:         -82,
  APPROACHING_SLOW: -72,
  APPROACHING_FAST: -55,
  STATIONARY:       -41,
  DEPARTING_SLOW:   -55,
  DEPARTING_FAST:   -72,
  DISAPPEARED:      -98,
};

const DJI_PHASES = [
  ["APPEARED",         4],
  ["APPROACHING_SLOW", 6],
  ["APPROACHING_FAST", 5],
  ["STATIONARY",       9],
  ["DEPARTING_SLOW",   5],
  ["DEPARTING_FAST",   4],
  ["DISAPPEARED",      3],
];
const DJI_RSSI_BY_PHASE = {
  APPEARED:         -82,
  APPROACHING_SLOW: -68,
  APPROACHING_FAST: -52,
  STATIONARY:       -44,
  DEPARTING_SLOW:   -66,
  DEPARTING_FAST:   -82,
  DISAPPEARED:      -94,
};

const ADSB_PHASES = [
  ["APPEARED", 4],
  ["OVERHEAD", 10],
  ["DEPARTED", 4],
];
const ADSB_RSSI_BY_PHASE = {
  APPEARED: -85,
  OVERHEAD: -62,
  DEPARTED: -88,
};

// ─── RADAR CANVAS ─────────────────────────────────────────────────
const RADAR_SIZE = 480;

function rssiToDist(rssi, R) {
  const t = Math.max(0, Math.min(1, (-rssi - 30) / 70));
  return (0.12 + t * 0.78) * R;
}

function RadarCanvas({ signalsRef }) {
  const canvasRef  = useRef(null);
  const sweepRef   = useRef(0);
  const rafRef     = useRef(null);

  const draw = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    const W = RADAR_SIZE, H = RADAR_SIZE;
    const cx = W / 2, cy = H / 2;
    const R = W / 2 - 10;

    sweepRef.current = (sweepRef.current + 0.8) % 360;
    const sw = sweepRef.current; // degrees, 0=right, CW

    // Background
    ctx.fillStyle = "#030f08";
    ctx.fillRect(0, 0, W, H);

    // Clip to circle
    ctx.save();
    ctx.beginPath();
    ctx.arc(cx, cy, R, 0, Math.PI * 2);
    ctx.clip();

    // Sweep trail — 48 wedge sectors
    const TRAIL_DEG = 110;
    for (let i = 0; i < 48; i++) {
      const t = i / 48; // 0=tail end, 1=sweep front
      const angleDeg = sw - TRAIL_DEG + t * TRAIL_DEG;
      const a1 = ((angleDeg - 0.5) * Math.PI) / 180;
      const a2 = ((angleDeg + 0.5 + TRAIL_DEG / 48) * Math.PI) / 180;
      const alpha = t * t * 0.18;
      ctx.beginPath();
      ctx.moveTo(cx, cy);
      ctx.arc(cx, cy, R, a1, a2);
      ctx.closePath();
      ctx.fillStyle = `rgba(0,255,65,${alpha})`;
      ctx.fill();
    }

    // Rings
    const RING_LABELS = ["100m", "250m", "500m", "1km"];
    for (let i = 1; i <= 4; i++) {
      const r = (R * i) / 4;
      ctx.beginPath();
      ctx.arc(cx, cy, r, 0, Math.PI * 2);
      ctx.strokeStyle = "rgba(0,255,65,0.30)";
      ctx.lineWidth = 1;
      ctx.stroke();
      // Range label
      ctx.fillStyle = "rgba(0,255,65,0.80)";
      ctx.font = "bold 12px 'JetBrains Mono', monospace";
      ctx.fillText(RING_LABELS[i - 1], cx + 6, cy - r + 14);
    }

    // Radial lines every 30°
    for (let deg = 0; deg < 360; deg += 30) {
      const rad = (deg * Math.PI) / 180;
      ctx.beginPath();
      ctx.moveTo(cx, cy);
      ctx.lineTo(cx + Math.cos(rad) * R, cy + Math.sin(rad) * R);
      ctx.strokeStyle = "rgba(0,255,65,0.22)";
      ctx.lineWidth = 0.8;
      ctx.stroke();
    }

    // Tick marks every 10°
    for (let deg = 0; deg < 360; deg += 10) {
      if (deg % 30 === 0) continue;
      const rad = (deg * Math.PI) / 180;
      const len = 6;
      ctx.beginPath();
      ctx.moveTo(cx + Math.cos(rad) * (R - len), cy + Math.sin(rad) * (R - len));
      ctx.lineTo(cx + Math.cos(rad) * R,         cy + Math.sin(rad) * R);
      ctx.strokeStyle = "rgba(0,255,65,0.40)";
      ctx.lineWidth = 1;
      ctx.stroke();
    }

    // Cardinal labels
    const cardinals = [["N", -90], ["E", 0], ["S", 90], ["W", 180]];
    ctx.font = "bold 14px 'JetBrains Mono', monospace";
    for (const [label, deg] of cardinals) {
      const rad = (deg * Math.PI) / 180;
      const lx = cx + Math.cos(rad) * (R - 16);
      const ly = cy + Math.sin(rad) * (R - 16);
      ctx.fillStyle = "rgba(0,255,65,0.95)";
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      ctx.fillText(label, lx, ly);
    }
    ctx.textAlign = "left";
    ctx.textBaseline = "alphabetic";

    // Sweep line
    const swRad = (sw * Math.PI) / 180;
    const grad = ctx.createLinearGradient(cx, cy, cx + Math.cos(swRad) * R, cy + Math.sin(swRad) * R);
    grad.addColorStop(0, "rgba(0,255,65,0)");
    grad.addColorStop(1, "rgba(0,255,65,0.9)");
    ctx.beginPath();
    ctx.moveTo(cx, cy);
    ctx.lineTo(cx + Math.cos(swRad) * R, cy + Math.sin(swRad) * R);
    ctx.strokeStyle = grad;
    ctx.lineWidth = 2.5;
    ctx.shadowBlur = 10;
    ctx.shadowColor = C.radarGreen;
    ctx.stroke();
    ctx.shadowBlur = 0;

    // Signal blips
    const sigs = signalsRef.current || [];
    for (const sig of sigs) {
      if (!sig.bearing && sig.bearing !== 0) continue;

      // bearing: 0=N, 90=E, 180=S, 270=W → canvas angle: bearing - 90
      const sigCanvasDeg = sig.bearing - 90;
      const sigRad = (sigCanvasDeg * Math.PI) / 180;
      const dist = rssiToDist(sig.rssi, R);
      const bx = cx + Math.cos(sigRad) * dist;
      const by = cy + Math.sin(sigRad) * dist;

      // Phosphor brightness
      const diff = ((sw - sigCanvasDeg) + 360) % 360;
      const bright = diff < 110 ? 1 - diff / 110 : 0.12;

      // Color based on signal definition
      const def = SIGNAL_DEFS[sig.cls] || {};
      const baseColor = def.rgb || "0,255,65";

      // Glow
      const glowSize = 22 + bright * 18;
      const glowGrad = ctx.createRadialGradient(bx, by, 0, bx, by, glowSize);
      glowGrad.addColorStop(0, `rgba(${baseColor},${0.65 * bright})`);
      glowGrad.addColorStop(1, `rgba(${baseColor},0)`);
      ctx.beginPath();
      ctx.arc(bx, by, glowSize, 0, Math.PI * 2);
      ctx.fillStyle = glowGrad;
      ctx.fill();

      // Core dot
      const dotR = 4 + bright * 3;
      ctx.beginPath();
      ctx.arc(bx, by, dotR, 0, Math.PI * 2);
      ctx.fillStyle = `rgba(${baseColor},${0.6 + bright * 0.4})`;
      ctx.shadowBlur = bright > 0.4 ? 14 : 4;
      ctx.shadowColor = `rgb(${baseColor})`;
      ctx.fill();
      ctx.shadowBlur = 0;

      // Label when bright
      if (bright > 0.2) {
        ctx.font = "bold 18px 'JetBrains Mono', monospace";
        ctx.fillStyle = `rgba(${baseColor},${0.5 + bright * 0.5})`;
        ctx.fillText(`${def.shortLabel || sig.cls}  ${sig.rssi}dBm`, bx + 12, by - 8);
      }
    }

    // Center dot
    ctx.beginPath();
    ctx.arc(cx, cy, 4, 0, Math.PI * 2);
    ctx.fillStyle = "rgba(0,255,65,1.0)";
    ctx.shadowBlur = 10;
    ctx.shadowColor = "#00ff41";
    ctx.fill();
    ctx.shadowBlur = 0;

    ctx.restore();

    rafRef.current = requestAnimationFrame(draw);
  }, [signalsRef]);

  useEffect(() => {
    rafRef.current = requestAnimationFrame(draw);
    return () => cancelAnimationFrame(rafRef.current);
  }, [draw]);

  return (
    <canvas
      ref={canvasRef}
      width={RADAR_SIZE}
      height={RADAR_SIZE}
      style={{
        borderRadius: "50%",
        boxShadow: `0 0 40px rgba(0,255,65,0.30), 0 0 80px rgba(0,255,65,0.10), inset 0 0 30px rgba(0,0,0,0.4)`,
        display: "block",
      }}
    />
  );
}

// ─── STRENGTH BAR ─────────────────────────────────────────────────
function StrengthBar({ rssi }) {
  // -30 dBm = full (10 bars), -100 dBm = 0 bars
  const bars = Math.max(0, Math.min(10, Math.round(((rssi + 100) / 70) * 10)));
  return (
    <span style={{ fontFamily: "monospace", letterSpacing: 2, fontSize: 14 }}>
      <span style={{ color: C.radarGreen }}>{"█".repeat(bars)}</span>
      <span style={{ color: C.textMuted }}>{"░".repeat(10 - bars)}</span>
    </span>
  );
}

// ─── FORMAT ACTIVE TIME ───────────────────────────────────────────
function fmtTime(seconds) {
  const h = Math.floor(seconds / 3600);
  const m = Math.floor((seconds % 3600) / 60);
  const s = seconds % 60;
  return `${String(h).padStart(2, "0")}:${String(m).padStart(2, "0")}:${String(s).padStart(2, "0")}`;
}

// ─── SIGNAL CARD ──────────────────────────────────────────────────
function SignalCard({ signal }) {
  const def = SIGNAL_DEFS[signal.cls] || {};
  const isAlert = def.category === "ALERT";
  const sentence = getSentence(signal.cls, signal.state);

  const trendSign = signal.trend >= 0 ? "↑ +" : "↓ ";
  const trendStr  = `${trendSign}${Math.abs(signal.trend || 0).toFixed(0)} dBm / 10s`;

  const borderColor = isAlert ? C.red : def.color;
  const headerColor = isAlert ? C.red : def.color;

  return (
    <div style={{
      background: C.surface,
      border: `1px solid ${borderColor}55`,
      borderLeft: `4px solid ${borderColor}`,
      borderRadius: 8,
      padding: "14px 16px",
      marginBottom: 10,
      fontFamily: "'JetBrains Mono', monospace",
      fontSize: 13,
    }}>
      {/* Header */}
      <div style={{ color: headerColor, fontWeight: 700, fontSize: 15, marginBottom: 6 }}>
        {def.category === "ALERT" ? "● ALERT"
          : def.category === "AIRSPACE" ? "▲ AIRSPACE"
          : "○ COMMS"} — {def.label}
      </div>

      {/* Sentence */}
      <div style={{
        color: C.textDim, fontSize: 12, fontStyle: "italic",
        marginBottom: 10, lineHeight: 1.5,
        borderBottom: `1px solid ${C.border}`, paddingBottom: 10,
      }}>
        "{sentence}"
      </div>

      {/* Details grid */}
      <div style={{ display: "flex", flexDirection: "column", gap: 5, color: C.text, fontSize: 13 }}>
        <Row label="Type"      value={def.shortLabel || signal.cls} />
        <Row label="Frequency" value={def.freq || "—"} />
        <div style={{ display: "flex", gap: 0, alignItems: "center" }}>
          <span style={{ color: C.textDim, width: 90, flexShrink: 0 }}>Strength</span>
          <span style={{ display: "flex", alignItems: "center", gap: 8 }}>
            <StrengthBar rssi={signal.rssi} />
            <span style={{ color: C.text }}>{signal.rssi} dBm</span>
          </span>
        </div>
        <Row label="Trend"     value={trendStr} color={signal.trend >= 0 ? C.red : C.clear} />
        <Row label="Active for" value={fmtTime(signal.activeFor || 0)} />
      </div>

      {/* Footer badges */}
      <div style={{
        display: "flex", gap: 10, marginTop: 10,
        fontSize: 11, color: C.textDim,
      }}>
        <span style={{
          background: `${borderColor}22`, border: `1px solid ${borderColor}50`,
          padding: "3px 9px", borderRadius: 4,
        }}>
          CNN: {((signal.conf || 0.9) * 100).toFixed(0)}% confidence
        </span>
        {signal.bearing != null && (
          <span style={{
            background: `${C.cyan}15`, border: `1px solid ${C.cyan}40`,
            padding: "3px 9px", borderRadius: 4, color: C.cyan,
          }}>
            Bearing: {signal.bearing}°
          </span>
        )}
      </div>
    </div>
  );
}

function Row({ label, value, color }) {
  return (
    <div style={{ display: "flex" }}>
      <span style={{ color: C.textDim, width: 90, flexShrink: 0 }}>{label}</span>
      <span style={{ color: color || C.text }}>{value}</span>
    </div>
  );
}

// ─── ALERT LOG ────────────────────────────────────────────────────
function AlertLog({ alerts }) {
  const levelColors = { CRITICAL: C.red, ELEVATED: C.elevated, MODERATE: C.amber, CLEAR: C.clear };
  return (
    <div style={{
      background: C.surface, border: `1px solid ${C.border}`,
      borderRadius: 8, padding: "14px 16px",
      maxHeight: 220, overflowY: "auto",
    }}>
      <div style={{
        fontSize: 13, fontWeight: 700, color: C.radarGreen,
        letterSpacing: 2, marginBottom: 10,
        fontFamily: "'JetBrains Mono', monospace",
      }}>
        ━━━ ALERT LOG ━━━
      </div>
      <div style={{ display: "flex", flexDirection: "column", gap: 5 }}>
        {alerts.map((a, i) => (
          <div key={i} style={{
            display: "flex", gap: 10, alignItems: "flex-start",
            opacity: Math.max(0.35, 1 - i * 0.09),
            fontSize: 12,
            fontFamily: "'JetBrains Mono', monospace",
          }}>
            <span style={{ color: C.textDim, flexShrink: 0 }}>{a.time}</span>
            <span style={{
              width: 8, height: 8, borderRadius: "50%", flexShrink: 0,
              background: levelColors[a.level] || C.textDim,
              marginTop: 3, boxShadow: i === 0 ? `0 0 6px ${levelColors[a.level]}` : "none",
            }} />
            <span style={{ color: C.text, fontSize: 12 }}>{a.message}</span>
          </div>
        ))}
      </div>
    </div>
  );
}

// ─── LIVE CLOCK ───────────────────────────────────────────────────
function LiveClock() {
  const [t, setT] = useState(new Date());
  useEffect(() => {
    const id = setInterval(() => setT(new Date()), 1000);
    return () => clearInterval(id);
  }, []);
  return (
    <span style={{
      fontFamily: "'JetBrains Mono', monospace",
      fontSize: 14, color: C.radarGreen, letterSpacing: 1,
    }}>
      {t.toLocaleTimeString("en-GB")} UTC+8
    </span>
  );
}

// ─── MAIN DASHBOARD ───────────────────────────────────────────────
// Simulation state (all in ref to avoid re-triggering the RAF draw loop)
const WAIT_TICKS  = 20;  // 8s before first key signal
const LOOP_WAIT   = 16;  // ticks between loops

export default function SpectrumEyeDashboard() {
  // Signals displayed in cards (React state — triggers re-render)
  const [cardSignals, setCardSignals] = useState([]);
  const [alerts, setAlerts] = useState([
    {
      time: new Date().toLocaleTimeString("en-GB"),
      level: "CLEAR",
      message: "System initialized — all signals nominal",
    },
  ]);

  // WebSocket live mode
  const [wsLive, setWsLive] = useState(false);
  const wsLiveRef = useRef(false);
  const [liveThreat, setLiveThreat] = useState(null); // null = use JS-derived

  // Signals passed to radar (via ref — no re-render needed)
  const signalsRef = useRef([]);

  // Sim state (all in ref)
  const simRef = useRef({
    tick:         0,
    waiting:      true,
    waitLeft:     WAIT_TICKS,
    phaseIdx:     0,
    phaseTimer:   0,
    keyActive:    false,
    keyRssi:      -82,
    keyBearing:   52,
    keyConf:      0.94,
    keyState:     "APPEARED",
    keyActiveFor: 0,
    prevKeyRssi:  -82,
    // background always-on signals
    cellRssi:     -72,   // cellular_network
    fmRssi:       -81,   // radio_fm
    cellActiveFor: 0,
    fmActiveFor:  0,
    // aircraft_tracking — periodic flyover
    adsbActive:    false,
    adsbPhaseIdx:  0,
    adsbPhaseTimer: 0,
    adsbRssi:      -88,
    adsbBearing:   310,
    adsbActiveFor: 0,
    adsbCooldown:  22,
    prevAdsbRssi:  -88,
    // wireless_controllers — periodic RC/drone ALERT
    djiActive:     false,
    djiPhaseIdx:   0,
    djiPhaseTimer: 0,
    djiRssi:       -85,
    djiBearing:    80,
    djiActiveFor:  0,
    djiCooldown:   45,
    djiState:      "APPEARED",
    prevDjiRssi:   -85,
  });

  function pushAlert(level, message) {
    const time = new Date().toLocaleTimeString("en-GB");
    setAlerts(prev => [{ time, level, message }, ...prev.slice(0, 14)]);
  }

  // ── WebSocket client — connects to edge/main.py --display ws ────
  useEffect(() => {
    let ws = null;
    let retryTimer = null;

    function connect() {
      ws = new WebSocket(WS_URL);

      ws.onopen = () => {
        wsLiveRef.current = true;
        setWsLive(true);
        const time = new Date().toLocaleTimeString("en-GB");
        setAlerts(prev => [
          { time, level: "CLEAR", message: "● Pipeline connected — receiving live CNN data" },
          ...prev.slice(0, 14),
        ]);
      };

      ws.onmessage = (evt) => {
        let data;
        try { data = JSON.parse(evt.data); } catch { return; }

        // Map ws payload → radar signal objects
        const mapped = (data.signals || []).map(s => ({
          id:        s.id,
          cls:       s.cls,
          state:     s.state,
          rssi:      s.rssi,
          conf:      s.conf,
          bearing:   s.bearing,
          trend:     s.trend,
          activeFor: s.activeFor,
        }));

        signalsRef.current = mapped;
        setCardSignals(mapped);
        if (data.threat_level) setLiveThreat(data.threat_level);

        // Incoming alert from pipeline
        if (data.alert) {
          const time = new Date().toLocaleTimeString("en-GB");
          setAlerts(prev => [
            { time, level: data.alert.level, message: data.alert.message },
            ...prev.slice(0, 14),
          ]);
        }
      };

      ws.onerror = () => {};

      ws.onclose = () => {
        if (wsLiveRef.current) {
          const time = new Date().toLocaleTimeString("en-GB");
          setAlerts(prev => [
            { time, level: "MODERATE", message: "Pipeline disconnected — reverting to simulation" },
            ...prev.slice(0, 14),
          ]);
        }
        wsLiveRef.current = false;
        setWsLive(false);
        setLiveThreat(null);
        // Auto-reconnect after 3s
        retryTimer = setTimeout(connect, 3000);
      };
    }

    connect();
    return () => {
      clearTimeout(retryTimer);
      if (ws) ws.close();
    };
  }, []);

  // Master 400ms tick
  useEffect(() => {
    const id = setInterval(() => {
      // Pause JS simulation when real pipeline is connected
      if (wsLiveRef.current) return;

      const s = simRef.current;
      s.tick++;
      s.cellActiveFor++;
      s.fmActiveFor++;

      // Base background signals drift
      s.cellRssi = Math.round(-72 + (Math.random() - 0.5) * 4);
      s.fmRssi   = Math.round(-81 + (Math.random() - 0.5) * 4);

      // ── ADS-B: periodic aircraft flyover ──────────────────────────
      if (s.adsbActive) {
        s.adsbPhaseTimer++;
        s.adsbActiveFor++;
        const [apName, apDur] = ADSB_PHASES[s.adsbPhaseIdx];
        s.prevAdsbRssi = s.adsbRssi;
        const apTarget = ADSB_RSSI_BY_PHASE[apName];
        // Linear interpolation within phase
        if (apName === "OVERHEAD") {
          const prevTarget = ADSB_RSSI_BY_PHASE[ADSB_PHASES[Math.max(0, s.adsbPhaseIdx - 1)][0]];
          s.adsbRssi = Math.round(prevTarget + (apTarget - prevTarget) * (s.adsbPhaseTimer / apDur) + (Math.random() - 0.5) * 2);
        } else if (apName === "DEPARTED") {
          s.adsbRssi = Math.round(ADSB_RSSI_BY_PHASE.OVERHEAD + (apTarget - ADSB_RSSI_BY_PHASE.OVERHEAD) * (s.adsbPhaseTimer / apDur) + (Math.random() - 0.5) * 2);
        } else {
          s.adsbRssi = Math.round(apTarget + (Math.random() - 0.5) * 2);
        }
        // Bearing drifts — aircraft moving
        s.adsbBearing = (s.adsbBearing + 4 + (Math.random() - 0.5) * 1.5) % 360;
        if (s.adsbPhaseTimer >= apDur) {
          s.adsbPhaseIdx++;
          s.adsbPhaseTimer = 0;
          if (s.adsbPhaseIdx >= ADSB_PHASES.length) {
            s.adsbActive   = false;
            s.adsbPhaseIdx = 0;
            s.adsbCooldown = 30 + Math.floor(Math.random() * 20);
          }
        }
      } else {
        s.adsbCooldown--;
        if (s.adsbCooldown <= 0) {
          s.adsbActive    = true;
          s.adsbPhaseIdx  = 0;
          s.adsbPhaseTimer = 0;
          s.adsbBearing   = Math.floor(Math.random() * 360);
          s.adsbRssi      = -88;
          s.adsbActiveFor = 0;
          pushAlert("MODERATE", "ADS-B transponder detected — aircraft entering local airspace");
        }
      }

      // ── DJI Drone: periodic ALERT ─────────────────────────────────
      if (s.djiActive) {
        s.djiPhaseTimer++;
        s.djiActiveFor++;
        const [dpName, dpDur] = DJI_PHASES[s.djiPhaseIdx];
        s.prevDjiRssi = s.djiRssi;
        const dpTarget = DJI_RSSI_BY_PHASE[dpName];
        s.djiState = dpName;
        if (dpName === "APPROACHING_SLOW" || dpName === "APPROACHING_FAST") {
          const prevTarget = DJI_RSSI_BY_PHASE[DJI_PHASES[Math.max(0, s.djiPhaseIdx - 1)][0]];
          s.djiRssi = Math.round(prevTarget + (dpTarget - prevTarget) * (s.djiPhaseTimer / dpDur) + (Math.random() - 0.5) * 2);
        } else if (dpName === "DEPARTING_SLOW" || dpName === "DEPARTING_FAST") {
          const prevTarget = DJI_RSSI_BY_PHASE[DJI_PHASES[s.djiPhaseIdx - 1][0]];
          s.djiRssi = Math.round(prevTarget + (dpTarget - prevTarget) * (s.djiPhaseTimer / dpDur) + (Math.random() - 0.5) * 2);
        } else {
          s.djiRssi = Math.round(dpTarget + (Math.random() - 0.5) * 2);
        }
        if (dpName !== "STATIONARY") {
          s.djiBearing = (s.djiBearing + (Math.random() - 0.5) * 5 + 360) % 360;
        }
        // Phase transition alerts
        if (s.djiPhaseTimer === 1) {
          const djiAlertMap = {
            APPROACHING_FAST: ["ELEVATED",  "Drone signal strengthening — UAS closing on this location"],
            STATIONARY:       ["CRITICAL",  "⚠ Drone hovering at close range — OcuSync link stable"],
            DEPARTING_SLOW:   ["MODERATE",  "Drone departing — signal weakening"],
            DISAPPEARED:      ["CLEAR",     "DJI drone signal lost — airspace clear"],
          };
          if (djiAlertMap[dpName]) pushAlert(...djiAlertMap[dpName]); // eslint-disable-line no-unused-expressions
        }
        if (s.djiPhaseTimer >= dpDur) {
          s.djiPhaseIdx++;
          s.djiPhaseTimer = 0;
          if (s.djiPhaseIdx >= DJI_PHASES.length) {
            s.djiActive    = false;
            s.djiPhaseIdx  = 0;
            s.djiCooldown  = 55 + Math.floor(Math.random() * 30);
          }
        }
      } else {
        s.djiCooldown--;
        if (s.djiCooldown <= 0) {
          s.djiActive     = true;
          s.djiPhaseIdx   = 0;
          s.djiPhaseTimer = 0;
          s.djiBearing    = Math.floor(Math.random() * 360);
          s.djiRssi       = -85;
          s.djiActiveFor  = 0;
          s.djiState      = "APPEARED";
          pushAlert("ELEVATED", "RC/IoT 433 MHz controller signal detected — possible drone or RC device");
        }
      }

      // Waiting phase
      if (s.waiting) {
        s.waitLeft--;
        if (s.waitLeft <= 0) {
          s.waiting    = false;
          s.phaseIdx   = 0;
          s.phaseTimer = 0;
          s.keyActive  = true;
          s.keyState   = KEY_PHASES[0][0];
          s.keyRssi    = KEY_RSSI_BY_PHASE[KEY_PHASES[0][0]];
          s.keyBearing = 45 + Math.floor(Math.random() * 60);
          s.keyActiveFor = 0;
          pushAlert("ELEVATED", "Short-range device signal detected — key fob or remote at 315 MHz");
        }
        // Update base-only signals for radar
        updateRadarAndCards(s, false);
        return;
      }

      // Active phase
      if (s.keyActive) {
        s.keyActiveFor++;
        const [phaseName, phaseDuration] = KEY_PHASES[s.phaseIdx];
        s.phaseTimer++;

        // Interpolate RSSI within phase
        const targetRssi = KEY_RSSI_BY_PHASE[phaseName];
        s.prevKeyRssi = s.keyRssi;
        s.keyRssi = Math.round(targetRssi + (Math.random() - 0.5) * 3);

        // Trend
        if (phaseName === "APPROACHING_SLOW" || phaseName === "APPROACHING_FAST") {
          s.keyRssi = Math.round(
            KEY_RSSI_BY_PHASE[KEY_PHASES[Math.max(0, s.phaseIdx - 1)][0]] +
            (targetRssi - KEY_RSSI_BY_PHASE[KEY_PHASES[Math.max(0, s.phaseIdx - 1)][0]]) *
            (s.phaseTimer / phaseDuration) + (Math.random() - 0.5) * 2
          );
        }
        if (phaseName === "DEPARTING_SLOW" || phaseName === "DEPARTING_FAST") {
          const prevTarget = KEY_RSSI_BY_PHASE[KEY_PHASES[s.phaseIdx - 1][0]];
          s.keyRssi = Math.round(
            prevTarget + (targetRssi - prevTarget) *
            (s.phaseTimer / phaseDuration) + (Math.random() - 0.5) * 2
          );
        }

        s.keyState = phaseName;
        s.keyConf  = phaseName === "DISAPPEARED" ? 0 : 0.91 + Math.random() * 0.07;

        // Bearing drift
        if (phaseName !== "STATIONARY") {
          s.keyBearing = (s.keyBearing + (Math.random() - 0.5) * 4 + 360) % 360;
        }

        // Phase transition alerts
        if (s.phaseTimer === 1) {
          const alertMap = {
            APPROACHING_FAST: ["CRITICAL", "Short-range device approaching rapidly — possible remote device"],
            STATIONARY:       ["ELEVATED", "Short-range device signal stable — device stationary nearby"],
            DEPARTING_SLOW:   ["MODERATE", "Short-range device signal weakening — source departing area"],
            DISAPPEARED:      ["CLEAR",    "Short-range device signal lost — area clear"],
          };
          if (alertMap[phaseName]) pushAlert(...alertMap[phaseName]);
        }

        // Phase advance
        if (s.phaseTimer >= phaseDuration) {
          s.phaseIdx++;
          s.phaseTimer = 0;
          if (s.phaseIdx >= KEY_PHASES.length) {
            // Cycle complete — wait again
            s.keyActive  = false;
            s.waiting    = true;
            s.waitLeft   = LOOP_WAIT;
            s.phaseIdx   = 0;
          }
        }

        updateRadarAndCards(s, s.keyActive && s.keyState !== "DISAPPEARED");
      }
    }, 400);

    return () => clearInterval(id);
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  function updateRadarAndCards(s, keyVisible) {
    const base = [
      {
        id: "cell1", cls: "cellular_network", state: "STATIONARY",
        rssi: s.cellRssi, conf: 0.96,
        bearing: 200, trend: -1,
        activeFor: s.cellActiveFor,
      },
      {
        id: "fm1", cls: "radio_fm", state: "STATIONARY",
        rssi: s.fmRssi, conf: 0.89,
        bearing: 180, trend: 0,
        activeFor: s.fmActiveFor,
      },
    ];

    const extras = [];

    if (s.adsbActive) {
      extras.push({
        id: "adsb1", cls: "aircraft_tracking",
        state: ADSB_PHASES[s.adsbPhaseIdx]?.[0] === "OVERHEAD" ? "STATIONARY"
             : ADSB_PHASES[s.adsbPhaseIdx]?.[0] === "DEPARTED"  ? "DEPARTING_FAST"
             : "APPEARED",
        rssi: s.adsbRssi, conf: 0.99,
        bearing: Math.round(s.adsbBearing),
        trend: s.adsbRssi - s.prevAdsbRssi,
        activeFor: s.adsbActiveFor,
      });
    }

    if (s.djiActive && s.djiState !== "DISAPPEARED") {
      extras.push({
        id: "rc1", cls: "wireless_controllers", state: s.djiState,
        rssi: s.djiRssi, conf: 0.93,
        bearing: Math.round(s.djiBearing),
        trend: s.djiRssi - s.prevDjiRssi,
        activeFor: s.djiActiveFor,
      });
    }

    if (keyVisible) {
      const keyTrend = s.keyRssi - s.prevKeyRssi;
      const keySig = {
        id: "srd1", cls: "short_range_devices", state: s.keyState,
        rssi: s.keyRssi, conf: s.keyConf,
        bearing: Math.round(s.keyBearing), trend: keyTrend,
        activeFor: s.keyActiveFor,
      };
      signalsRef.current = [keySig, ...extras, ...base];
      setCardSignals([keySig, ...extras, ...base]);
    } else {
      signalsRef.current = [...extras, ...base];
      setCardSignals([...extras, ...base]);
    }
  }

  // Derived threat for JS sim mode — driven by ALERT category signals.
  // In live mode the pipeline sends threat_level directly (handled in ws.onmessage).
  const alertCards = cardSignals.filter(s => SIGNAL_DEFS[s.cls]?.category === "ALERT");
  const criticalStates = ["APPROACHING_FAST", "STATIONARY"];
  const elevatedStates = ["APPEARED", "APPROACHING_SLOW"];
  const isCritical = alertCards.some(s => criticalStates.includes(s.state));
  const isElevated = alertCards.some(s => elevatedStates.includes(s.state));
  const simThreat = isCritical ? "CRITICAL"
    : isElevated ? "ELEVATED"
    : alertCards.length > 0 ? "MODERATE"
    : "CLEAR";
  const threatLevel = liveThreat || simThreat;
  const threatColors = { CRITICAL: C.red, ELEVATED: C.elevated, MODERATE: C.amber, CLEAR: C.clear };
  const threatColor  = threatColors[threatLevel];

  const activeCount  = cardSignals.length;
  const threatCount  = cardSignals.filter(s => SIGNAL_DEFS[s.cls]?.category === "ALERT").length;
  const sortedCards  = [...cardSignals].sort((a, b) => {
    const o = { ALERT: 0, COMMS: 1 };
    return (o[SIGNAL_DEFS[a.cls]?.category] ?? 2) - (o[SIGNAL_DEFS[b.cls]?.category] ?? 2);
  });

  return (
    <div style={{
      minHeight: "100vh", background: C.bg,
      color: C.text,
      fontFamily: "'Segoe UI', system-ui, sans-serif",
      display: "flex", flexDirection: "column",
    }}>
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700;800&display=swap');
        * { box-sizing: border-box; margin: 0; padding: 0; }
        ::-webkit-scrollbar { width: 3px; }
        ::-webkit-scrollbar-track { background: ${C.bg}; }
        ::-webkit-scrollbar-thumb { background: ${C.border}; border-radius: 2px; }
        @keyframes blink { 0%,100%{opacity:1} 50%{opacity:0} }
        @keyframes pulseBorder {
          0%,100% { box-shadow: 0 0 8px ${C.red}40; }
          50%      { box-shadow: 0 0 20px ${C.red}70; }
        }
      `}</style>

      {/* ── HEADER ────────────────────────────────────────────────── */}
      <div style={{
        padding: "10px 24px",
        borderBottom: `1px solid ${C.border}`,
        background: C.panel,
        display: "flex", justifyContent: "space-between", alignItems: "center",
      }}>
        {/* Logo */}
        <div style={{ display: "flex", alignItems: "center", gap: 14 }}>
          <div style={{
            width: 34, height: 34, borderRadius: "50%",
            border: `2px solid ${C.radarGreen}50`,
            display: "flex", alignItems: "center", justifyContent: "center",
            fontSize: 16, background: C.radarDim,
          }}>
            📡
          </div>
          <div>
            <div style={{
              fontSize: 19, fontWeight: 800, letterSpacing: 3,
              fontFamily: "'JetBrains Mono', monospace",
              color: C.radarGreen,
            }}>
              SPECTRUM<span style={{ color: C.cyan }}>EYE</span>
            </div>
            <div style={{ fontSize: 12, color: C.textDim, letterSpacing: 2 }}>
              RF SITUATIONAL AWARENESS · TAINAN, TW
            </div>
          </div>
        </div>

        {/* Center — threat status */}
        <div style={{
          display: "flex", alignItems: "center", gap: 10,
          padding: "8px 20px", borderRadius: 6,
          border: `1px solid ${threatColor}60`,
          background: `${threatColor}15`,
          animation: threatLevel === "CRITICAL" ? "pulseBorder 1.5s ease-in-out infinite" : "none",
        }}>
          <div style={{
            width: 10, height: 10, borderRadius: "50%",
            background: threatColor,
            boxShadow: `0 0 8px ${threatColor}`,
            animation: threatLevel !== "CLEAR" ? "blink 1.2s step-end infinite" : "none",
          }} />
          <span style={{
            fontSize: 14, fontWeight: 700, letterSpacing: 2,
            color: threatColor, fontFamily: "'JetBrains Mono', monospace",
          }}>
            THREAT: {threatLevel}
          </span>
        </div>

        {/* Right — clock + source indicator */}
        <div style={{ display: "flex", alignItems: "center", gap: 16 }}>
          <LiveClock />
          <div style={{
            fontSize: 12, fontWeight: 700,
            fontFamily: "'JetBrains Mono', monospace", letterSpacing: 1,
            color:   wsLive ? C.clear   : C.amber,
            padding: "5px 12px", borderRadius: 4,
            background: wsLive ? `${C.clear}18` : `${C.amber}18`,
            border: `1px solid ${wsLive ? C.clear : C.amber}50`,
          }}>
            {wsLive ? "● LIVE · CNN" : "◌ SIMULATION"}
          </div>
        </div>
      </div>

      {/* ── MAIN ──────────────────────────────────────────────────── */}
      <div style={{
        flex: 1, display: "flex", gap: 0,
        overflow: "hidden",
      }}>

        {/* ── LEFT PANEL (55%) ──────────────────────────────────── */}
        <div style={{
          width: "55%", display: "flex", flexDirection: "column",
          alignItems: "center", justifyContent: "flex-start",
          padding: "32px 20px 24px",
          borderRight: `1px solid ${C.border}`,
          gap: 16,
        }}>
          {/* Radar */}
          <RadarCanvas signalsRef={signalsRef} />

          {/* Stats row below radar */}
          <div style={{
            display: "flex", gap: 0,
            border: `1px solid ${C.border}`, borderRadius: 6,
            overflow: "hidden", width: "100%", maxWidth: RADAR_SIZE,
            fontFamily: "'JetBrains Mono', monospace",
          }}>
            {[
              { label: "ACTIVE",  value: activeCount, color: C.cyan },
              { label: "THREATS", value: threatCount,  color: threatCount > 0 ? C.red : C.clear },
              { label: "RANGE",   value: "1km",        color: C.radarGreen },
              { label: "SWEEP",   value: "360°",       color: C.radarGreen },
            ].map((stat, i) => (
              <div key={i} style={{
                flex: 1, padding: "10px 0", textAlign: "center",
                background: C.surface,
                borderRight: i < 3 ? `1px solid ${C.border}` : "none",
              }}>
                <div style={{ fontSize: 22, fontWeight: 800, color: stat.color }}>
                  {stat.value}
                </div>
                <div style={{ fontSize: 10, color: C.textDim, letterSpacing: 2, marginTop: 3 }}>
                  {stat.label}
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* ── RIGHT PANEL (45%) ─────────────────────────────────── */}
        <div style={{
          width: "45%", display: "flex", flexDirection: "column",
          padding: "16px 16px",
          overflowY: "auto",
          gap: 12,
        }}>

          {/* Section header */}
          <div style={{
            fontSize: 13, fontWeight: 700, color: C.radarGreen,
            letterSpacing: 3, fontFamily: "'JetBrains Mono', monospace",
            textAlign: "center", padding: "8px 0",
            borderBottom: `1px solid ${C.radarGreen}40`,
          }}>
            ━━━ SIGNAL INTELLIGENCE ━━━
          </div>

          {/* Signal cards */}
          {sortedCards.length === 0 ? (
            <div style={{
              color: C.textDim, fontSize: 12, textAlign: "center",
              padding: "20px 0", fontFamily: "'JetBrains Mono', monospace",
            }}>
              Scanning… no signals detected
            </div>
          ) : (
            sortedCards.map(s => <SignalCard key={s.id} signal={s} />)
          )}

          {/* Alert log */}
          <AlertLog alerts={alerts} />

          {/* System panel */}
          <div style={{
            background: C.surface, border: `1px solid ${C.border}`,
            borderRadius: 8, padding: "14px 16px",
            fontFamily: "'JetBrains Mono', monospace", fontSize: 12,
          }}>
            <div style={{
              color: C.radarGreen, fontWeight: 700, letterSpacing: 2,
              marginBottom: 10, fontSize: 13,
            }}>
              ━━━ SYSTEM ━━━
            </div>
            <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
              {[
                ["Edge AI",  "Pi 5 · MobileNetV2",     wsLive ? "LIVE" : "SIM"],
                ["SDR",      "RTL-SDR Blog V4",         wsLive ? "LIVE" : "SIM"],
                ["Model",    "v3_colab · 10-class",     "LOADED"],
                ["CNN",      "97.45% · real RF data",   "OK"],
                ["Cloud",    "AWS IoT Core",            "SIM"],
              ].map(([lbl, val, st]) => (
                <div key={lbl} style={{
                  display: "flex", justifyContent: "space-between", alignItems: "center",
                  color: C.textDim,
                }}>
                  <span>{lbl}</span>
                  <span style={{ color: C.text }}>{val}</span>
                  <span style={{
                    color: st === "OK" || st === "LOADED" ? C.clear : C.amber,
                    fontSize: 11, letterSpacing: 1,
                    background: st === "OK" || st === "LOADED" ? `${C.clear}15` : `${C.amber}15`,
                    padding: "1px 6px", borderRadius: 3,
                  }}>
                    {st}
                  </span>
                </div>
              ))}
            </div>
          </div>

        </div>
      </div>
    </div>
  );
}
