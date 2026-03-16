import { useState, useEffect, useRef, useCallback } from "react";

// ─── DESIGN SYSTEM ───────────────────────────────────────────────
const C = {
  bg:          "#030912",
  panel:       "#06101c",
  surface:     "#0a1628",
  border:      "#0d2035",
  radarGreen:  "#00ff41",
  radarDim:    "#003010",
  red:         "#ff3355",
  amber:       "#ffaa00",
  purple:      "#a78bfa",
  cyan:        "#00d4ff",
  text:        "#c8d8e8",
  textDim:     "#4a6080",
  textMuted:   "#2a4060",
  clear:       "#22c55e",
  critical:    "#ff3355",
  elevated:    "#f97316",
  moderate:    "#ffaa00",
};

// ─── SIGNAL DEFINITIONS ──────────────────────────────────────────
const SIGNAL_DEFS = {
  Key_Signal: {
    label: "KEY FOB / REMOTE",
    shortLabel: "Key Fob",
    color: C.red,
    category: "ALERT",
    freq: "433.920 MHz",
    freqNum: 433.92,
  },
  Walkie_Talkie: {
    label: "WALKIE-TALKIE",
    shortLabel: "Walkie-Talkie",
    color: C.amber,
    category: "COMMS",
    freq: "162.000 MHz",
    freqNum: 162.0,
  },
  LTE: {
    label: "LTE CELLULAR",
    shortLabel: "LTE",
    color: C.purple,
    category: "COMMS",
    freq: "2100.000 MHz",
    freqNum: 2100.0,
  },
};

// ─── BIE SENTENCES ───────────────────────────────────────────────
const SENTENCES = {
  Key_Signal: {
    APPEARED:         "A key fob OOK carrier at 433 MHz has appeared in this area — source and intent unknown.",
    APPROACHING_SLOW: "A remote control device is moving closer — recommend visual sweep of perimeter.",
    APPROACHING_FAST: "The device is moving toward this location at speed. Possible remote detonator or vehicle entry device.",
    STATIONARY:       "A key fob signal is active and stable. The remote device appears to be stationary nearby.",
    DEPARTING_SLOW:   "The device is moving away from this location. Continue monitoring until signal is lost.",
    DEPARTING_FAST:   "The key fob signal has dropped sharply. The device is departing this area quickly.",
    DISAPPEARED:      "The key fob signal has been lost. The device has moved out of range or been deactivated.",
  },
  Walkie_Talkie: {
    STATIONARY: "Narrowband FM signal active at 162 MHz. Likely personnel radio in the area.",
    APPEARED:   "A walkie-talkie transmission has been detected at 162 MHz — possible personnel nearby.",
  },
  LTE: {
    STATIONARY: "Normal LTE cellular activity. Signal is stable — no anomalies detected.",
  },
};

function getSentence(cls, state) {
  return SENTENCES[cls]?.[state] || SENTENCES[cls]?.STATIONARY || "Signal detected. Monitoring.";
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
    ctx.fillStyle = "#010d05";
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
      const alpha = t * t * 0.12;
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
      ctx.strokeStyle = "rgba(0,255,65,0.18)";
      ctx.lineWidth = 0.5;
      ctx.stroke();
      // Range label
      ctx.fillStyle = "rgba(0,255,65,0.45)";
      ctx.font = "9px 'JetBrains Mono', monospace";
      ctx.fillText(RING_LABELS[i - 1], cx + 4, cy - r + 12);
    }

    // Radial lines every 30°
    for (let deg = 0; deg < 360; deg += 30) {
      const rad = (deg * Math.PI) / 180;
      ctx.beginPath();
      ctx.moveTo(cx, cy);
      ctx.lineTo(cx + Math.cos(rad) * R, cy + Math.sin(rad) * R);
      ctx.strokeStyle = "rgba(0,255,65,0.12)";
      ctx.lineWidth = 0.5;
      ctx.stroke();
    }

    // Tick marks every 10°
    for (let deg = 0; deg < 360; deg += 10) {
      if (deg % 30 === 0) continue;
      const rad = (deg * Math.PI) / 180;
      const len = 5;
      ctx.beginPath();
      ctx.moveTo(cx + Math.cos(rad) * (R - len), cy + Math.sin(rad) * (R - len));
      ctx.lineTo(cx + Math.cos(rad) * R,         cy + Math.sin(rad) * R);
      ctx.strokeStyle = "rgba(0,255,65,0.25)";
      ctx.lineWidth = 0.5;
      ctx.stroke();
    }

    // Cardinal labels
    const cardinals = [["N", -90], ["E", 0], ["S", 90], ["W", 180]];
    ctx.font = "bold 11px 'JetBrains Mono', monospace";
    for (const [label, deg] of cardinals) {
      const rad = (deg * Math.PI) / 180;
      const lx = cx + Math.cos(rad) * (R - 14);
      const ly = cy + Math.sin(rad) * (R - 14);
      ctx.fillStyle = "rgba(0,255,65,0.7)";
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
    ctx.lineWidth = 1.5;
    ctx.shadowBlur = 6;
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

      // Color based on category
      const def = SIGNAL_DEFS[sig.cls] || {};
      const baseColor = def.category === "ALERT" ? "255,51,85" :
                        def.category === "COMMS" ? (sig.cls === "LTE" ? "167,139,250" : "255,170,0") :
                        "0,255,65";

      // Glow
      const glowSize = 18 + bright * 14;
      const glowGrad = ctx.createRadialGradient(bx, by, 0, bx, by, glowSize);
      glowGrad.addColorStop(0, `rgba(${baseColor},${0.5 * bright})`);
      glowGrad.addColorStop(1, `rgba(${baseColor},0)`);
      ctx.beginPath();
      ctx.arc(bx, by, glowSize, 0, Math.PI * 2);
      ctx.fillStyle = glowGrad;
      ctx.fill();

      // Core dot
      const dotR = 3 + bright * 2;
      ctx.beginPath();
      ctx.arc(bx, by, dotR, 0, Math.PI * 2);
      ctx.fillStyle = `rgba(${baseColor},${0.5 + bright * 0.5})`;
      ctx.shadowBlur = bright > 0.5 ? 8 : 0;
      ctx.shadowColor = `rgb(${baseColor})`;
      ctx.fill();
      ctx.shadowBlur = 0;

      // Label when bright
      if (bright > 0.3) {
        ctx.font = "9px 'JetBrains Mono', monospace";
        ctx.fillStyle = `rgba(${baseColor},${0.4 + bright * 0.6})`;
        ctx.fillText(`${def.shortLabel || sig.cls}  ${sig.rssi}dBm`, bx + 8, by - 5);
      }
    }

    // Center dot
    ctx.beginPath();
    ctx.arc(cx, cy, 3, 0, Math.PI * 2);
    ctx.fillStyle = "rgba(0,255,65,0.8)";
    ctx.fill();

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
        boxShadow: `0 0 32px rgba(0,255,65,0.18), 0 0 60px rgba(0,255,65,0.06)`,
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
    <span style={{ fontFamily: "monospace", letterSpacing: 1 }}>
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
      border: `1px solid ${borderColor}40`,
      borderLeft: `3px solid ${borderColor}`,
      borderRadius: 8,
      padding: "12px 14px",
      marginBottom: 8,
      fontFamily: "'JetBrains Mono', monospace",
      fontSize: 12,
    }}>
      {/* Header */}
      <div style={{ color: headerColor, fontWeight: 700, fontSize: 13, marginBottom: 4 }}>
        {isAlert ? "🔴 ALERT" : "🔵 COMMS"} — {def.label}
      </div>

      {/* Sentence */}
      <div style={{
        color: C.textDim, fontSize: 11, fontStyle: "italic",
        marginBottom: 8, lineHeight: 1.4,
        borderBottom: `1px solid ${C.border}`, paddingBottom: 8,
      }}>
        "{sentence}"
      </div>

      {/* Details grid */}
      <div style={{ display: "flex", flexDirection: "column", gap: 3, color: C.text, fontSize: 11 }}>
        <Row label="Type"      value={def.shortLabel || signal.cls} />
        <Row label="Frequency" value={def.freq || "—"} />
        <div style={{ display: "flex", gap: 0 }}>
          <span style={{ color: C.textDim, width: 80, flexShrink: 0 }}>Strength</span>
          <span><StrengthBar rssi={signal.rssi} />{"  "}{signal.rssi} dBm</span>
        </div>
        <Row label="Trend"     value={trendStr} color={signal.trend >= 0 ? C.red : C.clear} />
        <Row label="Active for" value={fmtTime(signal.activeFor || 0)} />
      </div>

      {/* Footer badges */}
      <div style={{
        display: "flex", gap: 10, marginTop: 8,
        fontSize: 10, color: C.textDim,
      }}>
        <span style={{
          background: `${borderColor}18`, border: `1px solid ${borderColor}30`,
          padding: "2px 7px", borderRadius: 3,
        }}>
          CNN: {((signal.conf || 0.9) * 100).toFixed(0)}% confidence
        </span>
        {signal.bearing != null && (
          <span style={{
            background: `${C.cyan}10`, border: `1px solid ${C.cyan}25`,
            padding: "2px 7px", borderRadius: 3, color: C.cyan,
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
      <span style={{ color: C.textDim, width: 80, flexShrink: 0 }}>{label}</span>
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
      borderRadius: 8, padding: "12px 14px",
      maxHeight: 200, overflowY: "auto",
    }}>
      <div style={{
        fontSize: 11, fontWeight: 700, color: C.radarGreen,
        letterSpacing: 2, marginBottom: 8,
        fontFamily: "'JetBrains Mono', monospace",
      }}>
        ━━━ ALERT LOG ━━━
      </div>
      <div style={{ display: "flex", flexDirection: "column", gap: 3 }}>
        {alerts.map((a, i) => (
          <div key={i} style={{
            display: "flex", gap: 8, alignItems: "flex-start",
            opacity: Math.max(0.3, 1 - i * 0.1),
            fontSize: 11,
            fontFamily: "'JetBrains Mono', monospace",
          }}>
            <span style={{ color: C.textDim, flexShrink: 0 }}>{a.time}</span>
            <span style={{
              width: 7, height: 7, borderRadius: "50%", flexShrink: 0,
              background: levelColors[a.level] || C.textDim,
              marginTop: 2,
            }} />
            <span style={{ color: C.text, fontSize: 11 }}>{a.message}</span>
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
      fontSize: 12, color: C.radarGreen, letterSpacing: 1,
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
    lteRssi:      -72,
    wkRssi:       -81,
    lteActiveFor: 0,
    wkActiveFor:  0,
  });

  function pushAlert(level, message) {
    const time = new Date().toLocaleTimeString("en-GB");
    setAlerts(prev => [{ time, level, message }, ...prev.slice(0, 14)]);
  }

  // Master 400ms tick
  useEffect(() => {
    const id = setInterval(() => {
      const s = simRef.current;
      s.tick++;
      s.lteActiveFor++;
      s.wkActiveFor++;

      // Base signals drift
      s.lteRssi = Math.round(-72 + (Math.random() - 0.5) * 4);
      s.wkRssi  = Math.round(-81 + (Math.random() - 0.5) * 4);

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
          pushAlert("ELEVATED", "Key fob signal detected — OOK carrier at 433 MHz");
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
            APPROACHING_FAST: ["CRITICAL", "⚠ Key fob approaching rapidly — possible remote device"],
            STATIONARY:       ["ELEVATED", "Key fob signal stable — device stationary nearby"],
            DEPARTING_SLOW:   ["MODERATE", "Key fob signal weakening — source departing area"],
            DISAPPEARED:      ["CLEAR",    "Key fob signal lost — area clear"],
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
        id: "lte1", cls: "LTE", state: "STATIONARY",
        rssi: s.lteRssi, conf: 0.96,
        bearing: 210, trend: -1,
        activeFor: s.lteActiveFor,
      },
      {
        id: "walkie1", cls: "Walkie_Talkie", state: "STATIONARY",
        rssi: s.wkRssi, conf: 0.89,
        bearing: 305, trend: 1,
        activeFor: s.wkActiveFor,
      },
    ];

    if (keyVisible) {
      const keyTrend = s.keyRssi - s.prevKeyRssi;
      const keySig = {
        id: "key1", cls: "Key_Signal", state: s.keyState,
        rssi: s.keyRssi, conf: s.keyConf,
        bearing: Math.round(s.keyBearing), trend: keyTrend,
        activeFor: s.keyActiveFor,
      };
      signalsRef.current = [keySig, ...base];
      setCardSignals([keySig, ...base]);
    } else {
      signalsRef.current = base;
      setCardSignals(base);
    }
  }

  // Derived threat
  const keyCard = cardSignals.find(s => s.cls === "Key_Signal");
  const threatLevel = keyCard
    ? (keyCard.state === "APPROACHING_FAST" ? "CRITICAL"
      : keyCard.state === "APPROACHING_SLOW" || keyCard.state === "APPEARED" ? "ELEVATED"
      : keyCard.state === "STATIONARY" ? "ELEVATED"
      : "MODERATE")
    : "CLEAR";
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
              fontSize: 15, fontWeight: 800, letterSpacing: 3,
              fontFamily: "'JetBrains Mono', monospace",
              color: C.radarGreen,
            }}>
              SPECTRUM<span style={{ color: C.cyan }}>EYE</span>
            </div>
            <div style={{ fontSize: 9, color: C.textDim, letterSpacing: 2 }}>
              RF SITUATIONAL AWARENESS · TAINAN, TW
            </div>
          </div>
        </div>

        {/* Center — threat status */}
        <div style={{
          display: "flex", alignItems: "center", gap: 8,
          padding: "6px 16px", borderRadius: 4,
          border: `1px solid ${threatColor}50`,
          background: `${threatColor}10`,
          animation: threatLevel === "CRITICAL" ? "pulseBorder 1.5s ease-in-out infinite" : "none",
        }}>
          <div style={{
            width: 8, height: 8, borderRadius: "50%",
            background: threatColor,
            boxShadow: `0 0 6px ${threatColor}`,
            animation: threatLevel !== "CLEAR" ? "blink 1.2s step-end infinite" : "none",
          }} />
          <span style={{
            fontSize: 11, fontWeight: 700, letterSpacing: 2,
            color: threatColor, fontFamily: "'JetBrains Mono', monospace",
          }}>
            THREAT: {threatLevel}
          </span>
        </div>

        {/* Right — clock + status */}
        <div style={{ display: "flex", alignItems: "center", gap: 16 }}>
          <LiveClock />
          <div style={{
            fontSize: 10, color: C.clear, fontWeight: 600,
            fontFamily: "'JetBrains Mono', monospace", letterSpacing: 1,
          }}>
            ● LIVE
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
          alignItems: "center", justifyContent: "center",
          padding: "24px 20px",
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
                <div style={{ fontSize: 18, fontWeight: 800, color: stat.color }}>
                  {stat.value}
                </div>
                <div style={{ fontSize: 8, color: C.textDim, letterSpacing: 2, marginTop: 2 }}>
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
            fontSize: 11, fontWeight: 700, color: C.radarGreen,
            letterSpacing: 3, fontFamily: "'JetBrains Mono', monospace",
            textAlign: "center", padding: "6px 0",
            borderBottom: `1px solid ${C.radarGreen}30`,
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
            borderRadius: 8, padding: "12px 14px",
            fontFamily: "'JetBrains Mono', monospace", fontSize: 10,
          }}>
            <div style={{
              color: C.radarGreen, fontWeight: 700, letterSpacing: 2,
              marginBottom: 8, fontSize: 11,
            }}>
              ━━━ SYSTEM ━━━
            </div>
            <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
              {[
                ["Edge AI",  "Pi 5 · MobileNetV2",   "SIM"],
                ["SDR",      "RTL-SDR Blog V4",       "SIM"],
                ["Model",    "v2_colab · 3-class",    "LOADED"],
                ["CNN",      "75.6ms · 100% acc.",    "OK"],
                ["Cloud",    "AWS IoT Core",          "SIM"],
              ].map(([lbl, val, st]) => (
                <div key={lbl} style={{
                  display: "flex", justifyContent: "space-between", alignItems: "center",
                  color: C.textDim,
                }}>
                  <span>{lbl}</span>
                  <span style={{ color: C.text }}>{val}</span>
                  <span style={{
                    color: st === "OK" || st === "LOADED" ? C.clear : C.amber,
                    fontSize: 9, letterSpacing: 1,
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
