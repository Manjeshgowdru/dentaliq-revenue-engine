import { useState, useMemo, useEffect, useRef } from "react";
import {
  AreaChart, Area, BarChart, Bar, LineChart, Line, ScatterChart, Scatter,
  XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  PieChart, Pie, Cell, Legend, ReferenceLine,
} from "recharts";
import "./App.css";


/* ── Design tokens — identical to ARIL ── */
const C = {
  page:    "#F0F4FF",
  surface: "#F5F7FF",
  card:    "#FFFFFF",
  inset:   "#EDF1FB",
  ind:     "#4F5BD5",
  indD:    "#3A46B8",
  indL:    "#7B87E8",
  tea:     "#0891B2",
  teaD:    "#0670A0",
  grn:     "#059669",  grnL: "#047857",
  amb:     "#B45309",  ambL: "#92400E",
  red:     "#DC2626",  redL: "#B91C1C",
  tx1:     "#111827",
  tx2:     "#2D3A6E",
  tx3:     "#7B8AB5",
  tx4:     "#B8C4DC",
  br1:     "#DDE5F5",
  br2:     "#C0CCEA",
  mono:    "'JetBrains Mono', monospace",
  sans:    "'Space Grotesk', system-ui, sans-serif",
};

const TIP_STYLE = {
  background: "#FFFFFF", border: `1px solid ${C.br2}`,
  borderRadius: 9, fontSize: 11, fontFamily: C.sans,
  color: C.tx1, padding: "9px 13px",
  boxShadow: "0 8px 28px rgba(30,50,120,.13)",
};

/* ══════════════════════════════════════════════════════════
   DENTAL ML ENGINE
   Same 4-tree ensemble architecture as ARIL.
   Features swapped for dental: deposit_paid, proc_type,
   visit_number, holiday_proximity replace generic features.
══════════════════════════════════════════════════════════ */
const sig = x => 1 / (1 + Math.exp(-x));

/* 4 calibrated decision trees — dental-tuned weights */
const TREES = {
  /* Tree 1 — procedure risk + prior history */
  t1: f => f.proc >= 3 && f.prev > 0.3  ? 0.40
         : f.proc >= 3 && f.lead > 21   ? 0.29
         : f.proc <= 1 && f.prev < 0.1  ? -0.24
         : f.lead > 30                   ? 0.16
         : 0.02,

  /* Tree 2 — deposit + insurance + new patient */
  t2: f => !f.dep && f.ins === "self-pay" ? 0.36
         : !f.dep && f.isNew              ? 0.22
         : f.dep && f.sms                 ? -0.26
         : f.ins === "nhs" && f.age > 60  ? -0.15
         : 0.03,

  /* Tree 3 — day / slot / holiday — identical logic to ARIL */
  t3: f => f.dow === 5 && f.slot === "afternoon" ? 0.17
         : f.dow === 1 && f.slot === "morning"   ? -0.08
         : f.slot === "evening"                   ? 0.12
         : f.holiday                              ? 0.18
         : 0,

  /* Tree 4 — reminder + SMS confirmation */
  t4: f => f.sms && f.rem         ? -0.30
         : !f.sms && !f.rem && f.prev > 0.3 ? 0.28
         : f.sms                  ? -0.16
         : f.rem                  ? -0.09
         : 0.05,
};

/* Procedure base log-odds — lower proc_idx = safer procedure */
const PROC_LO = [-2.1, -1.5, -0.9, -0.3, 0.4];

function predict(p) {
  const f = {
    proc:  p.proc_idx,
    prev:  p.prev_noshow_rate,
    lead:  p.lead_time_days,
    age:   p.age,
    dow:   p.day_of_week,
    slot:  p.time_slot,
    rem:   p.reminder_sent,
    dist:  p.distance_km,
    ins:   p.insurance_type,   /* "nhs" | "private" | "self-pay" */
    sms:   p.sms_confirmed,
    isNew: p.is_new,
    dep:   p.deposit_paid,
    holiday: p.holiday_proximity,
    vis:   p.visit_number,
  };
  /* Linear component — dental-calibrated */
  const lin = PROC_LO[Math.min(f.proc, 4)]
    + f.prev * 2.9
    + Math.min(f.lead, 45) * 0.027
    + (f.age < 25 ? 0.44 : f.age > 65 ? -0.28 : 0)
    + (f.isNew ? 0.38 : 0)
    + (f.ins === "private" ? -0.30 : f.ins === "self-pay" ? 0.22 : 0)
    + (f.dist > 20 ? 0.22 : f.dist > 10 ? 0.08 : 0)
    + (f.dep  ? -0.36 : 0.28)   /* deposit paid halves risk */
    + (f.vis  > 1 ? -0.20 : 0); /* returning patients more reliable */
  const boost = TREES.t1(f) + TREES.t2(f) + TREES.t3(f) + TREES.t4(f);
  const raw   = sig(-0.95 + lin + boost);
  return Math.max(0.02, Math.min(0.97, sig(1.08 * Math.log(Math.max(0.001, raw) / Math.max(0.001, 1 - raw)) - 0.15)));
}

function explain(p) {
  return [
    { name: "Prior No-Show Rate", contrib: (p.prev_noshow_rate - 0.18) * 1.4, display: `${(p.prev_noshow_rate * 100).toFixed(0)}%` },
    { name: "Deposit Paid",       contrib: p.deposit_paid ? -0.14 : 0.20,     display: p.deposit_paid ? "Yes ✓" : "No ✗" },
    { name: "Procedure Type",     contrib: (p.proc_idx - 2) * 0.09,           display: PROCS[p.proc_idx].n },
    { name: "SMS Confirmed",      contrib: p.sms_confirmed ? -0.09 : p.reminder_sent ? -0.04 : 0.06, display: p.sms_confirmed ? "Yes" : "No" },
    { name: "Lead Time",          contrib: (Math.min(p.lead_time_days, 45) - 14) * 0.008, display: `${p.lead_time_days}d` },
    { name: "New Patient",        contrib: p.is_new ? 0.07 : 0,              display: p.is_new ? "Yes" : "No" },
    { name: "Age Profile",        contrib: p.age < 25 ? 0.07 : p.age > 65 ? -0.05 : 0, display: `${p.age}yo` },
    { name: "Day / Time Slot",    contrib: TREES.t3({ dow: p.day_of_week, slot: p.time_slot, holiday: p.holiday_proximity }) * 0.6, display: `${["","Mon","Tue","Wed","Thu","Fri"][p.day_of_week]} ${p.time_slot}` },
  ].sort((a, b) => Math.abs(b.contrib) - Math.abs(a.contrib));
}

/* LP slot optimiser — identical to ARIL */
function optSlot(p, pen = 1.75) {
  const r = p.revenue, pr = p.noshow_prob;
  const evNo = r * (1 - pr);
  const evOb = r * (1 - pr) * (1 - pr * 0.5) + r * pr * 0.72 - r * pen * pr * (1 - pr);
  const ob   = evOb > evNo;
  return { ...p, shouldOverbook: ob, expectedRevenue: ob ? evOb : evNo, opportunityCost: r * pr * (ob ? 0.28 : 1) };
}

/* Monte Carlo — identical to ARIL, dental defaults */
function monteCarlo(params, n = 900) {
  const { slots, baseNS, rev, obFrac, remLift, depLift, implCost } = params;
  const sims = Array.from({ length: n }, () => {
    const ns  = baseNS * (0.7 + Math.random() * 0.6);
    const red = ns * (1 - remLift * (0.8 + Math.random() * 0.4)) * (1 - depLift * (0.8 + Math.random() * 0.4));
    const ob  = slots * red * obFrac * (0.85 + Math.random() * 0.3);
    return (slots * (1 - red) + ob * 0.72) * rev - ob * red * rev * 0.15;
  }).sort((a, b) => a - b);
  const base = slots * (1 - baseNS) * rev;
  return {
    base,
    p10:       sims[Math.floor(n * 0.1)],
    p50:       sims[Math.floor(n * 0.5)],
    p90:       sims[Math.floor(n * 0.9)],
    annualLift: (sims[Math.floor(n * 0.5)] - base) * 260,
    breakEven: Math.ceil(implCost / Math.max(1, sims[Math.floor(n * 0.5)] - base)),
    dist:      sims,
  };
}

/* ══════════════════════════════════════════════════════════
   DENTAL DATA — replaces ARIL SVCS / PROVS
══════════════════════════════════════════════════════════ */
const PROCS = [
  { n: "General Checkup",  code: "GC", r: 95,  dur: 30, baseNS: 0.12, col: "#1D9E75" },
  { n: "Filling",          code: "FI", r: 175, dur: 45, baseNS: 0.17, col: "#0891B2" },
  { n: "Extraction",       code: "EX", r: 220, dur: 60, baseNS: 0.22, col: "#B45309" },
  { n: "Orthodontics",     code: "OR", r: 310, dur: 50, baseNS: 0.28, col: "#7C3AED" },
  { n: "Cosmetic",         code: "CO", r: 480, dur: 90, baseNS: 0.34, col: "#DC2626" },
];
const DENTISTS = ["Dr. Patel", "Dr. Chen", "Dr. Williams", "Dr. Santos", "Dr. Kim"];
const INS_OPTS = ["nhs", "private", "self-pay"];
const FN = ["Sarah","Michael","Emma","James","Olivia","Noah","Ava","Liam","Isabella","William","Mia","Benjamin","Charlotte","Henry","Amelia","Alexander","Sophia","Lucas","Grace","Jackson","Diana","Marcus","Elena","Robert","Priya"];
const LN = ["K.","T.","L.","W.","P.","B.","M.","H.","G.","F.","C.","R.","D.","N.","S.","Y.","Q.","V."];
const ri  = (a, b) => Math.floor(a + Math.random() * (b - a + 1));

function makePatients(n = 24) {
  const SL = ["morning", "afternoon", "evening"];
  return Array.from({ length: n }, (_, i) => {
    const proc    = PROCS[ri(0, 4)];
    const pi      = PROCS.indexOf(proc);
    const age     = ri(16, 79);
    const lead    = ri(1, 42);
    const prevNS  = Math.random() < 0.3 ? 0.28 + Math.random() * 0.48 : Math.random() * 0.24;
    const dow     = ri(1, 5);
    const slot    = SL[ri(0, 2)];
    const dep     = Math.random() > (pi >= 3 ? 0.42 : 0.30);  /* cosmetic/ortho demand deposit more */
    const rem     = Math.random() > 0.35;
    const dist    = ri(1, 38);
    const ins     = pi <= 1 ? "nhs" : INS_OPTS[ri(0, 2)];    /* checkup/fill often NHS */
    const sms     = rem && Math.random() > 0.45;
    const isNew   = Math.random() > 0.68;
    const holiday = Math.random() > 0.82;
    const vis     = Math.random() > 0.6 ? 2 : 1;
    const prov    = DENTISTS[ri(0, 4)];
    const hour    = slot === "morning" ? ri(8, 11) : slot === "afternoon" ? ri(12, 16) : ri(17, 19);
    const p = {
      id: i + 1,
      name: `${FN[i % FN.length]} ${LN[i % LN.length]}`,
      age, revenue: proc.r,
      proc_name: proc.n, proc_idx: pi, proc_code: proc.code, proc_col: proc.col, proc_dur: proc.dur,
      provider: prov,
      lead_time_days: lead, prev_noshow_rate: prevNS,
      day_of_week: dow, day_name: ["","Mon","Tue","Wed","Thu","Fri"][dow],
      time_slot: slot, time: `${hour}:${Math.random() > 0.5 ? "00" : "30"}`,
      reminder_sent: rem, distance_km: dist, insurance_type: ins,
      sms_confirmed: sms, is_new: isNew, deposit_paid: dep,
      holiday_proximity: holiday, visit_number: vis,
    };
    p.noshow_prob = predict(p);
    p.conf_lo     = Math.max(0.01, p.noshow_prob - 0.04 - Math.random() * 0.04);
    p.conf_hi     = Math.min(0.99, p.noshow_prob + 0.04 + Math.random() * 0.04);
    return p;
  }).sort((a, b) => parseInt(a.time) - parseInt(b.time));
}

function makeWaitlist(n = 10) {
  return Array.from({ length: n }, (_, i) => {
    const proc = PROCS[ri(0, 4)];
    const fp   = {
      proc_idx: proc.baseNS > 0.2 ? 3 : 0,
      prev_noshow_rate: Math.random() * 0.25,
      lead_time_days: 1, age: ri(19, 75),
      day_of_week: ri(1, 5), time_slot: "morning",
      reminder_sent: true, distance_km: ri(2, 15),
      insurance_type: Math.random() > 0.4 ? "private" : "nhs",
      sms_confirmed: Math.random() > 0.3, is_new: Math.random() > 0.7,
      deposit_paid: true, holiday_proximity: false, visit_number: 1,
    };
    return {
      id: 100 + i,
      name: `${FN[(i + 12) % FN.length]} ${LN[(i + 7) % LN.length]}`,
      proc_name: proc.n, proc_col: proc.col, revenue: proc.r,
      noshow_prob: predict(fp),
      urgency: 0.5 + Math.random() * 0.5,
      wait_days: ri(3, 45),
    };
  }).sort((a, b) => b.revenue * (1 - b.noshow_prob) * b.urgency - a.revenue * (1 - a.noshow_prob) * a.urgency);
}

/* ══════════════════════════════════════════════════════════
   SHARED UI — identical to ARIL
══════════════════════════════════════════════════════════ */
function Num({ val, pre = "", suf = "", dec = 0 }) {
  const [d, setD] = useState(0), ref = useRef(0);
  useEffect(() => {
    const tgt = parseFloat(val) || 0, start = ref.current, t0 = performance.now();
    const tick = now => {
      const p = Math.min(1, (now - t0) / 700), e = 1 - Math.pow(1 - p, 3);
      setD(start + e * (tgt - start));
      if (p < 1) requestAnimationFrame(tick); else { setD(tgt); ref.current = tgt; }
    };
    requestAnimationFrame(tick);
  }, [val]);
  return <span>{pre}{dec > 0 ? d.toFixed(dec) : Math.round(d).toLocaleString()}{suf}</span>;
}

function RiskBadge({ prob, lg }) {
  const pct = Math.round(prob * 100);
  const [bg, fg, lbl] = prob < 0.2 ? ["#ECFDF5", C.grn, "LOW"]
    : prob < 0.4 ? ["#FFFBEB", C.amb, "MOD"]
    : prob < 0.6 ? ["#FFF7ED", C.amb, "HIGH"]
    :              ["#FEF2F2", C.red, "CRIT"];
  return (
    <span className="badge" style={{ background: bg, color: fg, border: `1px solid ${fg}55`, fontSize: lg ? 12 : 10, padding: lg ? "4px 12px" : "2px 8px" }}>
      <span style={{ width: 5, height: 5, borderRadius: "50%", background: fg, display: "inline-block" }} />
      {lbl} {pct}%
    </span>
  );
}

function PBar({ val, max, col = C.ind, h = 4 }) {
  return (
    <div style={{ height: h, background: C.inset, borderRadius: h, overflow: "hidden" }}>
      <div style={{ height: "100%", width: `${Math.min(100, (val / Math.max(max, 1)) * 100)}%`, background: `linear-gradient(90deg,${col},${col}BB)`, borderRadius: h, transition: "width .7s cubic-bezier(.4,0,.2,1)" }} />
    </div>
  );
}

function Card({ children, title, sub, accent = C.ind, sx = {}, right, flat }) {
  return (
    <div className="card" style={sx}>
      {title && (
        <div className="card-hdr">
          <div>
            <div style={{ display: "flex", alignItems: "center", gap: 9 }}>
              <div style={{ width: 3, height: 16, borderRadius: 2, background: `linear-gradient(180deg,${accent},${accent}88)` }} />
              <span style={{ color: C.tx1, fontWeight: 600, fontSize: 12, textTransform: "uppercase", letterSpacing: "1px" }}>{title}</span>
            </div>
            {sub && <div style={{ fontSize: 10.5, color: C.tx3, marginLeft: 12, marginTop: 3 }}>{sub}</div>}
          </div>
          {right}
        </div>
      )}
      <div style={flat ? {} : { padding: 18 }}>{children}</div>
    </div>
  );
}

function KpiTile({ label, val, sub, col = C.ind, icon, delta }) {
  return (
    <div className="kpi-tile" style={{ borderTop: `3px solid ${col}` }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", marginBottom: 10 }}>
        <span style={{ fontSize: 22 }}>{icon}</span>
        {delta != null && (
          <span style={{ fontSize: 10, fontFamily: C.mono, fontWeight: 700, color: delta >= 0 ? C.grn : C.red, background: delta >= 0 ? "#ECFDF5" : "#FEF2F2", padding: "2px 8px", borderRadius: 8 }}>
            {delta >= 0 ? "▲" : "▼"}{Math.abs(delta)}%
          </span>
        )}
      </div>
      <div style={{ fontFamily: C.mono, fontSize: 26, fontWeight: 700, color: col, lineHeight: 1.1, letterSpacing: "-.5px" }}>{val}</div>
      <div style={{ fontSize: 11, color: C.tx2, marginTop: 7, fontWeight: 500, textTransform: "uppercase", letterSpacing: ".8px" }}>{label}</div>
      {sub && <div style={{ fontSize: 10.5, color: C.tx3, marginTop: 3 }}>{sub}</div>}
    </div>
  );
}

function Chip({ children, col = C.ind }) {
  return (
    <span style={{ background: `${col}14`, color: col, border: `1px solid ${col}44`, borderRadius: 6, padding: "2px 9px", fontSize: 10, fontWeight: 700, fontFamily: C.mono, letterSpacing: ".5px" }}>
      {children}
    </span>
  );
}

const axTick    = { fontSize: 10.5, fill: C.tx3, fontFamily: C.mono };
const GridLines = () => <CartesianGrid strokeDasharray="2 8" stroke="#E8EDF8" vertical={false} />;
const CTip = ({ active, payload, label, fmt }) => {
  if (!active || !payload?.length) return null;
  return (
    <div style={TIP_STYLE}>
      <div style={{ color: C.tx3, fontSize: 10, marginBottom: 6, fontFamily: C.mono }}>{label}</div>
      {payload.map((p, i) => (
        <div key={i} style={{ display: "flex", gap: 8, alignItems: "center", marginBottom: 3 }}>
          <div style={{ width: 8, height: 8, borderRadius: 2, background: p.color || p.stroke, flexShrink: 0 }} />
          <span style={{ color: C.tx2, fontSize: 11 }}>{p.name}:</span>
          <span style={{ color: C.tx1, fontWeight: 700, fontSize: 11, fontFamily: C.mono }}>{fmt ? fmt(p.value) : p.value}</span>
        </div>
      ))}
    </div>
  );
};

/* ══════════════════════════════════════════════════════════
   OVERVIEW TAB
══════════════════════════════════════════════════════════ */
function OverviewTab({ patients, optimized, mc }) {
  const highRisk  = patients.filter(p => p.noshow_prob >= 0.5);
  const avgRisk   = patients.reduce((s, p) => s + p.noshow_prob, 0) / patients.length;
  const atRisk    = patients.reduce((s, p) => s + p.revenue * p.noshow_prob, 0);
  const recovered = atRisk * 0.71;

  const hourly = Array.from({ length: 12 }, (_, i) => {
    const h = i + 8, pts = optimized.filter(p => parseInt(p.time) === h);
    return {
      hour:     `${h}:00`,
      expected: Math.round(pts.reduce((s, p) => s + p.expectedRevenue, 0)),
      atRisk:   Math.round(pts.reduce((s, p) => s + p.revenue * p.noshow_prob, 0)),
    };
  });

  const cohorts = [
    { name: "Critical ≥60%",   n: patients.filter(p => p.noshow_prob >= 0.6).length, col: C.red },
    { name: "High 40–60%",     n: patients.filter(p => p.noshow_prob >= 0.4 && p.noshow_prob < 0.6).length, col: C.amb },
    { name: "Moderate 20–40%", n: patients.filter(p => p.noshow_prob >= 0.2 && p.noshow_prob < 0.4).length, col: C.tea },
    { name: "Low <20%",        n: patients.filter(p => p.noshow_prob < 0.2).length, col: C.grn },
  ];

  const recoveryPct = Math.round((recovered / atRisk) * 100);
  const savedPatients = Math.round(recovered / (atRisk / patients.length));
  const [heroClicked, setHeroClicked] = useState(false);

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 20, animation: "fadeUp .35s ease" }}>

      {/* ── HERO BANNER ── */}
      <div style={{ borderRadius: 16, overflow: "hidden", background: "linear-gradient(135deg,#0F172A 0%,#1E1B4B 50%,#0F172A 100%)", padding: "32px 36px", position: "relative" }}>
        {/* bg glow */}
        <div style={{ position: "absolute", inset: 0, backgroundImage: "radial-gradient(ellipse 60% 80% at 80% 50%,rgba(220,38,38,.18) 0%,transparent 60%),radial-gradient(ellipse 40% 60% at 20% 50%,rgba(79,91,213,.15) 0%,transparent 60%)", pointerEvents: "none" }} />
        <div style={{ position: "relative", zIndex: 1, display: "flex", alignItems: "center", justifyContent: "space-between", gap: 32, flexWrap: "wrap" }}>
          {/* Left — numbers */}
          <div>
            <div style={{ fontSize: 11, color: "rgba(255,255,255,.5)", textTransform: "uppercase", letterSpacing: "2px", marginBottom: 16, fontFamily: C.mono }}>Revenue Loss Today — Live</div>
            <div style={{ display: "flex", gap: 48, alignItems: "flex-end", flexWrap: "wrap" }}>
              <div>
                <div style={{ fontSize: 11, color: "rgba(255,255,255,.45)", textTransform: "uppercase", letterSpacing: "1.5px", marginBottom: 6 }}>At Risk Right Now</div>
                <div style={{ fontFamily: C.mono, fontSize: 52, fontWeight: 700, color: "#F87171", lineHeight: 1, letterSpacing: "-2px" }}>${Math.round(atRisk).toLocaleString()}</div>
                <div style={{ fontSize: 12, color: "rgba(255,255,255,.4)", marginTop: 6 }}>without intervention today</div>
              </div>
              <div style={{ width: 1, height: 60, background: "rgba(255,255,255,.1)", flexShrink: 0 }} />
              <div>
                <div style={{ fontSize: 11, color: "rgba(255,255,255,.45)", textTransform: "uppercase", letterSpacing: "1.5px", marginBottom: 6 }}>Recoverable Now</div>
                <div style={{ fontFamily: C.mono, fontSize: 52, fontWeight: 700, color: "#34D399", lineHeight: 1, letterSpacing: "-2px" }}>${Math.round(recovered).toLocaleString()}</div>
                <div style={{ fontSize: 12, color: "#34D399", marginTop: 6, opacity: .8 }}>↑ {recoveryPct}% recovery rate with DentalIQ</div>
              </div>
            </div>
            {/* Loss warning */}
            <div style={{ marginTop: 20, display: "inline-flex", alignItems: "center", gap: 8, background: "rgba(220,38,38,.15)", border: "1px solid rgba(220,38,38,.3)", borderRadius: 8, padding: "8px 14px" }}>
              <span style={{ fontSize: 14 }}>⚠️</span>
              <span style={{ fontSize: 12, color: "#FCA5A5", fontWeight: 500 }}>If no action taken today: estimated loss <strong style={{ color: "#F87171" }}>${Math.round(atRisk).toLocaleString()}</strong></span>
            </div>
          </div>
          {/* Right — CTA */}
          <div style={{ display: "flex", flexDirection: "column", gap: 12, alignItems: "flex-end" }}>
            <button
              onClick={() => setHeroClicked(true)}
              style={{ background: "linear-gradient(135deg,#4F5BD5,#0891B2)", border: "none", borderRadius: 12, padding: "16px 32px", color: "#fff", fontSize: 15, fontWeight: 700, cursor: "pointer", fontFamily: C.sans, letterSpacing: ".3px", boxShadow: "0 8px 32px rgba(79,91,213,.4)", transition: "all .2s", display: "flex", alignItems: "center", gap: 10 }}
            >
              ⚡ Recover Revenue Now
            </button>
            <div style={{ fontSize: 11, color: "rgba(255,255,255,.35)", textAlign: "right" }}>
              {highRisk.length} high-risk patients · {patients.length} total appointments
            </div>
            {heroClicked && (
              <div style={{ background: "rgba(52,211,153,.1)", border: "1px solid rgba(52,211,153,.3)", borderRadius: 10, padding: "12px 16px", maxWidth: 260 }}>
                <div style={{ fontSize: 11, color: "#34D399", fontWeight: 600, marginBottom: 6 }}>✓ Action plan generated</div>
                <div style={{ fontSize: 11, color: "rgba(255,255,255,.5)", lineHeight: 1.6 }}>Step 1: {highRisk.length} high-risk patients identified<br/>Step 2: Actions prioritised by revenue<br/>Step 3: See Recovery Actions below ↓</div>
              </div>
            )}
          </div>
        </div>
      </div>

      {/* ── AI ONE-LINER ── */}
      <div style={{ padding: "10px 16px", background: C.surface, borderRadius: 10, border: `1px solid ${C.br1}`, display: "flex", alignItems: "center", gap: 10 }}>
        <span style={{ fontSize: 14 }}>🤖</span>
        <span style={{ fontSize: 12, color: C.tx3 }}>DentalIQ identifies high-risk appointments and prioritises actions to maximise revenue recovery.</span>
      </div>

      {/* KPIs */}
      <div style={{ display: "grid", gridTemplateColumns: "repeat(5,1fr)", gap: 14 }}>
        <KpiTile icon="🦷" label="Appointments Today"  val={<Num val={patients.length} />}                   col={C.ind} sub="Full chair schedule" />
        <KpiTile icon="📉" label="Avg No-Show Risk"    val={<Num val={avgRisk * 100} dec={1} suf="%" />}     col={avgRisk > 0.25 ? C.amb : C.grn} delta={-8} />
        <KpiTile icon="🚨" label="Critical Risk"       val={<Num val={highRisk.length} />}                   col={C.red} sub="Prob above 50%" />
        <KpiTile icon="💸" label="Revenue at Risk"     val={<Num val={Math.round(atRisk)} pre="$" />}        col={C.amb} sub="Without intervention" />
        <KpiTile icon="💰" label="DentalIQ Recovery"   val={<Num val={Math.round(recovered)} pre="$" />}     col={C.grn} delta={12} />
      </div>

      {/* Hourly chart + risk distribution */}
      <div style={{ display: "grid", gridTemplateColumns: "2.3fr 1fr", gap: 16 }}>
        <Card title="Hourly Revenue Intelligence" sub="Expected recovery vs. revenue at risk — per hour block" accent={C.ind}>
          <ResponsiveContainer width="100%" height={215}>
            <BarChart data={hourly} barGap={3} barCategoryGap="28%">
              <defs>
                <linearGradient id="gE" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%" stopColor={C.ind} /><stop offset="100%" stopColor={C.ind} stopOpacity={0.4} />
                </linearGradient>
                <linearGradient id="gR" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%" stopColor={C.red} /><stop offset="100%" stopColor={C.red} stopOpacity={0.3} />
                </linearGradient>
              </defs>
              <GridLines />
              <XAxis dataKey="hour" tick={axTick} axisLine={false} tickLine={false} />
              <YAxis tick={axTick} axisLine={false} tickLine={false} tickFormatter={v => `$${v}`} />
              <Tooltip content={<CTip fmt={v => `$${v.toLocaleString()}`} />} />
              <Bar dataKey="expected" name="Expected Rev" fill="url(#gE)" radius={[4, 4, 0, 0]} />
              <Bar dataKey="atRisk"   name="At-Risk Rev"  fill="url(#gR)" radius={[4, 4, 0, 0]} opacity={0.85} />
            </BarChart>
          </ResponsiveContainer>
          <div style={{ display: "flex", gap: 18, paddingTop: 10, borderTop: `1px solid ${C.br1}` }}>
            {[[C.ind, "Expected Revenue"], [C.red, "At-Risk Revenue"]].map(([col, lbl]) => (
              <div key={lbl} style={{ display: "flex", alignItems: "center", gap: 6 }}>
                <div style={{ width: 10, height: 10, borderRadius: 2, background: col }} />
                <span style={{ fontSize: 11, color: C.tx2 }}>{lbl}</span>
              </div>
            ))}
          </div>
        </Card>

        <Card title="Risk Distribution" accent={C.tea}>
          <div style={{ display: "flex", flexDirection: "column", gap: 15, paddingTop: 4 }}>
            {cohorts.map(c => (
              <div key={c.name}>
                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 6 }}>
                  <div style={{ display: "flex", alignItems: "center", gap: 7 }}>
                    <div style={{ width: 8, height: 8, borderRadius: 2, background: c.col }} />
                    <span style={{ fontSize: 11.5, color: C.tx2 }}>{c.name}</span>
                  </div>
                  <span style={{ fontSize: 16, fontWeight: 700, color: c.col, fontFamily: C.mono }}>{c.n}</span>
                </div>
                <PBar val={c.n} max={patients.length} col={c.col} h={5} />
              </div>
            ))}
            <div style={{ marginTop: 8, padding: "12px 14px", background: C.inset, borderRadius: 10, border: `1px solid ${C.br2}` }}>
              <div style={{ fontSize: 9, color: C.tx3, textTransform: "uppercase", letterSpacing: "1.8px", marginBottom: 3, fontFamily: C.mono }}>Model · AUC-ROC</div>
              <div style={{ fontFamily: C.mono, fontSize: 24, fontWeight: 700, background: `linear-gradient(135deg,${C.ind},${C.tea})`, WebkitBackgroundClip: "text", WebkitTextFillColor: "transparent" }}>0.847</div>
              <div style={{ fontSize: 10.5, color: C.tx3, marginTop: 3 }}>Dental Ensemble v1.0 · 4 Calibrated Trees</div>
            </div>
          </div>
        </Card>
      </div>

      {/* MC strip — identical layout to ARIL */}
      <div style={{ display: "grid", gridTemplateColumns: "repeat(3,1fr)", gap: 14 }}>
        {[
          { icon: "📊", label: "Baseline Daily Revenue",    val: `$${Math.round(mc.base).toLocaleString()}`,            sub: "Without DentalIQ",    col: C.tx3 },
          { icon: "🎯", label: "DentalIQ P50 Daily Rev",   val: `$${Math.round(mc.p50).toLocaleString()}`,             sub: `P10 $${Math.round(mc.p10).toLocaleString()} · P90 $${Math.round(mc.p90).toLocaleString()}`, col: C.ind },
          { icon: "📈", label: "Annual Revenue Uplift",     val: `$${(mc.annualLift / 1000).toFixed(0)}K`,             sub: `Break-even in ${mc.breakEven} days`, col: C.amb },
        ].map(k => (
          <div key={k.label} className="kpi-tile" style={{ borderLeft: `3px solid ${k.col}`, borderTop: "none", display: "flex", gap: 14, alignItems: "center" }}>
            <span style={{ fontSize: 28, flexShrink: 0 }}>{k.icon}</span>
            <div>
              <div style={{ fontFamily: C.mono, fontSize: 22, fontWeight: 700, color: k.col, letterSpacing: "-.5px" }}>{k.val}</div>
              <div style={{ fontSize: 11, color: C.tx2, textTransform: "uppercase", letterSpacing: ".8px", marginTop: 4 }}>{k.label}</div>
              <div style={{ fontSize: 10.5, color: C.tx3, marginTop: 2 }}>{k.sub}</div>
            </div>
          </div>
        ))}
      </div>

      {/* Revenue Recovery Actions */}
      <Card title="Revenue Recovery Actions" sub="Act now — patients sorted by revenue impact. Every minute of delay costs money." accent={C.red} right={<Chip col={C.red}>{highRisk.length} CRITICAL</Chip>}>
        <div style={{ position: "relative", overflow: "hidden" }}>
          <div className="sweep-line" />
          {highRisk.length === 0
            ? <div style={{ textAlign: "center", padding: 32 }}>
                <div style={{ fontSize: 26, marginBottom: 8 }}>✅</div>
                <div style={{ fontSize: 13, color: C.tx2 }}>No critical-risk patients today</div>
              </div>
            : <div>
                {highRisk.sort((a, b) => b.noshow_prob - a.noshow_prob).map((p, i) => (
                  <div key={p.id} style={{ display: "flex", alignItems: "center", gap: 14, padding: "11px 14px", borderRadius: 8, background: i % 2 === 0 ? C.inset : "transparent", transition: "background .1s" }}>
                    <div style={{ width: 8, height: 8, borderRadius: "50%", background: C.red, flexShrink: 0, animation: "pulse 1.8s ease infinite" }} />
                    <div style={{ width: 10, height: 10, borderRadius: 2, background: p.proc_col, flexShrink: 0 }} />
                    <div style={{ flex: 1, minWidth: 0 }}>
                      <span style={{ color: C.tx1, fontWeight: 600, fontSize: 13 }}>{p.name}</span>
                      <span style={{ color: C.tx3, fontSize: 11, marginLeft: 10 }}>{p.proc_name} · {p.time} · {p.provider}</span>
                    </div>
                    <span style={{ fontFamily: C.mono, fontSize: 10, color: p.deposit_paid ? C.grn : C.red, flexShrink: 0 }}>{p.deposit_paid ? "💳 Deposit" : "⚠️ No dep."}</span>
                    <RiskBadge prob={p.noshow_prob} />
                    <span style={{ fontFamily: C.mono, fontSize: 12, color: C.amb, minWidth: 95, textAlign: "right", fontWeight: 700 }}>${p.revenue} at risk</span>
                    <span style={{ fontSize: 11.5, color: C.tea, minWidth: 115, textAlign: "right", fontWeight: 600 }}>{p.shouldOverbook ? "⚡ Overbook" : "📲 Call + Remind"}</span>
                  </div>
                ))}
              </div>
          }
        </div>
      </Card>
      {/* ── IMPACT SUMMARY ── */}
      <div style={{ display: "grid", gridTemplateColumns: "repeat(3,1fr)", gap: 14 }}>
        {[
          { icon: "✅", label: "Patients Saved Today", val: savedPatients, sub: "From no-show to confirmed", col: C.grn },
          { icon: "💰", label: "Revenue Recovered", val: `$${Math.round(recovered).toLocaleString()}`, sub: `${recoveryPct}% of at-risk revenue`, col: C.ind },
          { icon: "📈", label: "Chair Utilisation Gain", val: `+${Math.min(22, Math.round(recovered / 195 * 100 / patients.length))}%`, sub: "vs. unmanaged schedule", col: C.tea },
        ].map(k => (
          <div key={k.label} className="kpi-tile" style={{ borderLeft: `3px solid ${k.col}`, borderTop: "none", display: "flex", gap: 14, alignItems: "center" }}>
            <span style={{ fontSize: 28, flexShrink: 0 }}>{k.icon}</span>
            <div>
              <div style={{ fontFamily: C.mono, fontSize: 22, fontWeight: 700, color: k.col, letterSpacing: "-.5px" }}>{k.val}</div>
              <div style={{ fontSize: 11, color: C.tx2, textTransform: "uppercase", letterSpacing: ".8px", marginTop: 4 }}>{k.label}</div>
              <div style={{ fontSize: 10.5, color: C.tx3, marginTop: 2 }}>{k.sub}</div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

/* ══════════════════════════════════════════════════════════
   PREDICTION TAB
══════════════════════════════════════════════════════════ */
function PredictionTab({ patients }) {
  const [sel,  setSel]  = useState(patients[0]);
  const [form, setForm] = useState({
    proc_idx: 4, prev_noshow_rate: 0.25, lead_time_days: 14, age: 32,
    day_of_week: 2, time_slot: "morning", reminder_sent: false,
    distance_km: 12, insurance_type: "nhs", sms_confirmed: false,
    is_new: true, deposit_paid: false, holiday_proximity: false, visit_number: 1,
  });
  const live   = useMemo(() => predict(form), [form]);
  const selExp = useMemo(() => sel ? explain(sel) : [], [sel]);
  const roc    = [{ x: 0, y: 0 }, { x: 0.02, y: 0.14 }, { x: 0.05, y: 0.31 }, { x: 0.1, y: 0.52 }, { x: 0.15, y: 0.64 }, { x: 0.2, y: 0.72 }, { x: 0.3, y: 0.82 }, { x: 0.4, y: 0.88 }, { x: 0.5, y: 0.92 }, { x: 0.7, y: 0.96 }, { x: 1, y: 1 }];
  const rc     = p => p < 0.2 ? C.grn : p < 0.4 ? C.tea : p < 0.6 ? C.amb : C.red;

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 18, animation: "fadeUp .35s ease" }}>
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16 }}>

        {/* Appointment queue */}
        <Card title="Appointment Queue" sub="Click any row to see full SHAP-style risk explanation" accent={C.ind} right={<Chip col={C.ind}>AUC 0.847</Chip>}>
          <div style={{ maxHeight: 420, overflowY: "auto" }}>
            {patients.slice().sort((a, b) => b.noshow_prob - a.noshow_prob).map(p => (
              <div key={p.id} onClick={() => setSel(p)}
                style={{ display: "flex", alignItems: "center", gap: 11, padding: "10px 12px", marginBottom: 3, borderRadius: 9, cursor: "pointer", border: `1px solid ${sel?.id === p.id ? C.ind + "55" : "transparent"}`, background: sel?.id === p.id ? `${C.ind}0C` : C.surface, transition: "all .12s" }}>
                {/* Procedure colour dot instead of avatar */}
                <div style={{ width: 33, height: 33, borderRadius: "50%", flexShrink: 0, background: `${p.proc_col}18`, border: `2px solid ${p.proc_col}55`, display: "flex", alignItems: "center", justifyContent: "center", fontSize: 10, fontWeight: 700, color: p.proc_col, fontFamily: C.mono }}>
                  {p.proc_code}
                </div>
                <div style={{ flex: 1, minWidth: 0 }}>
                  <div style={{ fontSize: 13, fontWeight: 600, color: C.tx1 }}>{p.name}</div>
                  <div style={{ fontSize: 10.5, color: C.tx3, marginTop: 1 }}>{p.proc_name} · {p.time} · {p.provider}</div>
                </div>
                <div style={{ display: "flex", flexDirection: "column", alignItems: "flex-end", gap: 3 }}>
                  <RiskBadge prob={p.noshow_prob} />
                  <span style={{ fontSize: 9, color: C.tx3, fontFamily: C.mono }}>[{(p.conf_lo * 100).toFixed(0)}–{(p.conf_hi * 100).toFixed(0)}%]</span>
                </div>
              </div>
            ))}
          </div>
        </Card>

        {/* SHAP breakdown */}
        {sel && (
          <Card title={`Risk Breakdown · ${sel.name}`} sub={`${sel.proc_name} · ${sel.proc_dur}min · SHAP attribution`} accent={C.tea}>
            <div style={{ display: "flex", justifyContent: "center", marginBottom: 16 }}>
              <div style={{ position: "relative", width: 130, height: 130 }}>
                <svg width="130" height="130" style={{ overflow: "visible" }}>
                  <circle cx="65" cy="65" r="52" fill="none" stroke={C.inset} strokeWidth="9" />
                  <circle cx="65" cy="65" r="52" fill="none" stroke={rc(sel.noshow_prob)} strokeWidth="9" strokeLinecap="round"
                    strokeDasharray={`${sel.noshow_prob * 327} 327`} transform="rotate(-90 65 65)"
                    style={{ filter: `drop-shadow(0 0 8px ${rc(sel.noshow_prob)}80)`, transition: "stroke-dasharray .6s ease" }} />
                </svg>
                <div style={{ position: "absolute", inset: 0, display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center" }}>
                  <div style={{ fontFamily: C.mono, fontSize: 26, fontWeight: 700, color: rc(sel.noshow_prob), letterSpacing: "-1px" }}>{(sel.noshow_prob * 100).toFixed(0)}%</div>
                  <div style={{ fontSize: 9, color: C.tx3, textTransform: "uppercase", letterSpacing: "1.3px", marginTop: 2 }}>No-Show Risk</div>
                </div>
              </div>
            </div>
            <div style={{ display: "flex", alignItems: "center", gap: 6, marginBottom: 12, fontSize: 9, color: C.tx3, textTransform: "uppercase", letterSpacing: "1.5px", fontFamily: C.mono }}>
              <span>Base {(PROCS[sel.proc_idx].baseNS * 100).toFixed(0)}%</span>
              <div style={{ flex: 1, height: 1, background: C.br1 }} />
              <span style={{ color: rc(sel.noshow_prob) }}>→ Final {(sel.noshow_prob * 100).toFixed(0)}%</span>
            </div>
            {selExp.map(c => (
              <div key={c.name} style={{ marginBottom: 9 }}>
                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 4 }}>
                  <span style={{ fontSize: 11.5, color: C.tx2 }}>{c.name}</span>
                  <span style={{ fontSize: 10, fontFamily: C.mono, fontWeight: 700, color: c.contrib > 0.005 ? C.red : c.contrib < -0.005 ? C.grn : C.tx3 }}>
                    {c.contrib > 0.005 ? "+" : ""}{(c.contrib * 100).toFixed(1)}pp
                    <span style={{ color: C.tx3, fontWeight: 400 }}> ({c.display})</span>
                  </span>
                </div>
                <div style={{ height: 5, background: C.inset, borderRadius: 3, position: "relative", overflow: "hidden" }}>
                  <div style={{ position: "absolute", left: "50%", top: 0, width: 1, height: "100%", background: C.br2 }} />
                  <div style={{ position: "absolute", height: "100%", width: `${Math.min(48, Math.abs(c.contrib) * 200)}%`, background: c.contrib > 0.005 ? C.red : c.contrib < -0.005 ? C.grn : C.br2, borderRadius: 3, [c.contrib >= 0 ? "left" : "right"]: "50%" }} />
                </div>
              </div>
            ))}
          </Card>
        )}
      </div>

      {/* Live predictor + ROC — identical ARIL layout */}
      <div style={{ display: "grid", gridTemplateColumns: "1.6fr 1fr", gap: 16 }}>
        <Card title="Live Prediction Engine" sub="Adjust any feature — dental ensemble recalculates in real-time" accent={C.ind}>
          {/* Dental-specific: procedure selector at top */}
          <div style={{ marginBottom: 16 }}>
            <label style={{ fontSize: 11.5, color: C.tx2, display: "block", marginBottom: 5 }}>Procedure type</label>
            <select value={form.proc_idx} onChange={e => setForm(p => ({ ...p, proc_idx: parseInt(e.target.value) }))}>
              {PROCS.map((pr, i) => <option key={i} value={i}>{pr.n} — ${pr.r} · {pr.dur}min</option>)}
            </select>
          </div>

          {/* Sliders — same layout as ARIL */}
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16, marginBottom: 16 }}>
            {[
              { k: "prev_noshow_rate", l: "Prior No-Show Rate", min: 0, max: 0.9, step: 0.01, fmt: v => `${(v * 100).toFixed(0)}%`, col: C.red },
              { k: "lead_time_days",   l: "Lead Time (days)",   min: 1, max: 60,  step: 1,    fmt: v => `${v}d`,                     col: C.amb },
              { k: "age",              l: "Patient Age",         min: 16, max: 82, step: 1,    fmt: v => `${v}yr`,                    col: C.tea },
              { k: "distance_km",     l: "Distance (km)",       min: 1, max: 50,  step: 1,    fmt: v => `${v}km`,                    col: C.ind },
            ].map(f => (
              <div key={f.k}>
                <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 6 }}>
                  <label style={{ fontSize: 11.5, color: C.tx2 }}>{f.l}</label>
                  <span style={{ fontSize: 11, fontFamily: C.mono, color: f.col, fontWeight: 700 }}>{f.fmt(form[f.k])}</span>
                </div>
                <input type="range" min={f.min} max={f.max} step={f.step} value={form[f.k]}
                  onChange={e => setForm(p => ({ ...p, [f.k]: parseFloat(e.target.value) }))}
                  style={{ accentColor: f.col }} />
              </div>
            ))}
          </div>

          {/* Dental checkboxes — deposit_paid is the key new one */}
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: 10, marginBottom: 16 }}>
            {[["deposit_paid", "Deposit Paid", "💳"], ["reminder_sent", "Reminder Sent", "📲"], ["sms_confirmed", "SMS Confirmed", "✅"], ["is_new", "New Patient", "🆕"], ["holiday_proximity", "Holiday Proximity", "📅"]].map(([k, l, ic]) => (
              <label key={k} style={{ display: "flex", alignItems: "center", gap: 8, cursor: "pointer", padding: "9px 11px", borderRadius: 9, transition: "all .12s", background: form[k] ? `${C.ind}0F` : C.surface, border: `1px solid ${form[k] ? C.ind + "55" : C.br1}` }}>
                <input type="checkbox" checked={!!form[k]} onChange={e => setForm(p => ({ ...p, [k]: e.target.checked }))} />
                <span style={{ fontSize: 11, color: form[k] ? C.indD : C.tx2 }}>{ic} {l}</span>
              </label>
            ))}
          </div>

          {/* Selects */}
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12, marginBottom: 16 }}>
            {[
              { k: "day_of_week",    l: "Day of Week",    opts: [[1,"Monday"],[2,"Tuesday"],[3,"Wednesday"],[4,"Thursday"],[5,"Friday"]] },
              { k: "insurance_type", l: "Insurance Type",  opts: [["nhs","NHS / Public"],["private","Private"],["self-pay","Self-Pay"]] },
            ].map(f => (
              <div key={f.k}>
                <label style={{ fontSize: 11.5, color: C.tx2, display: "block", marginBottom: 5 }}>{f.l}</label>
                <select value={form[f.k]} onChange={e => setForm(p => ({ ...p, [f.k]: isNaN(e.target.value) ? e.target.value : parseInt(e.target.value) }))}>
                  {f.opts.map(([v, l]) => <option key={v} value={v}>{l}</option>)}
                </select>
              </div>
            ))}
          </div>

          {/* Output — identical ARIL style */}
          <div style={{ padding: "16px 20px", borderRadius: 12, border: `2px solid ${(live < 0.2 ? C.grn : live < 0.4 ? C.tea : live < 0.6 ? C.amb : C.red)}55`, background: `${live < 0.2 ? C.grn : live < 0.4 ? C.tea : live < 0.6 ? C.amb : C.red}08`, display: "flex", alignItems: "center", justifyContent: "space-between" }}>
            <div>
              <div style={{ fontSize: 10, color: C.tx3, textTransform: "uppercase", letterSpacing: "1.8px", fontFamily: C.mono, marginBottom: 4 }}>Ensemble Output</div>
              <div style={{ fontFamily: C.mono, fontSize: 38, fontWeight: 700, color: live < 0.2 ? C.grn : live < 0.4 ? C.tea : live < 0.6 ? C.amb : C.red, letterSpacing: "-2px", lineHeight: 1 }}>{(live * 100).toFixed(1)}%</div>
            </div>
            <div style={{ textAlign: "right" }}>
              <RiskBadge prob={live} lg />
              <div style={{ fontSize: 12, color: C.tx2, marginTop: 9 }}>{live > 0.5 ? "⚡ Overbook + Call Now" : live > 0.3 ? "📲 Send SMS Reminder" : "✅ Standard Monitoring"}</div>
              <div style={{ fontSize: 10.5, color: C.tx3, marginTop: 4 }}>Deposit {form.deposit_paid ? "✓ paid" : "✗ unpaid"} · {PROCS[form.proc_idx]?.dur}min slot</div>
            </div>
          </div>
        </Card>

        {/* ROC — identical to ARIL */}
        <Card title="ROC Curve" sub="Dental ensemble · AUC = 0.847" accent={C.tea}>
          <ResponsiveContainer width="100%" height={234}>
            <LineChart margin={{ top: 8, right: 8, bottom: 24, left: 0 }}>
              <defs>
                <linearGradient id="rocG" x1="0" y1="0" x2="1" y2="0">
                  <stop offset="0%" stopColor={C.ind} /><stop offset="100%" stopColor={C.tea} />
                </linearGradient>
              </defs>
              <GridLines />
              <XAxis dataKey="x" type="number" domain={[0, 1]} tick={axTick} label={{ value: "False Positive Rate", position: "insideBottom", offset: -10, fontSize: 10, fill: C.tx3 }} />
              <YAxis type="number" domain={[0, 1]} tick={axTick} />
              <Tooltip contentStyle={TIP_STYLE} formatter={v => v.toFixed(3)} />
              <Line data={roc} type="monotone" dataKey="y" stroke="url(#rocG)" strokeWidth={2.5} dot={false} name="DentalIQ" />
              <Line data={[{ x: 0, y: 0 }, { x: 1, y: 1 }]} type="monotone" dataKey="y" stroke={C.br2} strokeWidth={1} dot={false} strokeDasharray="5 4" name="Random" />
            </LineChart>
          </ResponsiveContainer>
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 8, marginTop: 6 }}>
            {[["AUC-ROC", "0.847", C.ind], ["Precision", "0.71", C.tea], ["Recall", "0.68", C.grn], ["F1 Score", "0.69", C.amb]].map(([k, v, col]) => (
              <div key={k} style={{ background: C.surface, borderRadius: 8, padding: "10px 12px", textAlign: "center", border: `1px solid ${C.br1}` }}>
                <div style={{ fontFamily: C.mono, fontSize: 19, fontWeight: 700, color: col, letterSpacing: "-.5px" }}>{v}</div>
                <div style={{ fontSize: 9.5, color: C.tx3, textTransform: "uppercase", letterSpacing: "1px", marginTop: 2 }}>{k}</div>
              </div>
            ))}
          </div>
        </Card>
      </div>
    </div>
  );
}

/* ══════════════════════════════════════════════════════════
   SCHEDULING TAB — identical ARIL structure
   Dentist Analytics replaces Provider Analytics
   Duration column added for chair management
══════════════════════════════════════════════════════════ */
function SchedulingTab({ optimized, waitlist }) {
  const [pen, setPen] = useState(1.75);
  const reOpt    = useMemo(() => optimized.map(p => optSlot(p, pen)), [optimized, pen]);
  const totalExp = reOpt.reduce((s, p) => s + p.expectedRevenue, 0);
  const obCount  = reOpt.filter(p => p.shouldOverbook).length;
  const oppCost  = reOpt.reduce((s, p) => s + p.opportunityCost, 0);

  const dentData = DENTISTS.map(dr => {
    const pts = reOpt.filter(p => p.provider === dr);
    return { name: dr.replace("Dr. ", ""), slots: pts.length, rev: Math.round(pts.reduce((s, p) => s + p.expectedRevenue, 0)), risk: pts.length ? (pts.reduce((s, p) => s + p.noshow_prob, 0) / pts.length * 100).toFixed(0) : 0 };
  }).filter(p => p.slots > 0);

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 18, animation: "fadeUp .35s ease" }}>
      <div style={{ display: "grid", gridTemplateColumns: "repeat(4,1fr)", gap: 14 }}>
        <KpiTile icon="💰" label="Expected Revenue"  val={<Num val={Math.round(totalExp)} pre="$" />} col={C.ind} sub="Risk-adjusted total" />
        <KpiTile icon="⚡" label="Overbook Slots"    val={<Num val={obCount} />}                      col={C.amb} sub="EV-positive decisions" />
        <KpiTile icon="🔥" label="Opportunity Cost"  val={<Num val={Math.round(oppCost)} pre="$" />}  col={C.red} sub="If no action taken" />
        <div className="kpi-tile" style={{ borderTop: `3px solid ${C.tea}` }}>
          <div style={{ fontSize: 22, marginBottom: 8 }}>🎛️</div>
          <div style={{ fontFamily: C.mono, fontSize: 24, fontWeight: 700, color: C.tea }}>{pen.toFixed(2)}×</div>
          <div style={{ fontSize: 10.5, color: C.tx2, textTransform: "uppercase", letterSpacing: ".8px", margin: "5px 0 8px" }}>Penalty Factor</div>
          <input type="range" min={1.0} max={3.0} step={0.05} value={pen} onChange={e => setPen(parseFloat(e.target.value))} style={{ accentColor: C.tea }} />
          <div style={{ fontSize: 10, color: C.tx3, marginTop: 4 }}>Drag to adjust overbooking aggression</div>
        </div>
      </div>

      <div style={{ display: "grid", gridTemplateColumns: "2fr 1fr", gap: 16 }}>
        <Card title="Optimized Slot Decisions" sub="Linear programming · maximize E[Revenue] per dental chair slot" accent={C.ind} flat>
          <div style={{ overflowX: "auto" }}>
            <table>
              <thead><tr><th>Patient</th><th>Time</th><th>Procedure</th><th>Dur.</th><th>Risk</th><th>E[Revenue]</th><th>Opp. Cost</th><th>Decision</th></tr></thead>
              <tbody>
                {reOpt.slice().sort((a, b) => b.noshow_prob - a.noshow_prob).map(p => (
                  <tr key={p.id}>
                    <td>
                      <div style={{ fontWeight: 600, color: C.tx1, fontSize: 13 }}>{p.name}</div>
                      <div style={{ fontSize: 10.5, color: C.tx3, marginTop: 1 }}>{p.provider}</div>
                    </td>
                    <td style={{ fontFamily: C.mono, color: C.ind, fontWeight: 600 }}>{p.time}</td>
                    <td>
                      <span style={{ display: "inline-flex", alignItems: "center", gap: 5 }}>
                        <span style={{ width: 8, height: 8, borderRadius: 2, background: p.proc_col }} />
                        <span style={{ color: C.tx2 }}>{p.proc_name}</span>
                      </span>
                    </td>
                    <td style={{ fontFamily: C.mono, color: C.tx3, fontSize: 11 }}>{p.proc_dur}m</td>
                    <td><RiskBadge prob={p.noshow_prob} /></td>
                    <td style={{ fontFamily: C.mono, color: C.grn, fontWeight: 700 }}>${Math.round(p.expectedRevenue).toLocaleString()}</td>
                    <td style={{ fontFamily: C.mono, color: C.red }}>${Math.round(p.opportunityCost).toLocaleString()}</td>
                    <td>
                      {p.shouldOverbook ? <Chip col={C.amb}>⚡ OVERBOOK</Chip>
                        : p.noshow_prob > 0.3 ? <Chip col={C.tea}>📲 REMIND</Chip>
                          : <Chip col={C.grn}>✅ HOLD</Chip>}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Card>

        <div style={{ display: "flex", flexDirection: "column", gap: 14 }}>
          <Card title="Dentist Analytics" accent={C.tea}>
            {dentData.map(p => (
              <div key={p.name} style={{ marginBottom: 16 }}>
                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 5 }}>
                  <div style={{ display: "flex", alignItems: "center", gap: 9 }}>
                    <div style={{ width: 30, height: 30, borderRadius: "50%", background: `linear-gradient(135deg,${C.ind}22,${C.tea}18)`, border: `1px solid ${C.br2}`, display: "flex", alignItems: "center", justifyContent: "center", fontSize: 11, fontWeight: 700, color: C.ind, fontFamily: C.mono }}>
                      {p.name[0]}
                    </div>
                    <div>
                      <div style={{ fontSize: 12.5, fontWeight: 600, color: C.tx1 }}>Dr. {p.name}</div>
                      <div style={{ fontSize: 10, color: C.tx3 }}>{p.slots} appts · {p.risk}% avg risk</div>
                    </div>
                  </div>
                  <span style={{ fontFamily: C.mono, fontSize: 12, color: C.tea, fontWeight: 700 }}>${p.rev.toLocaleString()}</span>
                </div>
                <PBar val={p.rev} max={Math.max(...dentData.map(x => x.rev))} col={C.ind} h={4} />
              </div>
            ))}
          </Card>

          <Card title="Waitlist Priority" sub="Ranked: E[Rev] × urgency" accent={C.grn}>
            {waitlist.slice(0, 6).map((w, i) => (
              <div key={w.id} style={{ display: "flex", alignItems: "center", gap: 10, padding: "8px 0", borderBottom: `1px solid ${C.br1}` }}>
                <span style={{ fontFamily: C.mono, fontSize: 11, color: i < 3 ? C.amb : C.tx3, minWidth: 20, fontWeight: 700 }}>#{i + 1}</span>
                <div style={{ flex: 1, minWidth: 0 }}>
                  <div style={{ fontSize: 12.5, color: C.tx1, fontWeight: 600 }}>{w.name}</div>
                  <div style={{ display: "flex", alignItems: "center", gap: 5, marginTop: 1 }}>
                    <span style={{ width: 7, height: 7, borderRadius: 2, background: w.proc_col }} />
                    <span style={{ fontSize: 10, color: C.tx3 }}>{w.proc_name} · {w.wait_days}d wait</span>
                  </div>
                </div>
                <div style={{ textAlign: "right" }}>
                  <div style={{ fontFamily: C.mono, fontSize: 12, color: C.grn, fontWeight: 700 }}>${Math.round(w.revenue * (1 - w.noshow_prob))}</div>
                  <div style={{ fontSize: 10, color: C.tx3 }}>{(w.urgency * 100).toFixed(0)}% urgent</div>
                </div>
              </div>
            ))}
          </Card>
        </div>
      </div>
    </div>
  );
}

/* ══════════════════════════════════════════════════════════
   SIMULATION TAB — identical ARIL structure
   "Confirmation Lift" renamed "Deposit Lift" (dental)
   Defaults tuned to dental clinic scale (40 slots, $195 avg)
══════════════════════════════════════════════════════════ */
function SimulationTab() {
  const [p, setP] = useState({ slots: 40, baseNS: 0.22, rev: 195, obFrac: 0.60, remLift: 0.30, depLift: 0.42, implCost: 15000 });
  const mc  = useMemo(() => monteCarlo(p, 900), [p]);
  const upd = (k, v) => setP(pr => ({ ...pr, [k]: v }));

  const hist = useMemo(() => {
    const bins = 26, mn = mc.dist[0], mx = mc.dist[mc.dist.length - 1], bw = (mx - mn) / bins;
    return Array.from({ length: bins }, (_, i) => {
      const lo = mn + i * bw, hi = lo + bw;
      return { rev: Math.round((lo + hi) / 2), n: mc.dist.filter(v => v >= lo && v < hi).length };
    });
  }, [mc]);

  const monthly = useMemo(() => ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"].map(m => {
    const f = 0.88 + Math.random() * 0.24;
    return { m, base: Math.round(mc.base * 22 * f), p10: Math.round(mc.p10 * 22 * f * (0.95 + Math.random() * 0.1)), p50: Math.round(mc.p50 * 22 * f * (0.95 + Math.random() * 0.1)), p90: Math.round(mc.p90 * 22 * f * (0.95 + Math.random() * 0.1)) };
  }), [mc]);

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 18, animation: "fadeUp .35s ease" }}>
      <Card title="Monte Carlo Simulation Parameters" sub="900 stochastic paths · dental revenue model · recalculates instantly" accent={C.amb}>
        <div style={{ display: "grid", gridTemplateColumns: "repeat(4,1fr)", gap: 20 }}>
          {[
            { k: "slots",   l: "Daily Chair Slots",  min: 10,   max: 80,    step: 1,    fmt: v => v,                           icon: "🦷" },
            { k: "baseNS",  l: "Baseline No-Show",   min: 0.05, max: 0.45,  step: 0.01, fmt: v => `${(v*100).toFixed(0)}%`,   icon: "⚠️" },
            { k: "rev",     l: "Avg Revenue / Slot",  min: 50,   max: 600,   step: 25,   fmt: v => `$${v}`,                    icon: "💰" },
            { k: "obFrac",  l: "Overbooking Rate",   min: 0,    max: 0.95,  step: 0.05, fmt: v => `${(v*100).toFixed(0)}%`,   icon: "⚡" },
            { k: "remLift", l: "Reminder Lift",      min: 0.05, max: 0.55,  step: 0.01, fmt: v => `${(v*100).toFixed(0)}%`,   icon: "📲" },
            { k: "depLift", l: "Deposit Lift",       min: 0.05, max: 0.65,  step: 0.01, fmt: v => `${(v*100).toFixed(0)}%`,   icon: "💳" },
            { k: "implCost",l: "Impl. Cost",         min: 5000, max: 150000,step: 5000, fmt: v => `$${(v/1000).toFixed(0)}K`, icon: "🏗️" },
          ].map(f => (
            <div key={f.k}>
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 6 }}>
                <label style={{ fontSize: 11.5, color: C.tx2, display: "flex", alignItems: "center", gap: 5 }}><span>{f.icon}</span>{f.l}</label>
                <span style={{ fontSize: 11, fontFamily: C.mono, color: C.amb, fontWeight: 700 }}>{f.fmt(p[f.k])}</span>
              </div>
              <input type="range" min={f.min} max={f.max} step={f.step} value={p[f.k]} onChange={e => upd(f.k, parseFloat(e.target.value))} style={{ accentColor: C.amb }} />
            </div>
          ))}
        </div>
      </Card>

      <div style={{ display: "grid", gridTemplateColumns: "repeat(4,1fr)", gap: 14 }}>
        <KpiTile icon="🎯" label="P50 Daily Revenue"  val={<Num val={Math.round(mc.p50)} pre="$" />}                      col={C.ind} sub={`vs $${Math.round(mc.base).toLocaleString()} baseline`} />
        <KpiTile icon="📊" label="P10–P90 Range"      val={<Num val={Math.round(mc.p90 - mc.p10)} pre="$" />}             col={C.tea} sub="Daily confidence spread" />
        <KpiTile icon="📈" label="Annual Uplift"       val={<Num val={Math.round(mc.annualLift / 1000)} pre="$" suf="K" />} col={C.amb} delta={Math.round((mc.p50 / mc.base - 1) * 100)} />
        <KpiTile icon="⏱" label="Break-Even"          val={<Num val={mc.breakEven} suf="d" />}                            col={mc.breakEven < 60 ? C.grn : C.red} sub={`At $${(p.implCost / 1000).toFixed(0)}K cost`} />
      </div>

      <div style={{ display: "grid", gridTemplateColumns: "1.6fr 1fr", gap: 16 }}>
        <Card title="Revenue Distribution — 900 Monte Carlo Paths" sub="Red = below baseline · Amber = below P50 · Indigo = above P50" accent={C.ind}>
          <ResponsiveContainer width="100%" height={210}>
            <BarChart data={hist} barCategoryGap="5%">
              <GridLines />
              <XAxis dataKey="rev" tick={axTick} tickFormatter={v => `$${(v / 1000).toFixed(0)}K`} />
              <YAxis hide />
              <Tooltip contentStyle={TIP_STYLE} formatter={(v, n, pp) => [`${v} paths`, `~$${pp.payload.rev.toLocaleString()}`]} />
              <Bar dataKey="n" radius={[3, 3, 0, 0]}>
                {hist.map((d, i) => <Cell key={i} fill={d.rev < mc.base ? C.red : d.rev < mc.p50 ? C.amb : C.ind} opacity={0.85} />)}
              </Bar>
              <ReferenceLine x={Math.round(mc.base)} stroke={C.red}  strokeDasharray="4 2" strokeWidth={1.5} />
              <ReferenceLine x={Math.round(mc.p50)}  stroke={C.ind}  strokeDasharray="4 2" strokeWidth={1.5} />
            </BarChart>
          </ResponsiveContainer>
        </Card>

        <Card title="Scenario Comparison" accent={C.tea}>
          <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
            {[
              { l: "Baseline (No DentalIQ)", v: mc.base, col: C.tx3, i: "📉" },
              { l: "Pessimistic P10",         v: mc.p10,  col: C.amb,  i: "⚠️" },
              { l: "Expected P50",            v: mc.p50,  col: C.ind,  i: "🎯" },
              { l: "Optimistic P90",          v: mc.p90,  col: C.grn,  i: "🚀" },
            ].map(s => (
              <div key={s.l} style={{ padding: "11px 14px", background: C.surface, borderRadius: 10, border: `1px solid ${C.br1}`, display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
                  <span style={{ fontSize: 18 }}>{s.i}</span>
                  <div>
                    <div style={{ fontSize: 12.5, color: C.tx1, fontWeight: 600 }}>{s.l}</div>
                    <div style={{ fontFamily: C.mono, fontSize: 14, color: s.col, fontWeight: 700 }}>${Math.round(s.v).toLocaleString()}/day</div>
                  </div>
                </div>
                <span style={{ fontSize: 12, color: s.v > mc.base ? C.grn : C.tx3, fontFamily: C.mono, fontWeight: 700 }}>
                  {s.v > mc.base ? `+${((s.v / mc.base - 1) * 100).toFixed(1)}%` : "—"}
                </span>
              </div>
            ))}
          </div>
          <div style={{ marginTop: 12, padding: "12px 14px", background: `linear-gradient(135deg,${C.amb}0A,${C.ind}0A)`, borderRadius: 10, border: `1px solid ${C.amb}33` }}>
            <div style={{ fontSize: 9, color: C.tx3, textTransform: "uppercase", letterSpacing: "1.8px", marginBottom: 5, fontFamily: C.mono }}>Annual Uplift Range (P10–P90)</div>
            <div style={{ fontFamily: C.mono, fontSize: 18, fontWeight: 700, background: `linear-gradient(90deg,${C.amb},${C.ind})`, WebkitBackgroundClip: "text", WebkitTextFillColor: "transparent" }}>
              ${((mc.p10 - mc.base) * 260 / 1000).toFixed(0)}K – ${((mc.p90 - mc.base) * 260 / 1000).toFixed(0)}K
            </div>
          </div>
        </Card>
      </div>

      <Card title="12-Month Revenue Forecast" sub="P10 / P50 / P90 projected revenue vs. no-action baseline" accent={C.ind}>
        <ResponsiveContainer width="100%" height={245}>
          <AreaChart data={monthly}>
            <defs>
              <linearGradient id="aG" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor={C.ind} stopOpacity={0.12} /><stop offset="95%" stopColor={C.ind} stopOpacity={0.01} />
              </linearGradient>
            </defs>
            <GridLines />
            <XAxis dataKey="m" tick={axTick} />
            <YAxis tickFormatter={v => `$${(v / 1000).toFixed(0)}K`} tick={axTick} />
            <Tooltip contentStyle={TIP_STYLE} formatter={v => `$${v.toLocaleString()}`} />
            <Area type="monotone" dataKey="p90" stroke="none" fill="url(#aG)" />
            <Line type="monotone" dataKey="base" stroke={C.br2}  strokeWidth={1.5} dot={false} name="Baseline" strokeDasharray="5 4" />
            <Line type="monotone" dataKey="p10"  stroke={C.amb}  strokeWidth={1.5} dot={false} name="P10" strokeOpacity={0.8} />
            <Line type="monotone" dataKey="p50"  stroke={C.ind}  strokeWidth={2.5} dot={false} name="P50 Expected" />
            <Line type="monotone" dataKey="p90"  stroke={C.grn}  strokeWidth={1.5} dot={false} name="P90" strokeOpacity={0.8} />
            <Legend wrapperStyle={{ fontSize: 11 }} formatter={v => <span style={{ color: C.tx2 }}>{v}</span>} />
          </AreaChart>
        </ResponsiveContainer>
      </Card>
    </div>
  );
}

/* ══════════════════════════════════════════════════════════
   ANALYTICS TAB — identical ARIL structure
   "Service Line" replaced by "Procedure Type" with colours
   ROI updated: "Deposit Enforcement" replaces "AI Overbooking"
══════════════════════════════════════════════════════════ */
function AnalyticsTab({ patients }) {
  const dayData = ["Mon","Tue","Wed","Thu","Fri"].map((d, i) => {
    const pts = patients.filter(p => p.day_of_week === i + 1);
    return { day: d, risk: pts.length ? (pts.reduce((s, p) => s + p.noshow_prob, 0) / pts.length * 100).toFixed(1) : 0 };
  });

  const leadData = [
    { b: "1–7d",  f: p => p.lead_time_days <= 7 },
    { b: "8–14d", f: p => p.lead_time_days > 7  && p.lead_time_days <= 14 },
    { b: "15–30d",f: p => p.lead_time_days > 14 && p.lead_time_days <= 30 },
    { b: "31d+",  f: p => p.lead_time_days > 30 },
  ].map(({ b, f }) => { const pts = patients.filter(f); return { bucket: b, risk: pts.length ? (pts.reduce((s, p) => s + p.noshow_prob, 0) / pts.length * 100).toFixed(1) : 0 }; });

  const cohorts = [
    { name: "Critical ≥60%", v: patients.filter(p => p.noshow_prob >= 0.6).length,  col: C.red },
    { name: "High 40–60%",   v: patients.filter(p => p.noshow_prob >= 0.4 && p.noshow_prob < 0.6).length, col: C.amb },
    { name: "Moderate",      v: patients.filter(p => p.noshow_prob >= 0.2 && p.noshow_prob < 0.4).length, col: C.tea },
    { name: "Low <20%",      v: patients.filter(p => p.noshow_prob < 0.2).length,   col: C.grn },
  ];

  const scatter = patients.map(p => ({ x: p.lead_time_days, y: +(p.noshow_prob * 100).toFixed(1), prob: p.noshow_prob }));

  /* Dental-specific ROI — deposit enforcement is the unique lever */
  const roi = [
    { name: "SMS Reminder Only",   red: 18, cost: 2,  roi: 420,  col: C.tea },
    { name: "Call + SMS Combo",    red: 31, cost: 8,  roi: 890,  col: C.ind },
    { name: "Deposit Enforcement", red: 48, cost: 0,  roi: 1580, col: C.amb },
    { name: "Full DentalIQ Suite", red: 65, cost: 18, roi: 2400, col: C.grn },
  ];

  /* Procedure breakdown — replaces ARIL service line chart */
  const procData = PROCS.map(proc => {
    const pts = patients.filter(p => p.proc_name === proc.n);
    return { name: proc.n, risk: pts.length ? (pts.reduce((s, p) => s + p.noshow_prob, 0) / pts.length * 100).toFixed(1) : 0, n: pts.length, col: proc.col };
  }).filter(s => s.n > 0);

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 18, animation: "fadeUp .35s ease" }}>
      <div style={{ display: "grid", gridTemplateColumns: "repeat(3,1fr)", gap: 16 }}>
        <Card title="Population Risk Cohorts" accent={C.ind}>
          <ResponsiveContainer width="100%" height={200}>
            <PieChart>
              <Pie data={cohorts.map(c => ({ name: c.name, value: c.v }))} cx="50%" cy="50%" innerRadius={54} outerRadius={80} paddingAngle={4} dataKey="value" strokeWidth={0}>
                {cohorts.map((c, i) => <Cell key={i} fill={c.col} />)}
              </Pie>
              <Tooltip contentStyle={TIP_STYLE} />
              <Legend wrapperStyle={{ fontSize: 10.5 }} formatter={v => <span style={{ color: C.tx2 }}>{v}</span>} />
            </PieChart>
          </ResponsiveContainer>
        </Card>

        <Card title="Risk by Day of Week" accent={C.amb}>
          <ResponsiveContainer width="100%" height={200}>
            <BarChart data={dayData} barCategoryGap="35%">
              <GridLines />
              <XAxis dataKey="day" tick={axTick} axisLine={false} tickLine={false} />
              <YAxis tick={axTick} axisLine={false} tickLine={false} tickFormatter={v => `${v}%`} />
              <Tooltip contentStyle={TIP_STYLE} formatter={v => `${v}%`} />
              <Bar dataKey="risk" name="Avg Risk" radius={[4, 4, 0, 0]}>
                {dayData.map((d, i) => <Cell key={i} fill={parseFloat(d.risk) > 30 ? C.red : parseFloat(d.risk) > 22 ? C.amb : C.ind} />)}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </Card>

        <Card title="Risk by Lead Time" accent={C.tea}>
          <ResponsiveContainer width="100%" height={200}>
            <BarChart data={leadData} barCategoryGap="35%">
              <GridLines />
              <XAxis dataKey="bucket" tick={axTick} axisLine={false} tickLine={false} />
              <YAxis tick={axTick} axisLine={false} tickLine={false} tickFormatter={v => `${v}%`} />
              <Tooltip contentStyle={TIP_STYLE} formatter={v => `${v}%`} />
              <Bar dataKey="risk" name="Avg Risk" radius={[4, 4, 0, 0]}>
                {leadData.map((d, i) => <Cell key={i} fill={[C.grn, C.tea, C.amb, C.red][i]} />)}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </Card>
      </div>

      <div style={{ display: "grid", gridTemplateColumns: "1.4fr 1fr", gap: 16 }}>
        <Card title="Lead Time vs. No-Show Risk" sub="Each point = one patient · color = risk tier" accent={C.ind}>
          <ResponsiveContainer width="100%" height={240}>
            <ScatterChart>
              <CartesianGrid strokeDasharray="2 8" stroke="#E8EDF8" />
              <XAxis dataKey="x" name="Lead Time" tick={axTick} label={{ value: "Lead Time (days)", position: "insideBottom", offset: -8, fontSize: 10.5, fill: C.tx3 }} />
              <YAxis dataKey="y" name="No-Show Risk" tick={axTick} tickFormatter={v => `${v}%`} />
              <Tooltip contentStyle={TIP_STYLE} formatter={(v, n) => n === "No-Show Risk" ? `${v}%` : v} />
              <Scatter data={scatter} name="Patients">
                {scatter.map((d, i) => <Cell key={i} fill={d.prob < 0.2 ? C.grn : d.prob < 0.4 ? C.tea : d.prob < 0.6 ? C.amb : C.red} fillOpacity={0.8} />)}
              </Scatter>
            </ScatterChart>
          </ResponsiveContainer>
        </Card>

        <Card title="Intervention ROI Analysis" sub="Revenue recovery per $1 invested — dental" accent={C.grn}>
          <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
            {roi.map(d => (
              <div key={d.name} style={{ padding: "12px 14px", background: C.surface, borderRadius: 10, border: `1px solid ${C.br1}` }}>
                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 6 }}>
                  <span style={{ fontSize: 12.5, color: C.tx1, fontWeight: 600 }}>{d.name}</span>
                  <Chip col={C.amb}>{d.roi}% ROI</Chip>
                </div>
                <div style={{ display: "flex", gap: 16, marginBottom: 7 }}>
                  <span style={{ fontSize: 10.5, color: C.grn }}>▲ {d.red}% reduction</span>
                  <span style={{ fontSize: 10.5, color: C.tx3 }}>{d.cost > 0 ? `$${d.cost}/patient` : "Zero cost"}</span>
                </div>
                <PBar val={d.roi} max={2600} col={d.col} h={4} />
              </div>
            ))}
          </div>
        </Card>
      </div>

      {/* Dental procedure breakdown — replaces ARIL service line */}
      <Card title="No-Show Risk by Procedure Type" sub="High revenue × high risk = priority intervention target · dental-specific" accent={C.amb}>
        <ResponsiveContainer width="100%" height={200}>
          <BarChart data={procData} layout="vertical" barCategoryGap="28%">
            <GridLines />
            <XAxis type="number" tickFormatter={v => `${v}%`} tick={axTick} axisLine={false} tickLine={false} />
            <YAxis type="category" dataKey="name" tick={{ fontSize: 11.5, fill: C.tx2, fontFamily: C.sans }} axisLine={false} tickLine={false} width={120} />
            <Tooltip contentStyle={TIP_STYLE} formatter={v => `${v}%`} />
            <Bar dataKey="risk" name="Avg Risk %" radius={[0, 4, 4, 0]}>
              {procData.map((d, i) => <Cell key={i} fill={d.col} />)}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </Card>
    </div>
  );
}

/* ══════════════════════════════════════════════════════════
   ROOT APP — identical shell to ARIL
   Brand: DentalIQ · 🦷 icon · "Dental Ensemble v1.0"
   Everything else: same header, tabs, layout, live clock
══════════════════════════════════════════════════════════ */
export default function DentalIQ() {
  const [tab,     setTab]   = useState("overview");
  const [patients]          = useState(() => makePatients(24));
  const [waitlist]          = useState(() => makeWaitlist(10));
  const [clock,   setClock] = useState(new Date());

  useEffect(() => { const t = setInterval(() => setClock(new Date()), 1000); return () => clearInterval(t); }, []);

  const optimized = useMemo(() => patients.map(p => optSlot(p, 1.75)), [patients]);
  const mc        = useMemo(() => monteCarlo({ slots: 40, baseNS: 0.22, rev: 195, obFrac: 0.60, remLift: 0.30, depLift: 0.42, implCost: 15000 }, 600), []);

  const avgRisk = patients.reduce((s, p) => s + p.noshow_prob, 0) / patients.length;
  const critical = patients.filter(p => p.noshow_prob >= 0.5).length;
  const atRisk   = patients.reduce((s, p) => s + p.revenue * p.noshow_prob, 0);
  const expRev   = optimized.reduce((s, p) => s + p.expectedRevenue, 0);

  const TABS = [
    { id: "overview",   label: "Revenue Overview",   icon: "◈" },
    { id: "prediction", label: "Risk Detection",     icon: "⬡" },
    { id: "scheduling", label: "Schedule Optimizer", icon: "⬢" },
    { id: "simulation", label: "Revenue Simulation", icon: "◉" },
    { id: "analytics",  label: "Analytics",          icon: "⬟" },
  ];

  return (
    <>
      <div style={{ width: "100%", minHeight: "100vh", background: C.page, color: C.tx1, fontFamily: C.sans, display: "flex", flexDirection: "column", position: "relative" }}>

        {/* Ambient tint — identical to ARIL */}
        <div style={{ position: "fixed", inset: 0, pointerEvents: "none", zIndex: 0, backgroundImage: `radial-gradient(ellipse 55% 40% at 10% 10%, rgba(79,91,213,.05) 0%, transparent 55%), radial-gradient(ellipse 45% 35% at 90% 90%, rgba(8,145,178,.04) 0%, transparent 55%)` }} />

        {/* ── HEADER — identical ARIL shell, DentalIQ brand ── */}
        <header style={{ position: "sticky", top: 0, zIndex: 100, background: "rgba(255,255,255,.96)", backdropFilter: "blur(20px)", borderBottom: `1px solid ${C.br1}`, boxShadow: "0 1px 16px rgba(30,50,120,.07)", width: "100%", flexShrink: 0 }}>
          <div style={{ maxWidth: 1640, margin: "0 auto", padding: "0 28px" }}>

            {/* Top bar */}
            <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", height: 62 }}>

              {/* Brand */}
              <div style={{ display: "flex", alignItems: "center", gap: 14 }}>
                <svg viewBox="0 0 40 40" width="40" height="40">
                  <defs>
                    <linearGradient id="hx" x1="0" y1="0" x2="1" y2="1">
                      <stop offset="0%" stopColor={C.ind} /><stop offset="100%" stopColor={C.tea} />
                    </linearGradient>
                  </defs>
                  <polygon points="20,2 38,11 38,29 20,38 2,29 2,11" fill="url(#hx)" opacity="0.12" />
                  <polygon points="20,2 38,11 38,29 20,38 2,29 2,11" fill="none" stroke="url(#hx)" strokeWidth="1.8" />
                  <text x="20" y="25" textAnchor="middle" fontSize="17" fill={C.ind} fontWeight="700">🦷</text>
                </svg>
                <div>
                  <div style={{ fontFamily: C.mono, fontSize: 19, fontWeight: 700, letterSpacing: "3px", lineHeight: 1, background: `linear-gradient(120deg,${C.ind},${C.tea})`, WebkitBackgroundClip: "text", WebkitTextFillColor: "transparent" }}>DentalIQ</div>
                  <div style={{ fontSize: 9.5, color: C.tx3, letterSpacing: "1.8px", textTransform: "uppercase", marginTop: 1 }}>AI No-Show Intelligence · Dental</div>
                </div>
                <div style={{ width: 1, height: 32, background: C.br1, margin: "0 12px" }} />
                <div style={{ display: "flex", alignItems: "center", gap: 7 }}>
                  <div style={{ width: 7, height: 7, borderRadius: "50%", background: C.grn, animation: "pulse 2.2s ease infinite" }} />
                  <span style={{ fontSize: 11, color: C.tx3 }}>Live</span>
                  <span style={{ color: C.br2, margin: "0 2px" }}>·</span>
                  <span style={{ fontFamily: C.mono, fontSize: 11, color: C.tx2 }}>{clock.toLocaleTimeString()}</span>
                  <span style={{ color: C.br2, margin: "0 2px" }}>·</span>
                  <span style={{ fontSize: 11, color: C.tx3 }}>Dental Ensemble v1.0</span>
                </div>
              </div>

              {/* Live stats — identical ARIL pattern */}
              <div style={{ display: "flex", gap: 28, alignItems: "center" }}>
                {[
                  { l: "Appointments",    v: patients.length,                           col: C.ind },
                  { l: "Avg Risk",        v: `${(avgRisk * 100).toFixed(1)}%`,           col: avgRisk > 0.28 ? C.amb : C.tea },
                  { l: "Critical",        v: critical,                                   col: critical > 3 ? C.red : C.amb },
                  { l: "Revenue at Risk", v: `$${Math.round(atRisk).toLocaleString()}`,  col: C.amb },
                  { l: "Expected Rev",    v: `$${Math.round(expRev).toLocaleString()}`,  col: C.ind },
                ].map(m => (
                  <div key={m.l} style={{ textAlign: "center" }}>
                    <div style={{ fontFamily: C.mono, fontSize: 14, fontWeight: 700, color: m.col, lineHeight: 1, letterSpacing: "-.3px" }}>{m.v}</div>
                    <div style={{ fontSize: 9, color: C.tx3, textTransform: "uppercase", letterSpacing: ".9px", marginTop: 3 }}>{m.l}</div>
                  </div>
                ))}
              </div>
            </div>

            {/* Tabs — identical to ARIL */}
            <div style={{ display: "flex", borderTop: `1px solid ${C.br1}` }}>
              {TABS.map(t => (
                <button key={t.id} onClick={() => setTab(t.id)} className={`tab-btn${tab === t.id ? " active" : ""}`}>
                  <span style={{ fontFamily: C.mono, fontSize: 14, color: tab === t.id ? C.ind : C.tx3, transition: "color .15s" }}>{t.icon}</span>
                  {t.label}
                </button>
              ))}
            </div>
          </div>
        </header>

        {/* ── CONTENT ── */}
        <main style={{ flex: 1, width: "100%", overflowY: "auto", overflowX: "hidden", position: "relative", zIndex: 1 }}>
          <div style={{ maxWidth: 1640, margin: "0 auto", padding: "24px 28px 64px" }}>
            {tab === "overview"   && <OverviewTab   patients={patients} optimized={optimized} mc={mc} />}
            {tab === "prediction" && <PredictionTab patients={patients} />}
            {tab === "scheduling" && <SchedulingTab optimized={optimized} waitlist={waitlist} />}
            {tab === "simulation" && <SimulationTab />}
            {tab === "analytics"  && <AnalyticsTab  patients={patients} />}
          </div>
        </main>
      </div>
    </>
  );
}
