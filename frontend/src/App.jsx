import { useCallback, useEffect, useMemo, useRef, useState } from "react";

const API = import.meta.env.VITE_API_BASE || "http://localhost:8000";
const assetUrl = (f) => `${API}/assets/${encodeURIComponent(f)}`;

// ── Hooks ─────────────────────────────────────────────────────────────────────

function useWebSocket(onMessage) {
  const ws = useRef(null);
  useEffect(() => {
    const connect = () => {
      const sock = new WebSocket(`ws://${new URL(API).host}/ws/events`);
      sock.onmessage = (e) => { try { onMessage(JSON.parse(e.data)); } catch {} };
      sock.onclose = () => setTimeout(connect, 2000);
      ws.current = sock;
    };
    connect();
    return () => ws.current?.close();
  }, []);
}

// ── Zone canvas overlay ───────────────────────────────────────────────────────

function ZoneCanvas({ width, height, points, onChange, locked }) {
  const ref = useRef(null);

  useEffect(() => {
    const canvas = ref.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    ctx.clearRect(0, 0, width, height);
    if (!points.length) return;
    ctx.beginPath();
    ctx.moveTo(points[0][0], points[0][1]);
    points.forEach(([x, y]) => ctx.lineTo(x, y));
    ctx.closePath();
    ctx.strokeStyle = "rgba(24,168,241,0.9)";
    ctx.lineWidth = 2;
    ctx.setLineDash([6, 3]);
    ctx.stroke();
    ctx.fillStyle = "rgba(24,168,241,0.12)";
    ctx.fill();
    points.forEach(([x, y]) => {
      ctx.beginPath();
      ctx.arc(x, y, 6, 0, Math.PI * 2);
      ctx.fillStyle = "#18a8f1";
      ctx.fill();
    });
  }, [points, width, height]);

  const handleClick = useCallback((e) => {
    if (locked) return;
    const rect = ref.current.getBoundingClientRect();
    const scaleX = width / rect.width;
    const scaleY = height / rect.height;
    const x = Math.round((e.clientX - rect.left) * scaleX);
    const y = Math.round((e.clientY - rect.top) * scaleY);
    onChange([...points, [x, y]]);
  }, [points, onChange, locked, width, height]);

  return (
    <canvas
      ref={ref}
      width={width}
      height={height}
      onClick={handleClick}
      style={{
        position: "absolute", inset: 0, width: "100%", height: "100%",
        cursor: locked ? "default" : "crosshair",
        borderRadius: "inherit",
      }}
    />
  );
}

// ── Zone Setup Modal ──────────────────────────────────────────────────────────

function ZoneSetupModal({ onClose, onSave }) {
  const [img, setImg] = useState(null);
  const [imgSize, setImgSize] = useState({ w: 640, h: 480 });
  const [points, setPoints] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [locked, setLocked] = useState(false);

  useEffect(() => {
    fetch(`${API}/api/first-frame`)
      .then(r => r.json())
      .then(d => {
        setImg(d.image);
        setImgSize({ w: d.width, h: d.height });
        setLoading(false);
      })
      .catch(() => { setError("Could not load first frame — check SOURCE in settings"); setLoading(false); });
    fetch(`${API}/api/zone`)
      .then(r => r.json())
      .then(d => { if (d.zone) setPoints(d.zone); })
      .catch(() => {});
  }, []);

  const save = async () => {
    if (points.length < 3) { setError("Need at least 3 points"); return; }
    await fetch(`${API}/api/zone`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ zone: points }),
    });
    onSave(points);
    onClose();
  };

  return (
    <div style={{ position:"fixed",inset:0,background:"rgba(2,8,20,0.85)",display:"flex",alignItems:"center",justifyContent:"center",zIndex:1000 }}>
      <div style={{ background:"var(--panel)",border:"1px solid var(--line)",borderRadius:24,padding:28,width:"min(90vw,780px)",maxHeight:"90vh",overflowY:"auto" }}>
        <div style={{ display:"flex",justifyContent:"space-between",alignItems:"center",marginBottom:16 }}>
          <div>
            <p className="eyebrow" style={{margin:0}}>Zone Definition</p>
            <h3 style={{margin:0}}>Click to place polygon points</h3>
          </div>
          <button className="ghost-btn" style={{borderRadius:12}} onClick={onClose}>Close</button>
        </div>

        {loading && <p className="muted">Loading first frame from source...</p>}
        {error && <p style={{color:"var(--danger)"}}>{error}</p>}

        {img && (
          <div style={{ position:"relative",borderRadius:12,overflow:"hidden",lineHeight:0,border:"1px solid var(--line)" }}>
            <img src={img} style={{ width:"100%",height:"auto",display:"block" }} alt="First frame" />
            <ZoneCanvas
              width={imgSize.w} height={imgSize.h}
              points={points} onChange={setPoints}
              locked={locked}
            />
          </div>
        )}

        <div style={{ display:"flex",gap:12,marginTop:16,flexWrap:"wrap",alignItems:"center" }}>
          <span style={{color:"var(--muted)",fontSize:13}}>{points.length} point{points.length!==1?"s":""} placed {points.length>=3?"✓":""}</span>
          <button className="ghost-btn" style={{borderRadius:12,color:"#ffffff"}} onClick={() => setPoints([])}>Reset</button>
          {points.length >= 3 && (
            <button className="ghost-btn" style={{borderRadius:12,color:"#ffffff"}} onClick={() => setLocked(l => !l)}>
              {locked ? "Unlock (edit)" : "Lock zone"}
            </button>
          )}
          <button className="primary-btn" style={{borderRadius:12,marginLeft:"auto",color:"#ffffff"}} onClick={save}>
            Save zone
          </button>
        </div>
        <p style={{color:"var(--muted)",fontSize:12,marginTop:8}}>
          Click on the image to place points. Close the polygon with at least 3 points. Saved to config.json.
        </p>
      </div>
    </div>
  );
}

// ── Settings Panel ────────────────────────────────────────────────────────────

const SETTING_DEFS = [
  { key:"SOURCE",            label:"Video source (RTSP URL or file path)", type:"text", wide:true },
  { key:"YOLO_CONFIDENCE",   label:"YOLO confidence threshold (0–1)", type:"number" },
  { key:"DWELL_SECONDS",     label:"Dwell before analysis (s)", type:"number" },
  { key:"GEMINI_INTERVAL_SECONDS", label:"Gemini repeat interval (s)", type:"number" },
  { key:"GEMINI_KEYFRAMES",  label:"Keyframes sent to Gemini", type:"number" },
  { key:"CLAUDE_FRAMES_DIRECT", label:"Max frames sent to Claude", type:"number" },
  { key:"ZONE_ENTRY_GRACE_SECONDS", label:"Zone entry grace (s)", type:"number" },
  { key:"ZONE_EXIT_GRACE_SECONDS",  label:"Zone exit grace (s)", type:"number" },
  { key:"POST_EXIT_BUFFER_SECONDS", label:"Post-exit frame buffer (s)", type:"number" },
  { key:"TRACKER_MATCH_DIST", label:"Person tracker match distance (px)", type:"number" },
  { key:"GEMINI_AUTO_DISABLE_AFTER", label:"Gemini auto-disable after N failures", type:"number" },
  { key:"GEMINI_COOLDOWN_MINUTES",   label:"Gemini cooldown after auto-disable (min)", type:"number" },
];

function SettingsPanel({ onZoneSetup, onConfigSaved }) {
  const [cfg, setCfg] = useState({});
  const [saved, setSaved] = useState(false);
  const [show, setShow] = useState({});

  useEffect(() => {
    fetch(`${API}/api/config`).then(r => r.json()).then(setCfg).catch(() => {});
  }, []);

  const save = async () => {
    await fetch(`${API}/api/config`, {
      method:"POST",
      headers:{"Content-Type":"application/json"},
      body: JSON.stringify({ values: cfg }),
    });
    setSaved(true);
    setTimeout(() => setSaved(false), 2000);
    if (onConfigSaved) onConfigSaved(cfg.SOURCE || "");
  };

  return (
    <section className="content-grid two-column">
      <div className="panel" style={{ gridColumn:"1 / -1" }}>
        <div className="panel-head">
          <div>
            <p className="section-label">Detector Configuration</p>
            <h3>Source, API keys and tuning parameters</h3>
          </div>
          <div style={{display:"flex",gap:8}}>
            <button className="ghost-btn" style={{borderRadius:12}} onClick={onZoneSetup}>
              Define zone
            </button>
            <button className="primary-btn" style={{borderRadius:12}} onClick={save}>
              {saved ? "Saved ✓" : "Save"}
            </button>
          </div>
        </div>

        <div style={{ display:"grid", gridTemplateColumns:"repeat(auto-fill,minmax(340px,1fr))", gap:12, marginTop:8 }}>
          {SETTING_DEFS.map(def => (
            <label key={def.key} style={{
              display:"flex", flexDirection:"column", gap:6,
              gridColumn: def.wide ? "1 / -1" : undefined,
            }}>
              <span style={{fontSize:12,color:"var(--muted)",textTransform:"uppercase",letterSpacing:"0.06em"}}>{def.label}</span>
              <div style={{display:"flex",gap:4}}>
                <input
                  type={def.type === "password" && !show[def.key] ? "password" : def.type === "number" ? "number" : "text"}
                  value={cfg[def.key] || ""}
                  onChange={e => setCfg(c => ({...c, [def.key]: e.target.value}))}
                  step={def.type === "number" ? "any" : undefined}
                  style={{
                    flex:1, background:"rgba(14,31,58,0.82)", border:"1px solid var(--line)",
                    borderRadius:10, padding:"8px 12px", color:"var(--text)", fontSize:14,
                  }}
                />
                {def.type === "password" && (
                  <button className="ghost-btn" style={{padding:"8px 10px",borderRadius:10,fontSize:12}}
                    onClick={() => setShow(s => ({...s,[def.key]:!s[def.key]}))}>
                    {show[def.key] ? "Hide" : "Show"}
                  </button>
                )}
              </div>
            </label>
          ))}
        </div>
      </div>
    </section>
  );
}

// ── Live panel ────────────────────────────────────────────────────────────────

function LivePanel({ data, activeAlert, setActiveAlertIndex, activeAlertIndex, clock, externalSource }) {
  const [running, setRunning] = useState(false);
  const [status, setStatus] = useState({});
  const [source, setSource] = useState("");
  const [events, setEvents] = useState([]);
  const [zonePoints, setZonePoints] = useState(null);
  const [showZoneModal, setShowZoneModal] = useState(false);
  const [liveSource, setLiveSource] = useState("");
  const [lastClaude, setLastClaude] = useState(null);
  const [startInFlight, setStartInFlight] = useState(false);
  const [progress, setProgress] = useState(null);
  const [playbackFinished, setPlaybackFinished] = useState(false);
  const eventsRef = useRef(null);

  // Load zone, status, and source from config on mount
  useEffect(() => {
    fetch(`${API}/api/zone`).then(r=>r.json()).then(d => { if(d.zone && d.zone.length) setZonePoints(d.zone); }).catch(()=>{});
    fetch(`${API}/api/detector/status`).then(r=>r.json()).then(s => { setStatus(s); setRunning(s.running); }).catch(()=>{});
    fetch(`${API}/api/config`).then(r=>r.json()).then(c => { if(c.SOURCE) setSource(c.SOURCE); }).catch(()=>{}); 
  }, []);

  useEffect(() => { if (externalSource) setSource(externalSource); }, [externalSource]);

  useWebSocket((msg) => {
    if (msg.type === "status") {
      setStatus(msg);
      setRunning(msg.running);
      if (msg.source) setSource(msg.source);
      if (msg.progress !== undefined) setProgress(msg.progress);
    }
    else if (msg.type === "playback_finished") {
      setPlaybackFinished(true);
      setRunning(false);
    }
    else if (msg.type === "claude_result") {
      setLastClaude(msg);
      setEvents(ev => [{...msg, ts: new Date().toLocaleTimeString()}, ...ev].slice(0, 100));
    }
    else if (["zone_entry","zone_exit","gemini_result","gemini_start","system"].includes(msg.type)) {
      setEvents(ev => [{...msg, ts: new Date().toLocaleTimeString()}, ...ev].slice(0, 100));
    }
  });

  useEffect(() => {
    if (eventsRef.current) eventsRef.current.scrollTop = 0;
  }, [events]);

  const toggle = async () => {
    if (startInFlight) return;
    if (running) {
      await fetch(`${API}/api/detector/stop`, { method:"POST" });
      setRunning(false);
    } else {
      setStartInFlight(true);
      const zone = zonePoints ? zonePoints.map(p => Array.isArray(p) ? p : [p[0],p[1]]) : null;
      try {
        const resp = await fetch(`${API}/api/detector/start`, {
          method:"POST",
          headers:{"Content-Type":"application/json"},
          body: JSON.stringify({ zone_points: zone }),
        });
        const data = await resp.json();
        if (data.started !== false) {
          setRunning(true);
          setEvents([]);
          setLastClaude(null);
          setProgress(null);
          setPlaybackFinished(false);
        }
      } finally {
        setStartInFlight(false);
      }
    }
  };

  const alertColor = lastClaude?.detected
    ? (lastClaude.confidence === "high" ? "var(--danger)" : "var(--warning)")
    : "var(--success)";

  const eventColor = (ev) => {
    if (ev.type === "claude_result") return ev.detected ? (ev.confidence==="high" ? "var(--danger)" : "var(--warning)") : "var(--muted)";
    if (ev.type === "zone_entry") return "var(--accent)";
    if (ev.type === "zone_exit") return "var(--muted)";
    if (ev.type === "gemini_result") return ev.verdict === "likely" ? "var(--warning)" : "var(--muted)";
    return "var(--muted)";
  };

  const eventLabel = (ev) => {
    if (ev.type === "claude_result") {
      if (ev.detected) return `⚠ ANOMALY: ${(ev.anomaly_type||"").replace(/_/g," ")} [${ev.confidence}] — ${ev.reason}`;
      return `✓ No anomaly — ${ev.reason}`;
    }
    if (ev.type === "zone_entry") return `→ ${ev.message}`;
    if (ev.type === "zone_exit") return `← ${ev.message}`;
    if (ev.type === "gemini_start") return `Gemini: sending ${ev.frames} frames (${ev.trigger})`;
    if (ev.type === "gemini_result") return `Gemini: ${(ev.verdict||"").toUpperCase()} — ${ev.description||""}`;
    if (ev.type === "system") return `◦ ${ev.message}`;
    return JSON.stringify(ev);
  };

  return (
    <>
      {showZoneModal && (
        <ZoneSetupModal
          onClose={() => setShowZoneModal(false)}
          onSave={(pts) => setZonePoints(pts)}
        />
      )}

      <section className="dashboard-grid">
        {/* Live feed panel */}
        <div className="panel feed-panel">
          <div className="panel-head">
            <div>
              <p className="section-label">{source && !source.toLowerCase().startsWith("rtsp") ? "Video File" : "Live Feed"}</p>
              <p style={{margin:0,fontSize:12,color:"var(--muted)",wordBreak:"break-all",maxWidth:360}}>{source || status.source || "No source configured"}</p>
              <p style={{margin:"4px 0 0",fontSize:11,color:"var(--muted)",letterSpacing:"0.02em"}}>
                {clock}{running || playbackFinished ? "  " + (running ? `${status.resolution||""} @ ${status.fps||0} fps` : playbackFinished ? "finished" : "stopped") : ""}
              </p>
            </div>
            <div className="panel-actions">
              <button className="chip" style={{color:"#ffffff"}} onClick={() => setShowZoneModal(true)}>
                {zonePoints ? "Edit zone" : "Define zone"}
              </button>
              <button
                className={running ? "chip active-chip" : "chip"}
                onClick={toggle}
                style={{
                  color: "#ffffff",
                  background: running ? "rgba(255,123,92,0.18)" : undefined,
                  borderColor: running ? "rgba(255,123,92,0.4)" : undefined,
                }}
              >
                {running ? "■ Stop" : "▶ Start"}
              </button>
            </div>
          </div>

          {/* ── FIXED VIDEO SECTION ── */}
          <div className="video-stage" style={{ position: "relative" }}>
            {/* Fixed aspect-ratio container – prevents cropping + layout shift */}
            <div 
              className="scan-frame"
              style={{
                position: "relative",
                width: "100%",
                aspectRatio: "16 / 9",           // ← change to "4 / 3" if your stream is 4:3
                background: "#0a172b",
                borderRadius: "12px",
                overflow: "hidden",
                minHeight: "380px",              // fallback for narrow viewports
              }}
            >
              {running ? (
                <img
                  src={`${API}/api/stream`}
                  style={{
                    width: "100%",
                    height: "100%",
                    objectFit: "contain",
                    display: "block",
                  }}
                  alt="Live stream"
                />
              ) : (
                <div style={{
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "center",
                  height: "100%",
                  color: "var(--muted)",
                  flexDirection: "column",
                  gap: 8,
                }}>
                  <span style={{ fontSize: 32 }}>◉</span>
                  <span>Press Start to begin detection</span>
                </div>
              )}

              {/* Risk badge stays on top of video */}
              {lastClaude?.detected && (
                <div style={{ position: "absolute", bottom: 12, right: 12, zIndex: 2 }}>
                  <div className={`risk-badge ${lastClaude.confidence === "high" ? "high" : "medium"}`}>
                    {(lastClaude.anomaly_type || "anomaly").replace(/_/g, " ")}
                  </div>
                </div>
              )}
            </div>
          </div>

          {/* Progress bar – always rendered with reserved height → no layout shift */}
          {source && !source.toLowerCase().startsWith("rtsp") && (
            <div style={{
              padding: "8px 16px",
              background: "rgba(10,23,43,0.9)",
              borderTop: "1px solid var(--line)",
              display: "flex",
              alignItems: "center",
              gap: 12,
              minHeight: "52px",               // ← reserves space even when hidden
              opacity: progress && !playbackFinished ? 1 : 0.3,
            }}>
              <span style={{ fontSize: 12, color: "var(--muted)", minWidth: 110, fontFamily: "var(--font-mono)" }}>
                {playbackFinished
                  ? "00:00 / 00:00"
                  : `${String(Math.floor((progress?.current || 0) / 60)).padStart(2, "0")}:${String((progress?.current || 0) % 60).padStart(2, "0")} / ${String(Math.floor((progress?.total || 0) / 60)).padStart(2, "0")}:${String((progress?.total || 0) % 60).padStart(2, "0")}`}
              </span>
              <div style={{ flex: 1, height: 4, background: "rgba(148,163,184,0.15)", borderRadius: 999, overflow: "hidden" }}>
                <div style={{
                  width: `${playbackFinished ? 0 : (progress?.pct || 0)}%`,
                  height: "100%",
                  background: "var(--brand)",
                  borderRadius: 999,
                  transition: "width 0.5s linear",
                }} />
              </div>
              <span style={{ fontSize: 12, color: "var(--muted)", minWidth: 36, textAlign: "right" }}>
                {playbackFinished ? "0%" : `${progress?.pct || 0}%`}
              </span>
            </div>
          )}

          {/* Claude status bar – always rendered with reserved height → no layout shift */}
          {lastClaude && (
            <div style={{
              padding: "12px 16px",
              background: "rgba(14,31,58,0.6)",
              borderTop: "1px solid var(--line)",
              borderBottomLeftRadius: 20,
              borderBottomRightRadius: 20,
              fontSize: 13,
              minHeight: "58px",               // ← reserves space
            }}>
              <span style={{ color: alertColor, fontWeight: 600 }}>
                {lastClaude.detected 
                  ? `⚠ ${(lastClaude.anomaly_type || "").replace(/_/g, " ").toUpperCase()} [${lastClaude.confidence}]` 
                  : "✓ No anomaly"}
              </span>
              <span style={{ color: "var(--muted)", marginLeft: 8 }}>{lastClaude.reason}</span>
            </div>
          )}
        </div>

        {/* Alerts panel — sample data */}
        <div className="panel alerts-panel">
          <div className="panel-head">
            <div>
              <p className="section-label">Sample Incidents</p>
              <h3>Demo alert feed</h3>
            </div>
          </div>
          <div className="alert-list">
            {data.alerts.map((alert, i) => (
              <button key={alert.title} className={`alert-card ${i===activeAlertIndex?"active-alert":""}`}
                onClick={() => setActiveAlertIndex(i)}>
                <div>
                  <p className="alert-title">{alert.title}</p>
                  <p className="alert-meta">{`${alert.camera} • ${alert.meta}`}</p>
                </div>
                <span className={`risk-badge ${alert.riskClass}`}>{alert.risk}</span>
              </button>
            ))}
          </div>
        </div>

        {/* Event log panel */}
        <div className="panel details-panel">
          <div className="panel-head">
            <div>
              <p className="section-label">Event Log</p>
              <h3>Live detection events</h3>
            </div>
            <button className="chip" onClick={() => setEvents([])}>Clear</button>
          </div>
          <div ref={eventsRef} style={{ overflowY:"auto",maxHeight:320,display:"flex",flexDirection:"column",gap:6 }}>
            {events.length === 0 && (
              <p style={{color:"var(--muted)",fontSize:13,padding:8}}>Events will appear here when the detector is running.</p>
            )}
            {events.map((ev, i) => (
              <div key={i} style={{
                padding:"8px 12px",borderRadius:10,background:"rgba(14,31,58,0.6)",
                fontSize:12,lineHeight:1.5,borderLeft:`3px solid ${eventColor(ev)}`,
              }}>
                <span style={{color:"var(--muted)",marginRight:8}}>{ev.ts}</span>
                <span style={{color:eventColor(ev)}}>{eventLabel(ev)}</span>
              </div>
            ))}
          </div>
        </div>

        {/* Camera fleet */}
        <div className="panel camera-panel">
          <div className="panel-head">
            <div>
              <p className="section-label">Camera Fleet</p>
              <h3>Health and coverage</h3>
            </div>
          </div>
          <div className="camera-list">
            {data.cameraFleet.map(cam => (
              <div key={cam.name} className="camera-row">
                <div><strong>{cam.name}</strong><span>{cam.location}</span></div>
                <span className={`status ${cam.statusClass}`}>{cam.status}</span>
              </div>
            ))}
          </div>
        </div>

        {/* Model metrics */}
        <div className="panel metrics-panel">
          <div className="panel-head">
            <div>
              <p className="section-label">Model Performance</p>
              <h3>Detection quality snapshot</h3>
            </div>
          </div>
          <div className="metric-bars">
            {data.modelMetrics.map(m => (
              <div key={m.label} className="metric-row">
                <span>{m.label}</span>
                <div className="bar"><i style={{width:`${m.width}%`}}></i></div>
                <strong>{m.value}</strong>
              </div>
            ))}
          </div>
        </div>
      </section>
    </>
  );
}

// ── App root ──────────────────────────────────────────────────────────────────

export default function App() {
  const [data, setData] = useState(null);
  const [activeTab, setActiveTab] = useState("live");
  const [activeAlertIndex, setActiveAlertIndex] = useState(0);
  const [timelineValue, setTimelineValue] = useState(30);
  const [clock, setClock] = useState("");
  const [showZoneModal, setShowZoneModal] = useState(false);
  const [liveSource, setLiveSource] = useState("");

  useEffect(() => {
    fetch(`${API}/api/dashboard`).then(r=>r.json()).then(setData).catch(()=>setData(null));
  }, []);

  useEffect(() => {
    const id = setInterval(() => {
      setClock(new Date().toLocaleTimeString([],{hour:"2-digit",minute:"2-digit",second:"2-digit"}));
    }, 1000);
    setClock(new Date().toLocaleTimeString([],{hour:"2-digit",minute:"2-digit",second:"2-digit"}));
    return () => clearInterval(id);
  }, []);

  const activeTabMeta = useMemo(() => data?.tabs?.find(t=>t.id===activeTab)??null, [activeTab,data]);
  const activeAlert = data?.alerts?.[activeAlertIndex]??null;

  if (!data || !activeTabMeta || !activeAlert) {
    return (
      <div className="loading-shell">
        <div className="loading-card">
          <h1>SentinelIQ</h1>
          <p>Connecting to backend...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="shell">
      {showZoneModal && (
        <ZoneSetupModal onClose={() => setShowZoneModal(false)} onSave={() => {}} />
      )}

      <aside className="sidebar">
        <div className="brand">
          <img className="brand-logo" src={assetUrl(data.branding.logoFile)} alt="logo" />
          <div className="brand-copy">
            <p className="eyebrow">AI Surveillance</p>
            <h1>{data.branding.companyName}</h1>
          </div>
        </div>

        <nav className="nav">
          {data.tabs.map(tab => (
            <button key={tab.id} className={`nav-item ${activeTab===tab.id?"active":""}`}
              onClick={() => setActiveTab(tab.id)}>
              {tab.label}
            </button>
          ))}
        </nav>

        <section className="sidebar-card">
          <p className="section-label">System Summary</p>
          <div className="summary-grid">
            {data.summary.map(item => (
              <div key={item.label}><span>{item.label}</span><strong>{item.value}</strong></div>
            ))}
          </div>
        </section>

        <section className="sidebar-card">
          <p className="section-label">Operator Tips</p>
          <ul className="tips">{data.tips.map(t=><li key={t}>{t}</li>)}</ul>
        </section>
      </aside>

      <main className="main-content">
        <header className="topbar">
          <div>
            <p className="eyebrow">Operations Center</p>
            <h2>{activeTabMeta.pageTitle}</h2>
          </div>
          <div className="topbar-actions">
            <button className="ghost-btn">Export Report</button>
            <button className="primary-btn">Acknowledge Selected</button>
          </div>
        </header>

        <section className="hero">
          <div className="hero-copy">
            <p className="hero-kicker">{activeTabMeta.heroKicker}</p>
            <h3>{activeTabMeta.heroTitle}</h3>
            <p>{activeTabMeta.heroBody}</p>
          </div>
          <div className="hero-brand">
            <img className="hero-logo" src={assetUrl(data.branding.logoFile)} alt="logo" />
            <div className="hero-brand-copy">
              <span className="hero-brand-label">{data.branding.companyName} Platform</span>
              <strong>{data.branding.tagline}</strong>
            </div>
          </div>
          <div className="hero-stats">
            {data.heroStats.map(item => (
              <div key={item.label} className={`stat-pill ${item.kind==="warning"?"warning":""}`}>
                {item.kind==="live" && <span className="dot live"></span>}
                {item.label}
              </div>
            ))}
          </div>
        </section>

        {activeTab === "live" && (
          <LivePanel
            data={data}
            activeAlert={activeAlert}
            activeAlertIndex={activeAlertIndex}
            setActiveAlertIndex={setActiveAlertIndex}
            clock={clock}
            externalSource={liveSource}
          />
        )}

        {activeTab === "settings" && (
          <SettingsPanel onZoneSetup={() => setShowZoneModal(true)} onConfigSaved={(src) => setLiveSource(src)} />
        )}

        {activeTab === "review" && (
          <section className="content-grid two-column">
            <div className="panel">
              <div className="panel-head">
                <div><p className="section-label">Review Queue</p><h3>Flagged clips waiting for triage</h3></div>
                <button className="chip">Show confirmed only</button>
              </div>
              <div className="review-queue">
                {data.reviewQueue.map((clip,i) => (
                  <div key={clip.clip} className={`queue-card ${i===0?"selected-queue":""}`}>
                    <div><strong>{clip.clip}</strong><span>{`${clip.camera} • ${clip.duration}`}</span></div>
                    <span className={`risk-badge ${clip.riskClass}`}>{clip.risk}</span>
                  </div>
                ))}
              </div>
            </div>
            <div className="panel">
              <div className="panel-head"><div><p className="section-label">Analyst Notes</p><h3>Example incident summary</h3></div></div>
              <div className="stack-list">
                {data.reviewNotes.map(n=><div key={n.label} className="info-card"><span>{n.label}</span><strong>{n.value}</strong></div>)}
              </div>
            </div>
            <div className="panel">
              <div className="panel-head"><div><p className="section-label">Evidence Timeline</p><h3>Annotated clip moments</h3></div></div>
              <div className="event-log">
                {data.reviewTimeline.map(ev=><div key={ev.time} className="log-row"><strong>{ev.time}</strong><span>{ev.text}</span></div>)}
              </div>
            </div>
            <div className="panel">
              <div className="panel-head"><div><p className="section-label">Disposition</p><h3>Analyst actions</h3></div></div>
              <div className="review-actions">
                <button className="primary-btn full-width">Confirm incident and escalate</button>
                <button className="ghost-btn full-width">Request second review</button>
                <button className="ghost-btn full-width">Mark as training false positive</button>
              </div>
            </div>
          </section>
        )}

        {activeTab === "health" && (
          <section className="content-grid two-column">
            <div className="panel">
              <div className="panel-head"><div><p className="section-label">Fleet Status</p><h3>Coverage by zone</h3></div></div>
              <div className="health-grid">
                {data.healthCards.map(c=><div key={c.label} className="health-card"><span>{c.label}</span><strong>{c.value}</strong><p>{c.text}</p></div>)}
              </div>
            </div>
            <div className="panel">
              <div className="panel-head"><div><p className="section-label">Maintenance Flags</p><h3>Needs attention soon</h3></div></div>
              <div className="event-log">
                {data.maintenanceFlags.map(f=><div key={f.title} className="log-row"><strong>{f.title}</strong><span>{f.text}</span></div>)}
              </div>
            </div>
            <div className="panel wide-panel">
              <div className="panel-head"><div><p className="section-label">Uptime History</p><h3>Last 7 days</h3></div></div>
              <div className="uptime-table">
                <div className="table-row header-row"><span>Camera</span><span>Uptime</span><span>Bandwidth</span><span>Last issue</span></div>
                {data.uptimeRows.map(r=><div key={r.camera} className="table-row"><span>{r.camera}</span><span>{r.uptime}</span><span>{r.bandwidth}</span><span>{r.issue}</span></div>)}
              </div>
            </div>
          </section>
        )}

        {activeTab === "insights" && (
          <section className="content-grid two-column">
            <div className="panel">
              <div className="panel-head"><div><p className="section-label">Model Drift</p><h3>Recent behavior change signals</h3></div></div>
              <div className="stack-list">
                {data.driftNotes.map(n=><div key={n.label} className="info-card"><span>{n.label}</span><strong>{n.value}</strong></div>)}
              </div>
            </div>
            <div className="panel">
              <div className="panel-head"><div><p className="section-label">Class Distribution</p><h3>Flag types this week</h3></div></div>
              <div className="metric-bars">
                {data.classDistribution.map(m=><div key={m.label} className="metric-row"><span>{m.label}</span><div className="bar"><i style={{width:`${m.width}%`}}></i></div><strong>{m.value}</strong></div>)}
              </div>
            </div>
            <div className="panel wide-panel">
              <div className="panel-head"><div><p className="section-label">Analyst Feedback Loop</p><h3>Labels ready for retraining</h3></div></div>
              <div className="uptime-table">
                <div className="table-row header-row"><span>Source</span><span>Confirmed</span><span>False positive</span><span>Ready</span></div>
                {data.feedbackRows.map(r=><div key={r.source} className="table-row"><span>{r.source}</span><span>{r.confirmed}</span><span>{r.falsePositive}</span><span>{r.ready}</span></div>)}
              </div>
            </div>
          </section>
        )}
      </main>
    </div>
  );
}