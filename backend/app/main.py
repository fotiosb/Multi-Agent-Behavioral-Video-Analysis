import asyncio
import json
import os
from pathlib import Path

from dotenv import dotenv_values, set_key
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from . import detector_process as det

ROOT = Path(__file__).parents[2]
ENV_PATH = ROOT / ".env"
# Assets: look alongside the project, then fall back to project root
_candidates = [ROOT.parent / "gui_code", ROOT]
ASSETS_DIR = next((p for p in _candidates if p.is_dir()), ROOT)

app = FastAPI(title="SentinelIQ API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173",
                   "http://localhost:8000", "http://127.0.0.1:8000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── WebSocket connection manager ──────────────────────────────────────────────

class ConnectionManager:
    def __init__(self):
        self.active: list[WebSocket] = []

    async def connect(self, ws: WebSocket):
        await ws.accept()
        self.active.append(ws)

    def disconnect(self, ws: WebSocket):
        if ws in self.active:
            self.active.remove(ws)

    async def broadcast(self, data: dict):
        dead = []
        for ws in self.active:
            try:
                await ws.send_json(data)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.disconnect(ws)

manager = ConnectionManager()

# Background task: drain event queue and broadcast to all WS clients
async def event_broadcaster():
    while True:
        try:
            if not det.event_queue.empty():
                event = det.event_queue.get_nowait()
                await manager.broadcast(event)
        except Exception:
            pass
        await asyncio.sleep(0.05)

@app.on_event("startup")
async def startup():
    asyncio.create_task(event_broadcaster())

# ── MJPEG stream ──────────────────────────────────────────────────────────────

async def mjpeg_generator():
    while True:
        try:
            if not det.frame_queue.empty():
                jpeg = det.frame_queue.get_nowait()
                yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n")
            else:
                await asyncio.sleep(0.005)
        except Exception:
            await asyncio.sleep(0.01)

@app.get("/api/stream")
async def video_stream():
    return StreamingResponse(
        mjpeg_generator(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )

# ── Detector control ──────────────────────────────────────────────────────────

class StartRequest(BaseModel):
    zone_points: list[list[int]] | None = None

@app.post("/api/detector/start")
def detector_start(req: StartRequest):
    if det.is_running():
        return {"started": False, "reason": "already running"}
    zone = [tuple(p) for p in req.zone_points] if req.zone_points else None
    det.start(zone_points=zone)
    return {"started": True}

@app.post("/api/detector/stop")
def detector_stop():
    det.stop()
    return {"stopped": True}

@app.get("/api/detector/status")
def detector_status():
    return det.get_status()

# ── Config (read/write .env) ──────────────────────────────────────────────────

ENV_KEYS = [
    "SOURCE", "YOLO_CONFIDENCE", "DWELL_SECONDS", "GEMINI_INTERVAL_SECONDS",
    "GEMINI_KEYFRAMES", "CLAUDE_FRAMES_DIRECT", "ZONE_EXIT_GRACE_SECONDS",
    "ZONE_ENTRY_GRACE_SECONDS", "TRACKER_MATCH_DIST", "POST_EXIT_BUFFER_SECONDS",
    "GEMINI_AUTO_DISABLE_AFTER", "GEMINI_COOLDOWN_MINUTES",
    "GEMINI_API_KEY", "ANTHROPIC_API_KEY",
]

@app.get("/api/config")
def get_config():
    vals = dotenv_values(ENV_PATH)
    return {k: vals.get(k, "") for k in ENV_KEYS}

class ConfigUpdate(BaseModel):
    values: dict[str, str]

@app.post("/api/config")
def update_config(update: ConfigUpdate):
    ENV_PATH.touch(exist_ok=True)
    for k, v in update.values.items():
        if k in ENV_KEYS:
            set_key(str(ENV_PATH), k, v)
    return {"saved": True}

# ── First frame for zone setup ────────────────────────────────────────────────

@app.get("/api/first-frame")
def first_frame():
    import cv2, base64
    vals = dotenv_values(ENV_PATH)
    source = vals.get("SOURCE", "")
    if not source:
        raise HTTPException(400, "SOURCE not set")
    cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise HTTPException(500, "Could not read frame from source")
    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
    b64 = base64.standard_b64encode(bytes(buf)).decode()
    h, w = frame.shape[:2]
    return {"image": f"data:image/jpeg;base64,{b64}", "width": w, "height": h}

# ── Zone persist/load ─────────────────────────────────────────────────────────

ZONE_FILE = ROOT / "config.json"

@app.get("/api/zone")
def get_zone():
    if not ZONE_FILE.exists():
        return {"zone": None}
    with open(ZONE_FILE) as f:
        return json.load(f)

class ZoneUpdate(BaseModel):
    zone: list[list[int]]

@app.post("/api/zone")
def save_zone(update: ZoneUpdate):
    with open(ZONE_FILE, "w") as f:
        json.dump({"zone": update.zone}, f, indent=2)
    return {"saved": True}

# ── WebSocket ─────────────────────────────────────────────────────────────────

@app.websocket("/ws/events")
async def ws_events(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        # Send current status immediately on connect
        await websocket.send_json({"type": "status", **det.get_status()})
        while True:
            await asyncio.sleep(0.3)
            await websocket.send_json({"type": "status", **det.get_status()})
    except WebSocketDisconnect:
        manager.disconnect(websocket)

# ── Dashboard (static mock data + live overrides) ─────────────────────────────

@app.get("/api/dashboard")
def dashboard():
    from .data import get_dashboard_data
    data = get_dashboard_data()
    # Patch live summary
    status = det.get_status()
    data["liveStatus"] = status
    return data

# ── Static assets (logo, demo videos) ────────────────────────────────────────

@app.get("/assets/{file_path:path}")
def asset(file_path: str):
    from fastapi.responses import FileResponse
    candidate = (ASSETS_DIR / file_path).resolve()
    if not str(candidate).startswith(str(ASSETS_DIR.resolve())) or not candidate.is_file():
        raise HTTPException(404, "Not found")
    return FileResponse(candidate)

@app.get("/api/health")
def health():
    return {"status": "ok"}
