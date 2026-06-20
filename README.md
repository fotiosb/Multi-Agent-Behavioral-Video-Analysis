# SentinelIQ — Multi-Agent Behavioral Video Analysis

An AI-powered surveillance anomaly detection system using a three-layer intelligence architecture: local YOLO person detection, Gemini Flash first-pass filter, and Claude deep semantic judgment. Includes a full web GUI built with React and FastAPI.

---

## Architecture

```
Video source (RTSP stream or local file)
        │
        ▼
┌─────────────────────────────────────────┐
│  Layer 1 — YOLOv8m  (local, always-on)  │
│  Detects persons, tracks zone entry/exit │
│  Cost: local compute only               │
└──────────────────┬──────────────────────┘
                   │ person confirmed in zone
                   ▼
┌─────────────────────────────────────────┐
│  Layer 2 — Gemini Flash                 │
│  Receives 8 keyframes, returns:         │
│  clear_no / maybe / likely + description│
│  Cost: ~$16/month per camera            │
└──────────────────┬──────────────────────┘
                   │ maybe or likely
                   ▼
┌─────────────────────────────────────────┐
│  Layer 3 — Claude Sonnet                │
│  Receives up to 20 frames (biased       │
│  sample: half spread, half from end).   │
│  Returns: detected, confidence,         │
│  anomaly_type, reason (JSON)            │
│  Cost: ~$45/month per camera            │
└─────────────────────────────────────────┘
        Total API cost: ~$60/month per camera
```

### Key design decisions

**Why three layers?** YOLO alone fires on presence, not behaviour. A single AI layer on every frame would be expensive and slow. The three-layer architecture gates API calls aggressively — Gemini only activates after a person has dwelled in the zone, Claude only activates when Gemini flags something worth a second opinion.

**Why keyframes instead of Gemini Live streaming?** For this MVP, Gemini receives evenly-sampled frames rather than a WebSocket stream. This works identically for both local files and RTSP streams, is significantly simpler, and produces equivalent detection quality for the target behaviours.

**Why biased Claude frame sampling?** With a large buffer (e.g. 900 frames), pure even sampling may miss a fire or explosion that appears in the final seconds. Claude receives half its frames evenly spread (for context) and half from the final quarter of the buffer (for consequences), ensuring post-departure events are always represented.

**Threaded capture:** YOLO inference on CPU takes longer than the RTSP frame interval. Without a dedicated capture thread the main loop would block on network I/O, causing stream timeouts. `StreamCapture` reads frames in a background thread; the detection loop always has a fresh frame without waiting on the network.

**Post-exit frame buffer:** When a person leaves the monitored zone, the system continues capturing frames for a configurable period (`POST_EXIT_BUFFER_SECONDS`, default 20s). These frames are appended to the person's buffer before the exit analysis fires. In file mode, post-exit frames are collected until the end of the file. This allows detection of consequences — fire from arson, a person left on the ground — that appear after the subject has departed.

**Automatic Gemini failover:** After `GEMINI_AUTO_DISABLE_AFTER` consecutive failures (default 3), Gemini is automatically disabled for `GEMINI_COOLDOWN_MINUTES` (default 10). During this period Claude acts as sole analyst using a full-length prompt and up to 20 frames. Gemini re-enables automatically after the cooldown.

---

## Anomaly types detected

| Category | Types |
|---|---|
| Violence | assault, fighting, abuse |
| Weapons | weapon (firearm, blade, blunt object) |
| Criminal | robbery, burglary, theft, shoplifting, vandalism |
| Fire | arson, fire |
| Distress | person_down, distress |
| Suspicious | loitering, suspicious_behaviour |
| Access | intrusion, tailgating |
| Other | abandoned_object, crowd_anomaly, other |

---

## Project structure

```
video-analytics/
├── main.py                  Standalone detector (OpenCV window, CLI only)
├── zone_setup.py            CLI zone definition tool (OpenCV window)
├── .env                     Active configuration (create from .env.example)
├── .env.example             Configuration template
├── config.json              Zone polygon (written by zone setup tool or GUI)
├── README.md
├── requirements.txt         Python deps for standalone CLI mode
├── start.bat                Windows: launches backend + frontend together
│
├── backend/                 FastAPI backend (GUI mode)
│   ├── requirements.txt
│   └── app/
│       ├── main.py          REST API, MJPEG stream, WebSocket events
│       ├── detector_process.py  Full detector running in a background thread
│       └── data.py          Mock dashboard data (review/health/insights tabs)
│
└── frontend/                React + Vite web UI
    ├── package.json
    ├── vite.config.js
    ├── index.html
    └── src/
        ├── App.jsx          Full integrated UI
        ├── main.jsx         React entry point
        └── styles.css       SentinelIQ design system
```

---

## Prerequisites

### All platforms
- **Python 3.11** (not 3.12+ — ultralytics and some OpenCV builds have compatibility issues)
- **Node.js 18+** and **npm** (for the React frontend)
- **Gemini API key** — from [Google AI Studio](https://aistudio.google.com)
- **Anthropic API key** — from [Anthropic Console](https://console.anthropic.com)

### Windows-specific
- Python 3.11 from [python.org](https://www.python.org/downloads/release/python-3119/) — use the 64-bit installer
- During install: check **"Add Python to PATH"** only if you want it as system default. For a clean install alongside other Python versions, use the Python Launcher (`py -3.11`)
- Node.js from [nodejs.org](https://nodejs.org) — LTS version recommended

### Linux-specific
- `sudo apt install python3.11 python3.11-venv python3-pip` (Ubuntu/Debian)
- `sudo apt install nodejs npm`
- For OpenCV camera access: `sudo apt install libgl1-mesa-glx libglib2.0-0`

---

## Installation

### Step 1 — Clone the repository
```bash
git clone https://github.com/fotiosb/Multi-Agent-Behavioral-Video-Analysis.git
cd Multi-Agent-Behavioral-Video-Analysis
```

### Step 2 — Create a Python 3.11 virtual environment

**Windows:**
```bat
py -3.11 -m venv venv
venv\Scripts\activate
```

**Linux/macOS:**
```bash
python3.11 -m venv venv
source venv/bin/activate
```

### Step 3 — Install Python dependencies

**Standalone CLI mode only:**
```bash
pip install --upgrade pip
pip install opencv-python ultralytics python-dotenv google-genai anthropic
```

**GUI mode (includes everything above plus FastAPI):**
```bash
pip install --upgrade pip
pip install opencv-python ultralytics python-dotenv google-genai anthropic
pip install fastapi "uvicorn[standard]" python-multipart
```

> `yolov8m.pt` (~52 MB) downloads automatically on first run.

### Step 4 — Install frontend dependencies
```bash
cd frontend
npm install
cd ..
```

### Step 5 — Configure

Copy `.env.example` to `.env` and fill in your values:

```bash
# Windows
copy .env.example .env

# Linux/macOS
cp .env.example .env
```

Minimum required values:
```
SOURCE=rtsp://your-camera-ip:port/stream
GEMINI_API_KEY=your_gemini_key_here
ANTHROPIC_API_KEY=your_anthropic_key_here
```

For local file testing:
```
SOURCE=C:\Users\YourName\Videos\test.mp4   # Windows
SOURCE=/home/yourname/videos/test.mp4      # Linux
```

---

## Running — GUI mode (recommended)

### Windows
Double-click `start.bat`, or run manually in two terminals:

```bat
:: Terminal 1 — backend
venv\Scripts\activate
uvicorn backend.app.main:app --reload --port 8000

:: Terminal 2 — frontend
cd frontend
npm run dev
```

### Linux/macOS
```bash
# Terminal 1 — backend
source venv/bin/activate
uvicorn backend.app.main:app --reload --port 8000

# Terminal 2 — frontend
cd frontend
npm run dev
```

Open **http://localhost:5173** in your browser.

---

## Running — Standalone CLI mode

### Define the zone (first time only)
```bash
# Windows
venv\Scripts\activate
python zone_setup.py

# Linux/macOS
source venv/bin/activate
python zone_setup.py
```
A window opens showing the first frame of your source. Click at least 3 points to define the monitoring polygon. Press `Enter` to save to `config.json`. Press `Q` to quit without saving.

> **Note (Windows):** Click on the OpenCV window to give it keyboard focus before pressing Enter or Q.

### Run the detector
```bash
python main.py
```
Press `Q` in the video window (with the window focused) to stop.

---

## GUI walkthrough

### Settings tab
- Edit all configuration values: source URL, tuning parameters
- Click **Save** to write values to `.env`
- Click **Define zone** to open the zone editor

> API keys are managed directly in `.env` and are not exposed in the GUI.

### Live Monitoring tab
1. Click **Edit zone** (or **Define zone**) — the zone editor opens on the first frame of your source. Click to place polygon points (minimum 3), then click **Save zone**
2. Click **▶ Start** — the annotated video feed appears as a live stream
3. The **Event Log** updates in real time showing: zone entries/exits, Gemini verdicts, Claude judgments with confidence and reason
4. When an anomaly is confirmed at high or medium confidence, the zone overlay turns red and a badge appears
5. For file sources: a progress bar shows playback position below the video. It resets to zero when you stop and replay
6. The panel header shows **"Live Feed"** for RTSP sources and **"Video File"** for local files
7. Click **■ Stop** to halt detection

### Other tabs
Review, Camera Health, Model Insights — show mock dashboard data. Real integration is a future milestone.

---

## Configuration reference

All values can be set in `.env` or via the GUI Settings tab.

| Key | Default | Description |
|---|---|---|
| `SOURCE` | — | RTSP URL or local video file path |
| `GEMINI_API_KEY` | — | Google Gemini API key |
| `ANTHROPIC_API_KEY` | — | Anthropic API key |
| `YOLO_CONFIDENCE` | `0.5` | Minimum confidence for person detection (0–1) |
| `DWELL_SECONDS` | `2.0` | Seconds in zone before interval analysis fires |
| `GEMINI_INTERVAL_SECONDS` | `5.0` | How often Gemini re-analyses during extended stays |
| `GEMINI_KEYFRAMES` | `8` | Number of frames sent to Gemini per call |
| `CLAUDE_FRAMES_DIRECT` | `20` | Max frames sent to Claude per call |
| `ZONE_ENTRY_GRACE_SECONDS` | `0.3` | Continuous presence required before zone entry is confirmed |
| `ZONE_EXIT_GRACE_SECONDS` | `3.0` | Continuous absence required before zone exit is confirmed |
| `POST_EXIT_BUFFER_SECONDS` | `20.0` | Seconds of post-departure frames captured before exit analysis (stream mode) |
| `TRACKER_MATCH_DIST` | `120` | Max pixel distance to match a detection to an existing tracker |
| `GEMINI_AUTO_DISABLE_AFTER` | `3` | Consecutive Gemini failures before auto-disable |
| `GEMINI_COOLDOWN_MINUTES` | `10` | Minutes Gemini stays disabled after auto-disable |

---

## Troubleshooting

**`ValueError: invalid literal for int() with base 10: '10.0'`**
Your `.env` has a float value where an integer is expected (e.g. `GEMINI_COOLDOWN_MINUTES=10.0`). Change it to `10`.

**`yolov8m.pt` download fails**
The model downloads from Ultralytics on first run. If your network blocks it, manually download from https://github.com/ultralytics/assets/releases and place in the project root.

**Stream timeouts on RTSP**
Normal on hotspot/mobile connections. The threaded capture handles reconnection automatically. If timeouts are frequent, try reducing `GEMINI_INTERVAL_SECONDS` to reduce CPU load during YOLO inference.

**"No source configured" in Live tab after saving settings**
Restart the backend (`Ctrl+C` and re-run `uvicorn`). The `.env` is read at startup; changes take effect on next start.

**Port already in use**
```bash
# Windows — find and kill process on port 8000
netstat -ano | findstr :8000
taskkill /PID <pid> /F
```

**OpenCV window not responding to keys (Windows)**
Click directly on the OpenCV window title bar to give it focus, then press keys.

---

## License

MIT — Coded by Fotios Basagiannis
