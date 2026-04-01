# Multi-Agent Behavioral Video Analysis

An AI-powered video analytics system that detects and evaluates human behavior in defined zones using a three-layer intelligence architecture. Built as a focused MVP/POC, it combines local computer vision with cloud AI to deliver high-confidence behavioral detection with minimal false positives.

---

## How it works

Most video analytics systems fire an alert every time any person enters a zone. This system fires an alert when a person enters a zone **and their behavior matches what you actually care about** — described in plain English, with tolerance for ambiguity.

The difference between a janitor and an intruder. Between someone pausing to check their phone and someone casing the room.

This is achieved through three AI layers working in sequence:

```
RTSP Stream
    │
    ▼
┌─────────────────────────────────────────┐
│  Layer 1 — YOLO  (local, always-on)     │
│  Detects persons in the defined zone.   │
│  Gates all downstream processing.       │
│  Cost: local compute only               │
└────────────────────┬────────────────────┘
                     │ person confirmed in zone
                     ▼
┌─────────────────────────────────────────┐
│  Layer 2 — Gemini Flash                 │
│  Analyses a sequence of frames for      │
│  temporal context and motion patterns.  │
│  Returns: clear_no / maybe / likely     │
│  Cost: ~$16/month per camera            │
└────────────────────┬────────────────────┘
                     │ maybe or likely
                     ▼
┌─────────────────────────────────────────┐
│  Layer 3 — Claude                       │
│  Deep semantic judgment on sampled      │
│  frames + Gemini's description.         │
│  Returns: detected, confidence, reason  │
│  Cost: ~$45/month per camera            │
└────────────────────┬────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────┐
│  Alert Layer                            │
│  low confidence  → log only             │
│  medium/high     → console alert        │
│                  + snapshot saved       │
│  All events      → SQLite database      │
└─────────────────────────────────────────┘
```

**Total estimated API cost: ~$60/month per camera**

---

## Tech stack

| Component | Technology |
|---|---|
| Stream capture | OpenCV `VideoCapture` (RTSP + local files) |
| Person detection | YOLOv8n (Ultralytics) |
| Zone logic | OpenCV polygon intersection |
| Temporal analysis | Gemini Flash (`generate_content` with keyframes) |
| Semantic judgment | Claude Haiku / Sonnet |
| Event storage | SQLite |
| Configuration | `.env` + `config.json` |

---

## Project status

This project is being built milestone by milestone. Each milestone delivers working, demonstrable functionality.

| Milestone | Description | Status |
|---|---|---|
| 1 | RTSP capture, click-to-define zone, YOLO person detection | ✅ Complete |
| 2 | Gemini Flash integration — frame buffer, keyframe analysis, verdict + description | 🔲 Upcoming |
| 3 | Claude integration — structured JSON judgment, full two-AI reasoning chain | 🔲 Upcoming |
| 4 | Alert tiering, SQLite event log, snapshot saving, event viewer | 🔲 Upcoming |
| 5 | RTSP live testing, bug fixes, documentation and handover | 🔲 Upcoming |

---

## Requirements

- Python 3.11
- A camera exposing an RTSP stream, or a local `.mp4` video file
- Anthropic API key (Milestone 3+)
- Google Gemini API key (Milestone 2+)

---

## Installation

**1. Clone the repository:**
```bash
git clone https://github.com/fotiosb/Multi-Agent-Behavioral-Video-Analysis.git
cd Multi-Agent-Behavioral-Video-Analysis
```

**2. Create and activate a Python 3.11 virtual environment:**

On Windows:
```bash
py -3.11 -m venv venv
venv\Scripts\activate
```

On macOS/Linux:
```bash
python3.11 -m venv venv
source venv/bin/activate
```

**3. Install dependencies:**
```bash
pip install --upgrade pip
pip install opencv-python ultralytics python-dotenv
```

**4. Create your `.env` file:**
```
RTSP_URL=rtsp://your-camera-ip:port/stream
ANTHROPIC_API_KEY=your_key_here      # required from Milestone 3
GEMINI_API_KEY=your_key_here         # required from Milestone 2
```

---

## Usage

### Step 1 — Define your zone

Run the zone setup tool. It will open a window showing the first frame of your stream. Click to place polygon points around the area you want to monitor, then press `Enter` to confirm. The zone is saved to `config.json`.

```bash
python zone_setup.py
```

**Controls:**
- `Left-click` — place a point
- `R` — reset all points
- `Enter` — confirm and save (minimum 3 points required)
- `Q` — quit without saving

> **Note:** Click on the video window first to give it keyboard focus before using keyboard shortcuts.

### Step 2 — Start detection

```bash
python main.py
```

The annotated video feed opens in a window. The defined zone is shown as a green polygon. Detected persons outside the zone are shown with grey bounding boxes. Persons inside the zone are shown in green with a live dwell timer.

Press `Q` (with the video window focused) to stop.

---

## Configuration

Edit `config.json` to adjust the zone polygon (auto-generated by `zone_setup.py`).

The following constants can be adjusted at the top of `main.py`:

| Constant | Default | Description |
|---|---|---|
| `YOLO_CONFIDENCE` | `0.5` | Minimum confidence threshold for person detection |
| `DWELL_SECONDS` | `2.0` | Seconds a person must be in the zone before AI analysis is triggered |

---

## Stream compatibility

`cv2.VideoCapture` works identically for both RTSP streams and local video files. To test with a local file, set your `.env` to:

```
RTSP_URL=path\to\your\video.mp4
```

No other changes required.

---

## Design decisions

**Why three AI layers instead of one?**
YOLO alone fires on presence, not behavior. A single AI layer (e.g. Claude directly on every frame) would be expensive and slow. The three-layer architecture gates API calls aggressively — Gemini only activates when YOLO confirms a person has dwelled in the zone, and Claude only activates when Gemini flags something worth a second opinion. This keeps costs low while preserving sensitivity to genuinely ambiguous situations.

**Why keyframes instead of the Gemini Live streaming API?**
For this MVP, Gemini receives a buffer of evenly-sampled frames rather than a live WebSocket stream. This approach works identically for both local video files and RTSP streams, is significantly simpler to implement, and produces equivalent detection quality for the behavioral patterns this system targets. The Live API is a defined future upgrade path for sub-second latency requirements.

**Why threaded capture?**
YOLO inference on CPU takes longer than the interval between RTSP frames. Without a dedicated capture thread, the main loop blocks on network I/O waiting for the next frame while YOLO is still processing the previous one, causing buffer starvation and stream timeouts. The `StreamCapture` class runs frame reading in a background thread so the network and the inference pipeline never block each other.

---

## License

MIT
