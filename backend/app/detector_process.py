"""
Runs the detector (main.py logic) in a way that:
- Streams annotated frames as JPEG bytes via a queue
- Emits structured events (zone entry/exit, Gemini, Claude) via a queue
- Can be started/stopped by the API
"""

import cv2
import json
import os
import time
import threading
import base64
import queue
import numpy as np
from pathlib import Path
from dotenv import load_dotenv
from ultralytics import YOLO
from google import genai
from google.genai import types
import anthropic

# ── shared state exposed to FastAPI ──────────────────────────────────────────

frame_queue: queue.Queue = queue.Queue(maxsize=4)   # latest annotated frame bytes
event_queue: queue.Queue = queue.Queue(maxsize=200) # structured event dicts
_stop_event = threading.Event()
_detector_thread: threading.Thread | None = None
_status = {"running": False, "source": "", "resolution": "", "fps": 0.0, "error": ""}

def get_status() -> dict:
    return dict(_status)

def is_running() -> bool:
    return _status["running"]

def push_event(event: dict):
    try:
        event_queue.put_nowait(event)
    except queue.Full:
        try:
            event_queue.get_nowait()
            event_queue.put_nowait(event)
        except Exception:
            pass

def push_frame(jpeg_bytes: bytes):
    try:
        frame_queue.put_nowait(jpeg_bytes)
    except queue.Full:
        try:
            frame_queue.get_nowait()
            frame_queue.put_nowait(jpeg_bytes)
        except Exception:
            pass

# ── Inline detector (same logic as main.py, adapted for subprocess-free use) ─

def load_config():
    env_path = Path(__file__).parents[2] / ".env"
    load_dotenv(env_path, override=True)
    return {
        "YOLO_CONFIDENCE":      float(os.getenv("YOLO_CONFIDENCE", "0.5")),
        "DWELL_SECONDS":        float(os.getenv("DWELL_SECONDS", "2.0")),
        "GEMINI_INTERVAL":      float(os.getenv("GEMINI_INTERVAL_SECONDS", "5.0")),
        "GEMINI_KEYFRAMES":     int(float(os.getenv("GEMINI_KEYFRAMES", "8"))),
        "CLAUDE_FRAMES_DIRECT": int(float(os.getenv("CLAUDE_FRAMES_DIRECT", "20"))),
        "ZONE_EXIT_GRACE":      float(os.getenv("ZONE_EXIT_GRACE_SECONDS", "3.0")),
        "ZONE_ENTRY_GRACE":     float(os.getenv("ZONE_ENTRY_GRACE_SECONDS", "0.3")),
        "TRACKER_MATCH_DIST":   int(float(os.getenv("TRACKER_MATCH_DIST", "120"))),
        "POST_EXIT_BUFFER":     float(os.getenv("POST_EXIT_BUFFER_SECONDS", "20.0")),
        "GEMINI_AUTO_DISABLE_AFTER": int(float(os.getenv("GEMINI_AUTO_DISABLE_AFTER", "3"))),
        "GEMINI_COOLDOWN_MINUTES":   int(float(os.getenv("GEMINI_COOLDOWN_MINUTES", "10"))),
        "GEMINI_API_KEY":       os.getenv("GEMINI_API_KEY", ""),
        "ANTHROPIC_API_KEY":    os.getenv("ANTHROPIC_API_KEY", ""),
        "SOURCE":               os.getenv("SOURCE", ""),
    }

def run_detector(zone_points: list[tuple[int,int]] | None = None):
    """Main detector loop. Runs in a thread. Pushes frames and events."""
    global _detector_thread

    cfg = load_config()
    source = cfg["SOURCE"]

    if not source:
        _status["error"] = "SOURCE not set in .env"
        _status["running"] = False
        push_event({"type": "error", "message": "SOURCE not set in .env"})
        return

    _status["running"] = True
    _status["source"] = source
    _status["error"] = ""

    # Clients
    gemini_client = genai.Client(api_key=cfg["GEMINI_API_KEY"]) if cfg["GEMINI_API_KEY"] else None
    claude_client = anthropic.Anthropic(api_key=cfg["ANTHROPIC_API_KEY"]) if cfg["ANTHROPIC_API_KEY"] else None

    # Gemini auto-disable state
    gemini_fail_count = [0]
    gemini_disabled_until = [0.0]

    def gemini_available():
        if not gemini_client:
            return False
        if time.time() < gemini_disabled_until[0]:
            return False
        if gemini_disabled_until[0] > 0 and time.time() >= gemini_disabled_until[0]:
            gemini_disabled_until[0] = 0.0
            push_event({"type": "system", "message": "Gemini re-enabled after cooldown"})
        return True

    is_file = not any(source.lower().startswith(p) for p in ("rtsp://","rtmp://","http://","https://"))
    _status["source_type"] = "file" if is_file else "stream"

    # YOLO
    model = YOLO("yolov8m.pt")

    # Stream capture thread
    latest_frame = [None]
    frame_lock = threading.Lock()
    finished_flag = [False]
    native_fps = [25.0]
    total_frames = [0]
    current_frame_num = [0]

    def capture_thread():
        cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
        if is_file:
            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps and fps > 0:
                native_fps[0] = fps
            tot = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            if tot and tot > 0:
                total_frames[0] = int(tot)
        interval = 1.0 / native_fps[0] if is_file else 0
        while not _stop_event.is_set():
            t = time.time()
            ret, frame = cap.read()
            if not ret:
                if is_file:
                    finished_flag[0] = True
                    break
                time.sleep(2)
                cap.open(source)
                continue
            with frame_lock:
                latest_frame[0] = frame
            current_frame_num[0] += 1
            if is_file and interval > 0:
                elapsed = time.time() - t
                sl = interval - elapsed
                if sl > 0:
                    time.sleep(sl)
        cap.release()

    cap_t = threading.Thread(target=capture_thread, daemon=True)
    cap_t.start()

    # Wait for first frame
    for _ in range(50):
        with frame_lock:
            if latest_frame[0] is not None:
                break
        time.sleep(0.1)

    with frame_lock:
        if latest_frame[0] is None:
            _status["error"] = "Could not read frames from source"
            _status["running"] = False
            return
        h, w = latest_frame[0].shape[:2]

    _status["resolution"] = f"{w}x{h}"
    push_event({"type": "system", "message": f"Stream open: {w}x{h} @ {native_fps[0]:.0f}fps"})

    # Zone polygon
    if zone_points and len(zone_points) >= 3:
        zone_poly = np.array(zone_points, dtype="int32")
    else:
        zone_poly = None

    # Colours
    COL_ZONE_IDLE   = (0, 200, 100)
    COL_ZONE_ACTIVE = (0, 255, 180)
    COL_ZONE_ALERT  = (0, 80, 255)
    COL_BOX_IN      = (0, 220, 80)
    COL_BOX_OUT     = (120, 120, 120)
    COL_LABEL_IN    = (0, 240, 100)
    COL_LABEL_OUT   = (150, 150, 150)

    # ── Prompts (same as main.py) ─────────────────────────────────────────────
    ANOMALY_PROMPT = """You are a trained security analyst reviewing sequential frames from a surveillance camera.

Your task is to assess whether anything in these frames constitutes a surveillance anomaly.

VIOLENCE & PHYSICAL THREAT
- Physical assault, fighting, punching, kicking, wrestling, or aggressive physical contact
- Abuse or threatening behaviour directed at another person
- A person being restrained, dragged, or forced against their will
- Aggressive or menacing posture — raising fists, aggressive gesturing
- Attacking someone from behind or by surprise

WEAPONS & DANGEROUS OBJECTS
- Any visible firearm, knife, blade, bat, club, or object carried in a threatening manner
- A person reaching for or drawing a concealed item suspiciously
- Large improvised weapons — heavy objects lifted or swung

CRIMINAL ACTIVITY
- Robbery or mugging — confronting someone, demanding or taking their belongings
- Burglary — breaking into or gaining unauthorised access to a space
- Theft or shoplifting — concealing items, taking things without apparent payment
- Vandalism — breaking, damaging, defacing, or marking property or surfaces:
    * Repeated arm/hand movement against a wall or surface across multiple frames
    * A person facing a wall making sustained contact for no apparent legitimate reason
    * Spray-painting, scratching, writing motions even if result not yet visible

FIRE & DESTRUCTION
- Visible fire, smoke, or flames anywhere in the scene
- A person deliberately igniting or causing fire (arson)
- A spreading liquid on the ground that was not there before, especially if it ignites
- Deliberate destruction of property or infrastructure

DISTRESS & MEDICAL EMERGENCY
- A person who has fallen and is not getting up
- Someone who appears injured, unconscious, or in physical distress
- A person on the ground who was not there before

SUSPICIOUS BEHAVIOUR
- Loitering — remaining far longer than any normal task would require, with no obvious purpose
- Repeatedly checking surroundings, looking over the shoulder, scanning exits
- Attempting to conceal face, body, or actions from the camera
- Erratic, frantic, or panicked movement inconsistent with the environment
- Running or sprinting where people normally walk

ACCESS & INTRUSION
- Tailgating — following closely through an access point
- Climbing over, jumping, or forcing past barriers
- Moving counter to normal traffic flow in a way suggesting evasion

ABANDONED OBJECTS
- A person leaving a bag, package, or object unattended and walking away

CROWD ANOMALIES
- Sudden crowd dispersal or panic
- People gathering urgently around a fallen or distressed individual

ENVIRONMENTAL CONSEQUENCES — CRITICAL
Look not only at what the person does, but what happens TO THE ENVIRONMENT as a result.
- Fire, flames, or smoke appearing anywhere — even in the background, even if small
- A spreading liquid on the ground that was not there before, especially if it ignites
- Damage appearing on surfaces that was not present in earlier frames
- A person on the ground who was not there in earlier frames
- Broken glass, scattered objects, or debris that appeared after a person's actions
- DO NOT require the person to still be present. If early frames show normal activity
  and later frames show fire, smoke, damage — that IS an anomaly.

WHAT IS NOT AN ANOMALY
- A person walking normally through the zone
- Delivery personnel carrying parcels purposefully
- People pausing briefly to use a phone, check directions, or wait for someone
- Normal purposeful activity consistent with the environment
"""

    GEMINI_PROMPT = ANOMALY_PROMPT + """
You are the FIRST-PASS filter. Be sensitive.

Pay close attention to:
- Changes in the number of people visible across frames
- Any person who ends up on the ground
- Sudden rapid movement directed at another person
- Changes in body posture suggesting impact, struggle, or distress
- ANY change in the environment between first and last frames

Reply with EXACTLY this format — two lines, nothing else:
VERDICT: <clear_no|maybe|likely>
DESCRIPTION: <one sentence describing what you observe>

Use clear_no ONLY when ALL of these are true:
  - Scene is unambiguously normal across every frame
  - No interaction with walls or surfaces in unusual way
  - No loitering, unusual posture, or repeated contact with any surface
  - You are confident a security professional would have zero interest
  When in doubt between clear_no and maybe, always choose maybe.
Use maybe when anything could plausibly be anomalous.
Use likely when something is clearly or strongly anomalous."""

    CLAUDE_DEEP = ANOMALY_PROMPT + """
You are the SECOND-PASS reviewer. A first AI flagged this scene as potentially anomalous.
Make the final structured judgment.

These frames span the COMPLETE duration including time after the person left.

CRITICAL INSTRUCTIONS:
1. Compare FIRST frame to LAST frame. Did the environment change? Fire, smoke, damage,
   liquid, person on ground? If yes — anomaly even if person no longer visible.
2. Look for REPEATED or SUSTAINED behaviour across frames. Pattern is the signal.
3. Do not require visible damage or weapon to flag. Sustained purposeless contact = anomaly.

First AI's observation: {gemini_description}

Return ONLY valid JSON:
{{
  "detected": true or false,
  "confidence": "high" or "medium" or "low",
  "anomaly_type": "assault|fighting|abuse|weapon|robbery|burglary|theft|shoplifting|vandalism|arson|fire|person_down|distress|loitering|suspicious_behaviour|intrusion|tailgating|abandoned_object|crowd_anomaly|other|none",
  "reason": "one concise sentence"
}}"""

    CLAUDE_SOLO = ANOMALY_PROMPT + """
You are the SOLE analyst — no prior AI has reviewed this scene.

CRITICAL INSTRUCTIONS:
1. Compare FIRST frame to LAST frame. Did the environment change?
2. Look for REPEATED or SUSTAINED behaviour across frames. Pattern is the signal.
3. Do not require visible damage or weapon to flag.

Return ONLY valid JSON:
{{
  "detected": true or false,
  "confidence": "high" or "medium" or "low",
  "anomaly_type": "assault|fighting|abuse|weapon|robbery|burglary|theft|shoplifting|vandalism|arson|fire|person_down|distress|loitering|suspicious_behaviour|intrusion|tailgating|abandoned_object|crowd_anomaly|other|none",
  "reason": "one concise sentence"
}}"""

    # ── API call functions ────────────────────────────────────────────────────

    def call_gemini(frames):
        if not gemini_available():
            return None, None
        parts = []
        for f in frames:
            _, buf = cv2.imencode(".jpg", f, [cv2.IMWRITE_JPEG_QUALITY, 85])
            parts.append(types.Part.from_bytes(data=bytes(buf), mime_type="image/jpeg"))
        prompt = f"These {len(frames)} images are sequential surveillance camera frames.\n\n" + GEMINI_PROMPT
        contents = parts + [types.Part.from_text(text=prompt)]
        for model_name in ["gemini-2.5-flash", "gemini-2.0-flash"]:
            for attempt in range(2):
                try:
                    resp = gemini_client.models.generate_content(model=model_name, contents=contents)
                    text = resp.text.strip()
                    verdict, description = "maybe", text
                    for line in text.splitlines():
                        line = line.strip()
                        if line.upper().startswith("VERDICT:"):
                            raw = line.split(":", 1)[1].strip().lower()
                            if "clear_no" in raw or "clear no" in raw:
                                verdict = "clear_no"
                            elif "likely" in raw:
                                verdict = "likely"
                            else:
                                verdict = "maybe"
                        elif line.upper().startswith("DESCRIPTION:"):
                            description = line.split(":", 1)[1].strip()
                    gemini_fail_count[0] = 0
                    return verdict, description
                except Exception as e:
                    err = str(e)
                    is_503 = "503" in err or "UNAVAILABLE" in err.upper()
                    last_attempt = attempt == 1
                    last_model = model_name == "gemini-2.0-flash"
                    if is_503 and last_attempt and not last_model:
                        break
                    elif last_attempt and last_model:
                        gemini_fail_count[0] += 1
                        if gemini_fail_count[0] >= cfg["GEMINI_AUTO_DISABLE_AFTER"]:
                            gemini_disabled_until[0] = time.time() + cfg["GEMINI_COOLDOWN_MINUTES"] * 60
                            gemini_fail_count[0] = 0
                            push_event({"type": "system", "message": f"Gemini auto-disabled for {cfg['GEMINI_COOLDOWN_MINUTES']}min"})
                        return None, None
                    else:
                        time.sleep(1)
        return None, None

    def call_claude(frames, gemini_description=None):
        if not claude_client:
            return {"detected": False, "confidence": "low", "anomaly_type": "none", "reason": "No Anthropic API key"}
        n = cfg["CLAUDE_FRAMES_DIRECT"]
        if len(frames) <= n:
            sampled = frames
        else:
            half = n // 2
            step = max(1, len(frames) // half)
            even = frames[::step][:half]
            tail_start = max(0, len(frames) * 3 // 4)
            tail = frames[tail_start:]
            tail_step = max(1, len(tail) // half)
            tail_s = tail[::tail_step][:half]
            seen, sampled = set(), []
            for f in even + tail_s:
                fid = id(f)
                if fid not in seen:
                    seen.add(fid)
                    sampled.append(f)
        content = []
        for f in sampled:
            _, buf = cv2.imencode(".jpg", f, [cv2.IMWRITE_JPEG_QUALITY, 85])
            b64 = base64.standard_b64encode(bytes(buf)).decode()
            content.append({"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": b64}})
        prompt = CLAUDE_SOLO if gemini_description is None else CLAUDE_DEEP.format(gemini_description=gemini_description)
        content.append({"type": "text", "text": prompt})
        for attempt in range(2):
            try:
                resp = claude_client.messages.create(
                    model="claude-sonnet-4-6",
                    max_tokens=300,
                    messages=[{"role": "user", "content": content}]
                )
                raw = resp.content[0].text.strip().replace("```json","").replace("```","").strip()
                start, end = raw.find("{"), raw.rfind("}") + 1
                return json.loads(raw[start:end])
            except Exception as e:
                if attempt == 1:
                    return {"detected": False, "confidence": "low", "anomaly_type": "none", "reason": str(e)[:80]}
                time.sleep(1)
        return {"detected": False, "confidence": "low", "anomaly_type": "none", "reason": "Claude unavailable"}

    # ── ZoneTracker (identical logic to main.py) ──────────────────────────────

    _tracker_id_counter = [0]

    class ZoneTracker:
        def __init__(self, first_seen):
            _tracker_id_counter[0] += 1
            self.id = _tracker_id_counter[0]
            self.first_seen = first_seen
            self.last_cx = self.last_cy = 0
            self.entry_confirmed = self.entry_logged = self.entry_trigger_fired = False
            self.entry_time = None
            self.last_seen = first_seen
            self.last_gemini_time = None
            self.frame_buffer = []
            self.gemini_running = self.claude_running = False
            self.last_gemini_verdict = self.last_gemini_description = None
            self.last_claude_result = None
            self.exit_time = None
            self.post_exit_frames = []
            self.exit_fired = False
            self._analysis_buffer = []

        def update(self, frame, cx, cy, now):
            self.last_cx, self.last_cy, self.last_seen = cx, cy, now
            self.frame_buffer.append(frame.copy())
            if len(self.frame_buffer) > 500:
                self.frame_buffer = self.frame_buffer[-500:]
            if not self.entry_confirmed and (now - self.first_seen) >= cfg["ZONE_ENTRY_GRACE"]:
                self.entry_confirmed = True
                self.entry_time = self.first_seen
            if self.entry_confirmed and not self.entry_logged:
                self.entry_logged = True
                push_event({"type": "zone_entry", "tracker_id": self.id, "message": f"Person #{self.id} entered zone"})

        def dwell(self, now):
            return now - self.entry_time if self.entry_time else 0.0

        def should_fire_entry(self):
            return self.entry_confirmed and not self.entry_trigger_fired and not self.gemini_running

        def should_fire_interval(self, now):
            if not self.entry_confirmed or self.dwell(now) < cfg["DWELL_SECONDS"] or self.gemini_running:
                return False
            if self.last_gemini_time is None:
                return False
            return (now - self.last_gemini_time) >= cfg["GEMINI_INTERVAL"]

        def fire(self, now, trigger_label):
            if len(self.frame_buffer) < 2:
                return
            self._analysis_buffer = list(self.frame_buffer)
            step = max(1, len(self._analysis_buffer) // cfg["GEMINI_KEYFRAMES"])
            gframes = self._analysis_buffer[::step][:cfg["GEMINI_KEYFRAMES"]]
            self.last_gemini_time = now
            self.gemini_running = True
            if trigger_label == "entry":
                self.entry_trigger_fired = True
            push_event({"type": "gemini_start", "tracker_id": self.id, "trigger": trigger_label, "frames": len(gframes)})
            threading.Thread(target=self._gemini_worker, args=(gframes, trigger_label), daemon=True).start()

        def begin_exit(self, now):
            self.exit_time = now

        def add_post_exit_frame(self, frame):
            self.post_exit_frames.append(frame.copy())

        def should_fire_exit(self, now):
            return not self.exit_fired and self.exit_time is not None and (now - self.exit_time) >= cfg["POST_EXIT_BUFFER"]

        def fire_exit(self):
            self.exit_fired = True
            combined = list(self.frame_buffer) + list(self.post_exit_frames)
            if len(combined) < 2:
                return
            self._analysis_buffer = combined
            step = max(1, len(combined) // cfg["GEMINI_KEYFRAMES"])
            gframes = combined[::step][:cfg["GEMINI_KEYFRAMES"]]
            push_event({"type": "gemini_start", "tracker_id": self.id, "trigger": "exit",
                        "frames": len(gframes), "post_exit": len(self.post_exit_frames)})
            threading.Thread(target=self._gemini_worker, args=(gframes, "exit"), daemon=True).start()

        def _gemini_worker(self, frames, trigger_label):
            verdict, description = call_gemini(frames)
            self.gemini_running = False
            full_buf = list(self._analysis_buffer) if self._analysis_buffer else list(self.frame_buffer)

            if verdict is None:
                if trigger_label == "entry":
                    return
                if not self.claude_running:
                    self.claude_running = True
                    push_event({"type": "gemini_result", "tracker_id": self.id, "verdict": "bypassed",
                                "description": "Gemini unavailable — going direct to Claude"})
                    threading.Thread(target=self._claude_worker, args=(full_buf, None), daemon=True).start()
                return

            self.last_gemini_verdict = verdict
            self.last_gemini_description = description
            push_event({"type": "gemini_result", "tracker_id": self.id, "verdict": verdict,
                        "description": description, "trigger": trigger_label})

            if verdict == "clear_no" and trigger_label == "entry":
                return

            if not self.claude_running:
                self.claude_running = True
                ctx = None if verdict == "clear_no" else description
                threading.Thread(target=self._claude_worker, args=(full_buf, ctx), daemon=True).start()

        def _claude_worker(self, frames, gemini_description=None):
            result = call_claude(frames, gemini_description)
            self.last_claude_result = result
            self.claude_running = False
            push_event({"type": "claude_result", "tracker_id": self.id, **result})

    def match_trackers(trackers, cx, cy):
        best_id, best_dist = None, float("inf")
        for tid, tracker in trackers.items():
            dist = ((tracker.last_cx - cx)**2 + (tracker.last_cy - cy)**2)**0.5
            if dist < best_dist:
                best_dist = dist
                best_id = tid
        return best_id if best_dist <= cfg["TRACKER_MATCH_DIST"] else None

    # ── Drawing helpers ───────────────────────────────────────────────────────

    last_claude_result = [None]

    def draw_zone(frame, poly, anyone, alert):
        if poly is None:
            return
        overlay = frame.copy()
        col = COL_ZONE_ALERT if alert else COL_ZONE_ACTIVE if anyone else COL_ZONE_IDLE
        alpha = 0.22 if alert else 0.12 if anyone else 0.08
        cv2.fillPoly(overlay, [poly], col)
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        cv2.polylines(frame, [poly], True, col, 2)

    def draw_box(frame, box, in_zone, dwell, tid):
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])
        col_box = COL_BOX_IN if in_zone else COL_BOX_OUT
        col_lbl = COL_LABEL_IN if in_zone else COL_LABEL_OUT
        cx, cy = (x1+x2)//2, (y1+y2)//2
        cv2.rectangle(frame, (x1,y1), (x2,y2), col_box, 2)
        cv2.circle(frame, (cx,cy), 4, col_box, -1)
        label = f"#{tid} {conf:.2f} dwell:{dwell:.1f}s" if in_zone else f"person {conf:.2f}"
        lx, ly = x1, max(y1-8, 16)
        cv2.putText(frame, label, (lx,ly), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 3, cv2.LINE_AA)
        cv2.putText(frame, label, (lx,ly), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col_lbl, 1, cv2.LINE_AA)

    def draw_hud(frame, fps, persons, anyone, last_gemini_v, last_claude_r, progress_str):
        fh, fw = frame.shape[:2]
        status = "ZONE ACTIVE" if anyone else "watching..."
        scol = (0,220,80) if anyone else (160,160,160)
        lines = [(f"fps:{fps:.0f} persons:{persons}" + (f"  {progress_str}" if progress_str else ""), (180,180,180)),
                 (status, scol)]
        if last_gemini_v:
            vcol = {"likely":(0,220,80),"maybe":(0,200,220),"clear_no":(150,150,150)}.get(last_gemini_v,(150,150,150))
            lines.append((f"Gemini:{last_gemini_v.replace('_',' ')}", vcol))
        if last_claude_r:
            conf = last_claude_r.get("confidence","")
            atype = last_claude_r.get("anomaly_type","").replace("_"," ")
            det = last_claude_r.get("detected", False)
            ccol = (0,80,255) if det and conf=="high" else (0,200,220) if det else (150,150,150)
            lines.append((f"Claude:{atype} [{conf}]", ccol))
        for i,(txt,col) in enumerate(lines):
            y = fh - 14 - (len(lines)-1-i)*22
            cv2.putText(frame, txt, (10,y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 3, cv2.LINE_AA)
            cv2.putText(frame, txt, (10,y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1, cv2.LINE_AA)

    # ── Main loop ─────────────────────────────────────────────────────────────

    trackers = {}
    last_seen_times = {}
    pending_exit = {}
    prev_time = time.time()
    frame_count = 0
    fps = 0.0
    last_gemini_display = ""
    last_claude_display = None

    push_event({"type": "system", "message": "Detector started"})

    while not _stop_event.is_set():
        if is_file and finished_flag[0]:
            push_event({"type": "system", "message": "File playback complete — flushing trackers"})
            now = time.time()
            for tid, tracker in list(trackers.items()):
                if tracker.entry_logged and not tracker.exit_fired:
                    push_event({"type": "zone_exit", "tracker_id": tracker.id,
                                "dwell": round(tracker.dwell(now), 1), "message": f"Person #{tracker.id} left zone (end of file)"})
                    tracker.begin_exit(now)
                    pending_exit[tracker.id] = tracker
                elif not tracker.entry_logged and len(tracker.frame_buffer) >= 2:
                    tracker.entry_confirmed = True
                    tracker.entry_time = tracker.first_seen
                    tracker.begin_exit(now)
                    pending_exit[tracker.id] = tracker
            trackers.clear()
            for tid, tracker in list(pending_exit.items()):
                if not tracker.exit_fired:
                    tracker.fire_exit()
            deadline = time.time() + 60
            while time.time() < deadline and not _stop_event.is_set():
                if all(not t.gemini_running and not t.claude_running for t in pending_exit.values()):
                    time.sleep(1)
                    break
                time.sleep(0.2)
            _status["progress"] = {"current": 0, "total": 0, "pct": 100, "finished": True}
            push_event({"type": "system", "message": "Detector finished"})
            push_event({"type": "playback_finished"})
            break

        with frame_lock:
            frame = latest_frame[0]
        if frame is None:
            time.sleep(0.02)
            continue

        frame = frame.copy()
        frame_count += 1
        now = time.time()
        elapsed = now - prev_time
        if elapsed >= 0.5:
            fps = frame_count / elapsed
            _status["fps"] = round(fps, 1)
            frame_count = 0
            prev_time = now

        results = model(frame, verbose=False, classes=[0])[0]
        anyone_in_zone = False
        matched_tids = set()

        for box in results.boxes:
            if float(box.conf[0]) < cfg["YOLO_CONFIDENCE"]:
                continue
            x1,y1,x2,y2 = map(int, box.xyxy[0])
            cx, cy = (x1+x2)//2, (y1+y2)//2
            in_zone = False
            if zone_poly is not None:
                in_zone = cv2.pointPolygonTest(zone_poly, (float(cx), float(cy)), False) >= 0

            dwell = 0.0
            if in_zone:
                anyone_in_zone = True
                tid = match_trackers(trackers, cx, cy)
                if tid is None:
                    t = ZoneTracker(now)
                    trackers[t.id] = t
                    tid = t.id
                tracker = trackers[tid]
                tracker.update(frame, cx, cy, now)
                last_seen_times[tid] = now
                matched_tids.add(tid)
                dwell = tracker.dwell(now)
                if tracker.should_fire_entry():
                    tracker.fire(now, "entry")
                if tracker.should_fire_interval(now):
                    tracker.fire(now, "interval")
                if tracker.last_gemini_verdict:
                    last_gemini_display = tracker.last_gemini_verdict
                if tracker.last_claude_result:
                    last_claude_display = tracker.last_claude_result
                    last_claude_result[0] = tracker.last_claude_result

            draw_box(frame, box, in_zone, dwell,
                     trackers[tid].id if in_zone and tid in trackers else 0)

        for tracker in pending_exit.values():
            tracker.add_post_exit_frame(frame)

        if not is_file:
            for tid in [k for k,t in pending_exit.items() if t.should_fire_exit(now)]:
                tracker = pending_exit.pop(tid)
                tracker.fire_exit()

        to_remove = []
        for tid, tracker in trackers.items():
            if tid not in matched_tids:
                absent = now - last_seen_times.get(tid, now)
                if absent >= cfg["ZONE_EXIT_GRACE"]:
                    to_remove.append(tid)
        for tid in to_remove:
            tracker = trackers.pop(tid)
            last_seen_times.pop(tid, None)
            if tracker.entry_logged:
                push_event({"type": "zone_exit", "tracker_id": tracker.id,
                            "dwell": round(tracker.dwell(now), 1),
                            "message": f"Person #{tracker.id} left zone (dwell {tracker.dwell(now):.1f}s)"})
                tracker.begin_exit(now)
                pending_exit[tracker.id] = tracker

        alert_active = (last_claude_result[0] is not None and
                        last_claude_result[0].get("detected") and
                        last_claude_result[0].get("confidence") in ("high","medium"))

        progress_str = ""
        if is_file and total_frames[0] > 0:
            cur = current_frame_num[0]
            pct = min(100, int(cur * 100 / total_frames[0]))
            es = int(cur / native_fps[0])
            ts = int(total_frames[0] / native_fps[0])
            progress_str = f"{es//60:02d}:{es%60:02d}/{ts//60:02d}:{ts%60:02d} ({pct}%)"
            _status["progress"] = {"current": es, "total": ts, "pct": pct, "finished": False}
        elif not is_file:
            _status["progress"] = None  # stream mode — no progress

        draw_zone(frame, zone_poly, anyone_in_zone, alert_active)
        draw_hud(frame, fps, len(results.boxes), anyone_in_zone,
                 last_gemini_display, last_claude_display, progress_str)

        _, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 65])
        push_frame(jpeg.tobytes())

    _status["running"] = False


def start(zone_points=None):
    global _detector_thread, _stop_event
    if _status["running"]:
        return False
    _stop_event.clear()
    _detector_thread = threading.Thread(target=run_detector, args=(zone_points,), daemon=True)
    _detector_thread.start()
    return True


def stop():
    global _stop_event
    _stop_event.set()
    _status["running"] = False
