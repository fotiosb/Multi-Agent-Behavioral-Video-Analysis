import cv2
import json
import os
import time
import threading
import numpy as np
from dotenv import load_dotenv
from ultralytics import YOLO

load_dotenv()

YOLO_CONFIDENCE = 0.5
DWELL_SECONDS = 2.0
WINDOW = "Video Analytics - Zone Detection  (Q to quit)"

# Colours (BGR)
COL_ZONE_IDLE   = (0, 200, 100)
COL_ZONE_ACTIVE = (0, 255, 180)
COL_BOX_IN      = (0, 220, 80)
COL_BOX_OUT     = (120, 120, 120)
COL_LABEL_IN    = (0, 240, 100)
COL_LABEL_OUT   = (150, 150, 150)


class StreamCapture:
    """Reads frames from RTSP in a background thread so the main loop never blocks on network I/O."""

    def __init__(self, url):
        self.url = url
        self.frame = None
        self.ok = False
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._reader, daemon=True)
        self._thread.start()

    def _open(self):
        cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)
        return cap

    def _reader(self):
        cap = self._open()
        while not self._stop.is_set():
            ret, frame = cap.read()
            if not ret:
                cap.release()
                time.sleep(2)
                cap = self._open()
                continue
            with self._lock:
                self.frame = frame
                self.ok = True

        cap.release()

    def read(self):
        with self._lock:
            if self.frame is None:
                return False, None
            return self.ok, self.frame.copy()

    def get_size(self):
        # Wait up to 5s for first frame
        for _ in range(50):
            with self._lock:
                if self.frame is not None:
                    h, w = self.frame.shape[:2]
                    return w, h
            time.sleep(0.1)
        return 640, 480

    def stop(self):
        self._stop.set()


def load_zone():
    if not os.path.exists("config.json"):
        print("ERROR: config.json not found. Run zone_setup.py first.")
        return None
    with open("config.json") as f:
        data = json.load(f)
    pts = [tuple(p) for p in data["zone"]]
    print(f"Zone loaded: {len(pts)} points")
    return np.array(pts, dtype="int32")


def person_in_zone(box, zone_poly):
    x1, y1, x2, y2 = map(int, box.xyxy[0])
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    return cv2.pointPolygonTest(zone_poly, (float(cx), float(cy)), False) >= 0


def draw_zone(frame, zone_poly, active):
    overlay = frame.copy()
    col = COL_ZONE_ACTIVE if active else COL_ZONE_IDLE
    cv2.fillPoly(overlay, [zone_poly], col)
    cv2.addWeighted(overlay, 0.12, frame, 0.88, 0, frame)
    cv2.polylines(frame, [zone_poly], True, col, 2)


def draw_box(frame, box, in_zone, dwell):
    x1, y1, x2, y2 = map(int, box.xyxy[0])
    conf = float(box.conf[0])
    col_box = COL_BOX_IN if in_zone else COL_BOX_OUT
    col_lbl = COL_LABEL_IN if in_zone else COL_LABEL_OUT
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    cv2.rectangle(frame, (x1, y1), (x2, y2), col_box, 2)
    cv2.circle(frame, (cx, cy), 4, col_box, -1)
    label = f"IN ZONE  {conf:.2f}  dwell: {dwell:.1f}s" if in_zone else f"person  {conf:.2f}"
    lx, ly = x1, max(y1 - 8, 16)
    cv2.putText(frame, label, (lx, ly), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(frame, label, (lx, ly), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, col_lbl, 1, cv2.LINE_AA)


def draw_hud(frame, fps, person_count, anyone_in_zone):
    h, w = frame.shape[:2]
    status = "ZONE ACTIVE - person detected" if anyone_in_zone else "watching..."
    status_col = (0, 220, 80) if anyone_in_zone else (160, 160, 160)
    lines = [
        (f"fps: {fps:.0f}  persons: {person_count}", (180, 180, 180)),
        (status, status_col),
    ]
    for i, (txt, col) in enumerate(lines):
        y = h - 14 - (len(lines) - 1 - i) * 22
        cv2.putText(frame, txt, (10, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(frame, txt, (10, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, col, 1, cv2.LINE_AA)


def main():
    rtsp_url = os.getenv("RTSP_URL")
    if not rtsp_url:
        print("ERROR: RTSP_URL not set in .env")
        return

    zone_poly = load_zone()
    if zone_poly is None:
        return

    print(f"Opening stream: {rtsp_url}")
    model = YOLO("yolov8n.pt")
    print("YOLO model ready.")

    stream = StreamCapture(rtsp_url)
    print("Waiting for first frame...")
    w, h = stream.get_size()
    print(f"Stream open: {w}x{h}")
    print("Watching for persons in zone... Press Q to quit.")
    print()

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW, min(w, 1280), min(h, 720))

    dwell_tracker = {}
    prev_time = time.time()
    frame_count = 0
    fps = 0.0

    while True:
        ok, frame = stream.read()
        if not ok or frame is None:
            time.sleep(0.05)
            continue

        frame_count += 1
        now = time.time()
        elapsed = now - prev_time
        if elapsed >= 0.5:
            fps = frame_count / elapsed
            frame_count = 0
            prev_time = now

        results = model(frame, verbose=False, classes=[0])[0]

        anyone_in_zone = False
        active_keys = set()

        for box in results.boxes:
            if float(box.conf[0]) < YOLO_CONFIDENCE:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            in_zone = person_in_zone(box, zone_poly)

            dwell = 0.0
            if in_zone:
                key = (cx // 80, cy // 80)
                active_keys.add(key)
                if key not in dwell_tracker:
                    dwell_tracker[key] = now
                    print(f"[{time.strftime('%H:%M:%S')}] Person entered zone at ~({cx},{cy})")
                dwell = now - dwell_tracker[key]
                anyone_in_zone = True

                if dwell >= DWELL_SECONDS:
                    pass  # Milestone 2: Gemini fires here

            draw_box(frame, box, in_zone, dwell)

        gone = set(dwell_tracker.keys()) - active_keys
        for key in gone:
            dwell = now - dwell_tracker.pop(key)
            print(f"[{time.strftime('%H:%M:%S')}] Person left zone - dwell was {dwell:.1f}s")

        draw_zone(frame, zone_poly, anyone_in_zone)
        draw_hud(frame, fps, len(results.boxes), anyone_in_zone)

        cv2.imshow(WINDOW, frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    stream.stop()
    try:
        cv2.destroyWindow(WINDOW)
        cv2.waitKey(1)
        cv2.destroyAllWindows()
        cv2.waitKey(1)
    except Exception:
        pass
    print("Stopped.")


if __name__ == "__main__":
    main()
