import cv2
import json
import os
from dotenv import load_dotenv

load_dotenv()

WINDOW = "Zone Setup — click to place points, Enter to confirm, R to reset"
points = []


def mouse_handler(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))
        print(f"  Point {len(points)}: ({x}, {y})")


def draw_overlay(frame, pts):
    overlay = frame.copy()
    if len(pts) >= 3:
        import numpy as np
        cv2.fillPoly(overlay, [np.array(pts, dtype="int32")], (0, 200, 100))
        cv2.addWeighted(overlay, 0.15, frame, 0.85, 0, frame)
        cv2.polylines(frame, [np.array(pts, dtype="int32")], True, (0, 200, 100), 2)
    for i, p in enumerate(pts):
        cv2.circle(frame, p, 6, (0, 200, 100), -1)
        cv2.circle(frame, p, 6, (255, 255, 255), 1)
        if i > 0:
            cv2.line(frame, pts[i - 1], p, (0, 200, 100), 2)
    if len(pts) >= 2:
        cv2.line(frame, pts[-1], pts[0], (0, 200, 100), 1)
    return frame


def main():
    rtsp_url = os.getenv("SOURCE")
    if not rtsp_url:
        print("ERROR: SOURCE not set in .env file.")
        return

    print(f"Connecting to: {rtsp_url}")
    cap = cv2.VideoCapture(rtsp_url)

    if not cap.isOpened():
        print("ERROR: Could not open stream. Check that Periscope HD is running and the URL is correct.")
        return

    ret, first_frame = cap.read()
    cap.release()

    if not ret:
        print("ERROR: Could not read frame from stream.")
        return

    h, w = first_frame.shape[:2]
    print(f"Frame size: {w}x{h}")
    print()
    print("Instructions:")
    print("  Left-click  — place a zone point")
    print("  R           — reset all points")
    print("  Enter       — confirm zone and save")
    print("  Q           — quit without saving")
    print()

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW, min(w, 1280), min(h, 720))
    cv2.setMouseCallback(WINDOW, mouse_handler)

    while True:
        display = first_frame.copy()
        draw_overlay(display, points)

        msg = "Click to place points. Enter=confirm  R=reset  Q=quit"
        cv2.putText(display, msg, (12, 28), cv2.FONT_HERSHEY_SIMPLEX,
                    0.65, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(display, msg, (12, 28), cv2.FONT_HERSHEY_SIMPLEX,
                    0.65, (255, 255, 255), 1, cv2.LINE_AA)

        count_msg = f"Points placed: {len(points)}  (need at least 3)"
        cv2.putText(display, count_msg, (12, 56), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(display, count_msg, (12, 56), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (200, 255, 200), 1, cv2.LINE_AA)

        cv2.imshow(WINDOW, display)
        key = cv2.waitKey(30) & 0xFF

        if key == ord("q"):
            print("Quit — no zone saved.")
            break

        elif key == ord("r"):
            points.clear()
            print("Points reset.")

        elif key in (13, 10):  # Enter
            if len(points) < 3:
                print(f"Need at least 3 points — you have {len(points)}. Keep clicking.")
            else:
                config = {"zone": points}
                with open("config.json", "w") as f:
                    json.dump(config, f, indent=2)
                print()
                print(f"Zone saved to config.json with {len(points)} points:")
                for p in points:
                    print(f"  {p}")
                print()
                print("Run main.py to start detection.")
                break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
