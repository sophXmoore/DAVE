"""
DAVE — TCP Sender
Runs alongside detect_shape_edges.py and streams detections to Rhino.

Usage:
    python tcp_sender.py
    (detect_shape_edges.py does NOT need to be modified — this script imports
    and calls its functions directly, adding the TCP stream on top.)

Press 'q' in the OpenCV window to quit. 's' snapshots detections.json.
"""

import sys
import json
import time
import socket
import threading
import cv2
import numpy as np

from detect_shape_edges import detect_shapes, save_detections, CAMERA_INDEX

# --- Config ---
HOST = "127.0.0.1"
PORT = 9877
SEND_INTERVAL = 1.0  # seconds between Rhino updates

# Map color name strings → RGB for Rhino object colors.
COLOR_RGB = {
    "red":   [220, 50,  50],
    "blue":  [50,  100, 220],
    "black": [30,  30,  30],
}

# Shared state written by capture loop, read by sender loop.
_latest_shapes = []
_lock = threading.Lock()


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

def normalize(detections):
    """
    Collapse the two schemas from detect_shape_edges into one:
      - filled shapes use 'coordinates' key
      - line strokes use 'points' key
    Both become 'points' here, and color strings become RGB lists.
    """
    out = []
    for d in detections:
        pts = d.get("points") or d.get("coordinates")
        if not pts:
            continue
        out.append({
            "id":     str(d["id"]),
            "color":  COLOR_RGB.get(d.get("color", "black"), [200, 200, 200]),
            "points": pts,
        })
    return out


# ---------------------------------------------------------------------------
# Sender thread
# ---------------------------------------------------------------------------

def sender_loop():
    """Connects to Rhino and sends the latest shapes every SEND_INTERVAL."""
    while True:
        try:
            print("[sender] Connecting to Rhino at {}:{}...".format(HOST, PORT))
            conn = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            conn.connect((HOST, PORT))
            print("[sender] Connected.")

            while True:
                with _lock:
                    shapes = list(_latest_shapes)

                payload = json.dumps({"shapes": shapes}) + "\n"
                conn.sendall(payload.encode("utf-8"))
                print("[sender] Sent {} shapes.".format(len(shapes)))
                time.sleep(SEND_INTERVAL)

        except ConnectionRefusedError:
            print("[sender] Rhino not ready, retrying in 2s...")
            time.sleep(2)
        except (BrokenPipeError, ConnectionResetError, OSError):
            print("[sender] Connection lost, retrying...")
            time.sleep(2)
        except Exception as e:
            print("[sender] Error: {}, retrying...".format(e))
            time.sleep(2)
        finally:
            try:
                conn.close()
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Main capture loop
# ---------------------------------------------------------------------------

def run():
    # Start sender in background — it will keep retrying until Rhino is up.
    t = threading.Thread(target=sender_loop, daemon=True)
    t.start()

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam (index {}). Try 1 or 2.".format(CAMERA_INDEX))

    last_print = time.time()
    print("[capture] Camera open. Press 'q' to quit, 's' to save detections.json.")

    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.05)
            continue

        frame = cv2.resize(frame, (960, 540))
        output, mask, detections = detect_shapes(frame)

        # Push normalized detections into shared state for the sender.
        with _lock:
            _latest_shapes.clear()
            _latest_shapes.extend(normalize(detections))

        cv2.imshow("Webcam", frame)
        cv2.imshow("Color Mask", mask)
        cv2.imshow("Detected Shapes", output)

        now = time.time()
        if now - last_print >= 5.0:
            print(json.dumps(detections, indent=2), flush=True)
            last_print = now

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("s"):
            save_detections(detections, "detections.json")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run()
