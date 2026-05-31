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
# Shape tracker — assigns stable cross-frame IDs
# ---------------------------------------------------------------------------

class ShapeTracker:
    """
    Matches detected shapes frame-to-frame by color + shape type + centroid
    proximity, issuing stable IDs so Rhino can track objects across frames.
    """
    MATCH_DIST = 60   # px — max centroid movement to consider same shape
    EMA_ALPHA  = 0.35 # smoothing factor: lower = smoother but slower to follow

    def __init__(self):
        self._tracked = {}  # stable_id -> {color, shape, centroid}
        self._next_id = 1

    @staticmethod
    def _centroid(pts):
        n = len(pts)
        return (sum(p[0] for p in pts) / n, sum(p[1] for p in pts) / n)

    def update(self, detections):
        unmatched = set(self._tracked)
        result = []

        for d in detections:
            pts = d.get("points") if d.get("points") is not None else d.get("coordinates")
            if not pts:
                continue
            color      = d.get("color", "black")
            shape_type = d.get("shape", "unknown")
            cx, cy     = self._centroid(pts)

            best_id, best_dist = None, self.MATCH_DIST
            for tid in list(unmatched):
                t = self._tracked[tid]
                if t["color"] != color or t["shape"] != shape_type:
                    continue
                tx, ty = t["centroid"]
                dist = ((cx - tx) ** 2 + (cy - ty) ** 2) ** 0.5
                if dist < best_dist:
                    best_dist, best_id = dist, tid

            if best_id is not None:
                unmatched.discard(best_id)
                tx, ty = self._tracked[best_id]["centroid"]
                self._tracked[best_id]["centroid"] = (
                    tx + self.EMA_ALPHA * (cx - tx),
                    ty + self.EMA_ALPHA * (cy - ty),
                )
                stable_id = best_id
            else:
                stable_id = self._next_id
                self._next_id += 1
                self._tracked[stable_id] = {
                    "color": color, "shape": shape_type, "centroid": (cx, cy)
                }

            result.append({
                "id":    str(stable_id),
                "color": COLOR_RGB.get(color, [200, 200, 200]),
                "shape": shape_type,
                "points": pts,
            })

        for tid in unmatched:
            del self._tracked[tid]

        return result


_tracker = ShapeTracker()


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

        # Push tracked/normalized detections into shared state for the sender.
        with _lock:
            _latest_shapes[:] = _tracker.update(detections)

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
