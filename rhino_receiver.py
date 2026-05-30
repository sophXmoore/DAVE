"""
DAVE — Rhino TCP Receiver
Run this inside Rhino via Tools > PythonScript > Run (or the Script Editor).

It listens for JSON from tcp_sender.py and redraws polylines in Rhino
every second. Press Escape in Rhino to stop.

Expected message format (newline-delimited JSON):
{
  "shapes": [
    { "id": "1", "color": [220, 50, 50], "points": [[x,y], ...] },
    ...
  ]
}
"""

import rhinoscriptsyntax as rs
import Rhino
import scriptcontext as sc
import socket
import threading
import json
import time
import System.Drawing
import System

# --- Config ---
HOST = "127.0.0.1"
PORT = 9877
UPDATE_INTERVAL = 1.0  # seconds

# --- Shared state ---
_latest_shapes = []
_lock = threading.Lock()
_running = True

# Maps shape id -> list of Rhino object GUIDs so we can delete & redraw.
_registry = {}


# ---------------------------------------------------------------------------
# Listener thread
# ---------------------------------------------------------------------------

def listener_loop():
    global _latest_shapes, _running

    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((HOST, PORT))
    srv.listen(1)
    srv.settimeout(1.0)
    print("[receiver] Listening on {}:{}".format(HOST, PORT))

    conn = None
    buf = ""

    while _running:
        if conn is None:
            try:
                conn, addr = srv.accept()
                conn.settimeout(1.0)
                buf = ""
                print("[receiver] Sender connected from {}".format(addr))
            except socket.timeout:
                continue

        try:
            chunk = conn.recv(8192).decode("utf-8")
            if not chunk:
                print("[receiver] Sender disconnected.")
                conn.close()
                conn = None
                continue

            buf += chunk
            while "\n" in buf:
                line, buf = buf.split("\n", 1)
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    with _lock:
                        _latest_shapes = data.get("shapes", [])
                except Exception as e:
                    print("[receiver] JSON error: {}".format(e))

        except socket.timeout:
            continue
        except Exception as e:
            print("[receiver] Socket error: {}".format(e))
            if conn:
                conn.close()
            conn = None

    srv.close()
    print("[receiver] Stopped.")


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _rgb_to_color(rgb):
    return System.Drawing.Color.FromArgb(int(rgb[0]), int(rgb[1]), int(rgb[2]))


def _delete_ids(guids):
    for g in guids:
        try:
            sc.doc.Objects.Delete(g, True)
        except Exception:
            pass


def redraw(shapes):
    incoming_ids = {s["id"] for s in shapes}

    # Remove shapes that have disappeared.
    for sid in list(_registry.keys()):
        if sid not in incoming_ids:
            _delete_ids(_registry.pop(sid))

    for shape in shapes:
        sid    = shape["id"]
        pts_2d = shape["points"]
        color  = _rgb_to_color(shape.get("color", [200, 200, 200]))

        # Build 3D points on the XY plane (z = 0).
        pts3d = [Rhino.Geometry.Point3d(p[0], p[1], 0) for p in pts_2d]

        # Close the polyline if it isn't already (filled shapes only —
        # line strokes are intentionally left open).
        if len(pts3d) > 2 and pts3d[0].DistanceTo(pts3d[-1]) > 1e-6:
            pts3d.append(pts3d[0])

        # Delete previous version of this shape.
        if sid in _registry:
            _delete_ids(_registry[sid])

        # Add new polyline curve.
        pl    = Rhino.Geometry.Polyline(pts3d)
        curve = pl.ToNurbsCurve()
        guid  = sc.doc.Objects.AddCurve(curve)

        if guid == System.Guid.Empty:
            _registry[sid] = []
            continue

        # Apply per-object color and name.
        obj  = sc.doc.Objects.Find(guid)
        if obj:
            attr = obj.Attributes.Duplicate()
            attr.ObjectColor  = color
            attr.ColorSource  = Rhino.DocObjects.ObjectColorSource.ColorFromObject
            attr.Name         = "dave_{}".format(sid)
            sc.doc.Objects.ModifyAttributes(obj, attr, True)

        _registry[sid] = [guid]

    sc.doc.Views.Redraw()


# ---------------------------------------------------------------------------
# Main update loop (runs on Rhino's main thread)
# ---------------------------------------------------------------------------

def run():
    global _running

    t = threading.Thread(target=listener_loop, daemon=True)
    t.start()

    print("[receiver] Running. Press Escape to stop.")

    while True:
        if sc.escape_test(False):
            print("[receiver] Escape pressed — shutting down.")
            _running = False
            break

        with _lock:
            shapes = list(_latest_shapes)

        if shapes:
            try:
                redraw(shapes)
            except Exception as e:
                print("[receiver] Redraw error: {}".format(e))

        time.sleep(UPDATE_INTERVAL)


run()
