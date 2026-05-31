"""
DAVE — Rhino TCP Receiver (Timer-based)
Run this inside Rhino via Tools > PythonScript > Run (or the Script Editor).

Uses a WinForms Timer instead of a blocking loop, so Rhino's main thread
(and Grasshopper) can continue running freely between ticks.

Expected message format (newline-delimited JSON):
{
  "shapes": [
    { "id": "1", "color": [220, 50, 50], "points": [[x,y], ...] },
    ...
  ]
}
"""

import Rhino
import scriptcontext as sc
import socket
import threading
import json
import System
import System.Drawing
import System.Windows.Forms as WinForms

# --- Config ---
HOST            = "127.0.0.1"
PORT            = 9877
UPDATE_INTERVAL = 1000  # milliseconds (WinForms Timer uses ms, not seconds)

# --- Shared state ---
_latest_shapes = []
_lock          = threading.Lock()
_running       = True
_registry      = {}   # shape id -> list of Rhino GUIDs
_centroids     = {}   # shape id -> last drawn centroid
MOVE_TOLERANCE = 5.0  # units — skip redraw if centroid moves less than this


# ---------------------------------------------------------------------------
# Centroid helpers
# ---------------------------------------------------------------------------

def _centroid(pts_2d):
    n = len(pts_2d)
    return (sum(p[0] for p in pts_2d) / n, sum(p[1] for p in pts_2d) / n)


def _has_moved(sid, pts_2d):
    cx, cy = _centroid(pts_2d)
    if sid not in _centroids:
        _centroids[sid] = (cx, cy)
        return True
    px, py = _centroids[sid]
    if ((cx - px) ** 2 + (cy - py) ** 2) ** 0.5 > MOVE_TOLERANCE:
        _centroids[sid] = (cx, cy)
        return True
    return False


# ---------------------------------------------------------------------------
# Listener thread — runs on a background thread, never touches Rhino objects
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
    buf  = ""

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
    print("[receiver] Listener stopped.")


# ---------------------------------------------------------------------------
# Geometry helpers — only called from the timer tick (main thread)
# ---------------------------------------------------------------------------

def _rgb_to_color(rgb):
    return System.Drawing.Color.FromArgb(int(rgb[0]), int(rgb[1]), int(rgb[2]))


VIS_LAYER = "DAVE_INPUT_VIS"


def _ensure_vis_layer():
    idx = sc.doc.Layers.FindByFullPath(VIS_LAYER, True)
    if idx < 0:
        layer = Rhino.DocObjects.Layer()
        layer.Name  = VIS_LAYER
        layer.Color = System.Drawing.Color.FromArgb(180, 180, 255)
        idx = sc.doc.Layers.Add(layer)
    return idx


def _delete_ids(guids):
    for g in guids:
        try:
            sc.doc.Objects.Delete(g, True)
        except Exception:
            pass


def _make_attr(color, name, layer_index):
    attr = Rhino.DocObjects.ObjectAttributes()
    attr.ObjectColor  = color
    attr.ColorSource  = Rhino.DocObjects.ObjectColorSource.ColorFromObject
    attr.Name         = name
    attr.LayerIndex   = layer_index
    return attr


def redraw(shapes):
    incoming_ids = {s["id"] for s in shapes}
    vis_layer    = _ensure_vis_layer()

    # Remove shapes that have disappeared.
    for sid in list(_registry.keys()):
        if sid not in incoming_ids:
            _delete_ids(_registry.pop(sid))
            _centroids.pop(sid, None)

    for shape in shapes:
        sid        = shape["id"]
        pts_2d     = shape["points"]
        color      = _rgb_to_color(shape.get("color", [200, 200, 200]))
        shape_type = shape.get("shape", "unknown")

        if not _has_moved(sid, pts_2d):
            continue

        pts3d   = [Rhino.Geometry.Point3d(p[0], p[1], 0) for p in pts_2d]
        is_line = (shape_type == "line")

        if not is_line and len(pts3d) > 2 and pts3d[0].DistanceTo(pts3d[-1]) > 1e-6:
            pts3d.append(pts3d[0])

        if sid in _registry:
            _delete_ids(_registry[sid])

        guids = []
        name  = "dave_{}".format(sid)

        pl    = Rhino.Geometry.Polyline(pts3d)
        curve = pl.ToNurbsCurve()
        guid  = sc.doc.Objects.AddCurve(curve)
        if guid != System.Guid.Empty:
            obj = sc.doc.Objects.Find(guid)
            if obj:
                attr = obj.Attributes.Duplicate()
                attr.ObjectColor = color
                attr.ColorSource = Rhino.DocObjects.ObjectColorSource.ColorFromObject
                attr.Name        = name
                sc.doc.Objects.ModifyAttributes(obj, attr, True)
            guids.append(guid)

        if not is_line and len(pts3d) >= 4:
            try:
                breps = Rhino.Geometry.Brep.CreatePlanarBreps(curve, sc.doc.ModelAbsoluteTolerance)
                if breps:
                    srf_color = System.Drawing.Color.FromArgb(
                        80, int(color.R), int(color.G), int(color.B)
                    )
                    for brep in breps:
                        srf_guid = sc.doc.Objects.AddBrep(
                            brep, _make_attr(srf_color, name + "_srf", vis_layer)
                        )
                        if srf_guid != System.Guid.Empty:
                            guids.append(srf_guid)
            except Exception as e:
                print("[receiver] Surface error for {}: {}".format(sid, e))

        _registry[sid] = guids

    sc.doc.Views.Redraw()


# ---------------------------------------------------------------------------
# Timer tick — fires on the main thread, so Rhino geometry calls are safe.
# Replaces the old blocking while-loop entirely.
# ---------------------------------------------------------------------------

def _on_tick(sender, event_args):
    with _lock:
        shapes = list(_latest_shapes)

    if shapes:
        try:
            redraw(shapes)
        except Exception as e:
            print("[receiver] Redraw error: {}".format(e))


# ---------------------------------------------------------------------------
# Public stop helper — run stop_receiver.py or call this from Script Editor
# ---------------------------------------------------------------------------

def stop_receiver():
    global _running
    _running = False
    _timer.Stop()
    _timer.Dispose()
    print("[receiver] Timer stopped. Listener thread will exit shortly.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

# Start the background listener thread.
_listener_thread = threading.Thread(target=listener_loop, daemon=True)
_listener_thread.start()

# Create and start the WinForms timer (fires on main thread — GH-safe).
_timer          = WinForms.Timer()
_timer.Interval = UPDATE_INTERVAL
_timer.Tick    += _on_tick
_timer.Start()

print("[receiver] Running (timer-based, interval={}ms).".format(UPDATE_INTERVAL))