import sys
import json
import time
import cv2
import numpy as np

# Try 0 first. If it doesn't open the right camera, try 1 or 2.
CAMERA_INDEX = 1

# Solid shapes must be at least this big (filters noise / tiny specks).
MIN_AREA = 1500

# Drawn lines are thin, so they have small area -- allow much smaller blobs
# through when we're looking for marker strokes.
MIN_LINE_AREA = 350

# A contour is treated as a drawn line (not a filled shape) when its estimated
# stroke thickness is below this many pixels. thickness ~= 2 * area / perimeter:
# for a long thin stroke this works out to roughly the marker's width, while a
# filled shape comes out much larger (~half its diameter).
LINE_MAX_THICKNESS = 14

# HSV color ranges. Hue in OpenCV is 0-179.
# Red wraps around 0/180, so it needs two ranges.
COLOR_RANGES = {
    "red": [
        (np.array([0,   120,  80]), np.array([10,  255, 255])),
        (np.array([170, 120,  80]), np.array([179, 255, 255])),
    ],
    "blue": [
        (np.array([95,  120,  50]), np.array([130, 255, 255])),
    ],
    "black": [
        (np.array([0,     0,   0]), np.array([179,  90,  70])),
    ],
}

# BGR draw colors per label.
DRAW_COLORS = {
    "red":   (0,   0, 255),
    "blue":  (255, 0,   0),
    "black": (0, 255, 255),  # yellow outline so it's visible on black
}


def mask_for_color(hsv, ranges):
    mask = None
    for lo, hi in ranges:
        m = cv2.inRange(hsv, lo, hi)
        mask = m if mask is None else cv2.bitwise_or(mask, m)
    # Gentle open (3x3) so thin marker strokes survive; close (5x5) to bridge
    # small gaps in a stroke that camera noise / glare would otherwise break up.
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    return mask


def stroke_thickness(area, perimeter):
    # Approximate stroke width. For a long thin stroke (area ~= length*width,
    # perimeter ~= 2*length) this reduces to roughly the width itself.
    if perimeter == 0:
        return 0.0
    return 2.0 * area / perimeter


def classify_polygon(approx):
    v = len(approx)
    if v == 3:
        return "triangle"
    if v == 4:
        return "quad"
    if v == 5:
        return "pentagon"
    if v >= 6:
        return "circle/poly"
    return f"{v}-gon"


def contour_points(contour):
    # Flatten an OpenCV contour into a plain list of [x, y] int pairs that is
    # safe to json.dump (numpy ints are not JSON-serializable on their own).
    return [[int(p[0][0]), int(p[0][1])] for p in contour]


def skeletonize(binary):
    # Zhang-Suen thinning: reduce a filled stroke to a clean 1px-wide centerline
    # with correct topology (no opencv-contrib needed). Vectorized with numpy so
    # it stays fast on the small per-stroke ROIs we hand it.
    img = (binary > 0).astype(np.uint8)
    changed = True
    while changed:
        changed = False
        for step in (0, 1):
            P = np.pad(img, 1)
            p2, p3, p4 = P[:-2, 1:-1], P[:-2, 2:], P[1:-1, 2:]
            p5, p6, p7 = P[2:, 2:], P[2:, 1:-1], P[2:, :-2]
            p8, p9 = P[1:-1, :-2], P[:-2, :-2]
            nbrs = [p2, p3, p4, p5, p6, p7, p8, p9]
            B = sum(nbrs)
            # A = number of 0->1 transitions going around p2,p3,...,p9,p2
            seq = nbrs + [p2]
            A = sum(((seq[k] == 0) & (seq[k + 1] == 1)).astype(np.uint8) for k in range(8))
            if step == 0:
                c1, c2 = (p2 * p4 * p6 == 0), (p4 * p6 * p8 == 0)
            else:
                c1, c2 = (p2 * p4 * p8 == 0), (p2 * p6 * p8 == 0)
            cond = (img == 1) & (B >= 2) & (B <= 6) & (A == 1) & c1 & c2
            if cond.any():
                img[cond] = 0
                changed = True
    return (img * 255).astype(np.uint8)


def order_points(points, start=None):
    # Greedy nearest-neighbor walk so the centerline reads as a path. Prefer a
    # caller-supplied endpoint; otherwise fall back to the point farthest from
    # the centroid (a rough guess for a stroke end).
    pts = np.asarray(points, dtype=np.float32)
    n = len(pts)
    if n <= 2:
        return [[int(p[0]), int(p[1])] for p in pts]
    if start is None:
        start = int(np.argmax(((pts - pts.mean(axis=0)) ** 2).sum(axis=1)))
    used = np.zeros(n, dtype=bool)
    order = [start]
    used[start] = True
    for _ in range(n - 1):
        d = ((pts - pts[order[-1]]) ** 2).sum(axis=1)
        d[used] = np.inf
        nxt = int(np.argmin(d))
        order.append(nxt)
        used[nxt] = True
    return [[int(pts[i][0]), int(pts[i][1])] for i in order]


def centerline_coords(contour, color_mask):
    # Centerline of a single stroke as an ordered list of [x, y] points.
    x, y, w, h = cv2.boundingRect(contour)
    # Isolate just this stroke's ink: fill its contour and AND with the color
    # mask. The AND keeps hollow strokes hollow (e.g. a drawn circle stays a
    # ring instead of collapsing to a filled disc).
    filled = np.zeros((h, w), np.uint8)
    cv2.drawContours(filled, [contour - np.array([[[x, y]]])], -1, 255, cv2.FILLED)
    region = cv2.bitwise_and(filled, color_mask[y:y + h, x:x + w])

    # Pad with a zero border so strokes that touch the bounding box edge keep
    # clean tips -- otherwise the endpoint test below miscounts border pixels.
    pad = 2
    region = cv2.copyMakeBorder(region, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)
    x, y = x - pad, y - pad

    skel = skeletonize(region)
    ys, xs = np.where(skel > 0)
    if len(xs) == 0:
        return []

    # A true endpoint is a skeleton pixel with exactly one neighbor. Open
    # strokes have them (closed loops don't -> start stays None).
    sk = (skel > 0).astype(np.uint8)
    neighbors = cv2.filter2D(
        sk, -1, np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], np.uint8),
        borderType=cv2.BORDER_CONSTANT,
    )
    ends = np.argwhere((sk == 1) & (neighbors == 1))  # rows of [y, x]
    start_xy = (int(ends[0][1] + x), int(ends[0][0] + y)) if len(ends) else None

    pts = np.stack([xs + x, ys + y], axis=1)
    if len(pts) > 600:  # cap ordering cost on very long strokes
        pts = pts[:: len(pts) // 600 + 1]

    start = None
    if start_xy is not None:
        start = int(((pts - np.array(start_xy)) ** 2).sum(axis=1).argmin())

    ordered = np.array(order_points(pts.tolist(), start), dtype=np.int32).reshape(-1, 1, 2)
    simplified = cv2.approxPolyDP(ordered, 2.0, False)
    return [[int(p[0][0]), int(p[0][1])] for p in simplified]


def detect_shapes(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    output = frame.copy()
    combined_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    detections = []
    next_id = 1

    for color_name, ranges in COLOR_RANGES.items():
        mask = mask_for_color(hsv, ranges)
        combined_mask = cv2.bitwise_or(combined_mask, mask)

        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        draw_bgr = DRAW_COLORS[color_name]

        for c in contours:
            area = cv2.contourArea(c)
            if area < MIN_LINE_AREA:
                continue

            perimeter = cv2.arcLength(c, True)
            thickness = stroke_thickness(area, perimeter)

            M = cv2.moments(c)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
            else:
                cx, cy = c[0][0]

            if thickness < LINE_MAX_THICKNESS:
                # Thin -> treat as a hand-drawn marker line/stroke. Reduce it to
                # a centerline path rather than tracing both sides of the stroke.
                coords = centerline_coords(c, mask)
                if len(coords) < 2:
                    continue
                pts = np.array(coords, dtype=np.int32).reshape(-1, 1, 2)
                cv2.polylines(output, [pts], False, draw_bgr, 2)
                cv2.putText(
                    output, f"{color_name} line", (cx - 40, cy - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, draw_bgr, 2
                )
                detections.append({
                    "id": next_id,
                    "color": color_name,
                    "points": coords,
                })
                next_id += 1
                continue

            # Fat -> filled shape. Apply the usual size floor and classify it.
            if area < MIN_AREA:
                continue

            epsilon = 0.03 * perimeter
            approx = cv2.approxPolyDP(c, epsilon, True)

            cv2.drawContours(output, [approx], -1, draw_bgr, 3)
            label = f"{color_name} {classify_polygon(approx)}"
            cv2.circle(output, (cx, cy), 5, draw_bgr, -1)
            cv2.putText(
                output, label, (cx - 50, cy - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, draw_bgr, 2
            )
            detections.append({
                "id": next_id,
                "color": color_name,
                "coordinates": contour_points(approx),
            })
            next_id += 1

    return output, combined_mask, detections


def save_detections(detections, path):
    with open(path, "w") as f:
        json.dump(detections, f, indent=2)
    print(f"Wrote {len(detections)} detections to {path}")


def run_on_image(path):
    frame = cv2.imread(path)
    if frame is None:
        raise RuntimeError(f"Could not read image: {path}")
    frame = cv2.resize(frame, (960, 540))
    output, mask, detections = detect_shapes(frame)
    save_detections(detections, "detections.json")
    cv2.imshow("Input", frame)
    cv2.imshow("Color Mask", mask)
    cv2.imshow("Detected Shapes", output)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def run_on_camera():
    # CAP_DSHOW (DirectShow) opens far faster than the MSMF default on Windows.
    #cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        raise RuntimeError (
            "Could not open webcam. Try changing CAMERA_INDEX to 1 or 2."
        )

    last_print = time.time()
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (960, 540))
        output, mask, detections = detect_shapes(frame)

        cv2.imshow("Webcam", frame)
        cv2.imshow("Color Mask", mask)
        cv2.imshow("Detected Shapes", output)

        # Print the current detections JSON every 5 seconds.
        now = time.time()
        if now - last_print >= 5.0:
            print(json.dumps(detections, indent=2), flush=True)
            last_print = now

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("s"):
            # Snapshot the current frame's detections to JSON.
            save_detections(detections, "detections.json")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # Usage:
    #   python detect_shape_edges.py                 -> live webcam ('s' saves
    #                                                   detections.json, 'q' quits)
    #   python detect_shape_edges.py path\to\img.png -> still image (writes
    #                                                   detections.json on load)
    if len(sys.argv) > 1:
        run_on_image(sys.argv[1])
    else:
        run_on_camera()
