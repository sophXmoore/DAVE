import sys
import cv2
import numpy as np

# Try 0 first. If it doesn't open the right camera, try 1 or 2.
CAMERA_INDEX = 0

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


def detect_shapes(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    output = frame.copy()
    combined_mask = np.zeros(frame.shape[:2], dtype=np.uint8)

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
                # Thin -> treat as a hand-drawn marker line/stroke. Trace the
                # stroke itself rather than approximating it to a polygon.
                cv2.drawContours(output, [c], -1, draw_bgr, 2)
                cv2.putText(
                    output, f"{color_name} line", (cx - 40, cy - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, draw_bgr, 2
                )
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

    return output, combined_mask


def run_on_image(path):
    frame = cv2.imread(path)
    if frame is None:
        raise RuntimeError(f"Could not read image: {path}")
    frame = cv2.resize(frame, (960, 540))
    output, mask = detect_shapes(frame)
    cv2.imshow("Input", frame)
    cv2.imshow("Color Mask", mask)
    cv2.imshow("Detected Shapes", output)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def run_on_camera():
    # CAP_DSHOW (DirectShow) opens far faster than the MSMF default on Windows.
    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise RuntimeError(
            "Could not open webcam. Try changing CAMERA_INDEX to 1 or 2."
        )

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (960, 540))
        output, mask = detect_shapes(frame)

        cv2.imshow("Webcam", frame)
        cv2.imshow("Color Mask", mask)
        cv2.imshow("Detected Shapes", output)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # Usage:
    #   python detect_shape_edges.py                 -> live webcam
    #   python detect_shape_edges.py path\to\img.png -> still image
    if len(sys.argv) > 1:
        run_on_image(sys.argv[1])
    else:
        run_on_camera()
