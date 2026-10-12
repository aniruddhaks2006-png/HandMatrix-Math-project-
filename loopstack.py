import math
import os
import threading
import time
import urllib.request
from collections import Counter, deque

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

SIZES = [3, 5, 7, 11, 15, 21, 31]
BORDER = cv2.BORDER_REFLECT
CANCEL_HOLD_SECONDS = 0.7
SMOOTH = 0.5

MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
             "hand_landmarker/float16/1/hand_landmarker.task")
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hand_landmarker.task")

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12), (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20), (0, 17),
]


def ensure_model():
    if os.path.exists(MODEL_PATH):
        return
    print("Downloading hand_landmarker.task (one time, ~8 MB)...")
    try:
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    except Exception as e:
        raise RuntimeError(
            f"Could not download the model ({e}).\nDownload it manually from:\n  {MODEL_URL}\n"
            f"and save it next to this script as hand_landmarker.task"
        )


# ----------------------------------------------------------------- kernels
def binomial_1d(size):
    return np.array([math.comb(size - 1, i) for i in range(size)], dtype=np.float32)


def derivative_1d(size):
    return np.arange(size, dtype=np.float32) - size // 2


def _normalize_abs(k):
    s = float(np.sum(np.abs(k)))
    return (k / s).astype(np.float32) if s else k.astype(np.float32)


def sobel_x_kernel(size):
    return _normalize_abs(np.outer(binomial_1d(size), derivative_1d(size)))


def sobel_y_kernel(size):
    return _normalize_abs(np.outer(derivative_1d(size), binomial_1d(size)))


def laplacian_kernel(size):
    if size == 3:
        return np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32)
    c = size // 2
    k = np.zeros((size, size), dtype=np.float32)
    for dx in range(-2, 3):
        for dy in range(-2, 3):
            d = abs(dx) + abs(dy)
            if 0 < d <= 2:
                k[c + dx, c + dy] = 1.0 if d == 1 else 2.0
    k[c, c] = -k.sum()
    return _normalize_abs(k)


def sharpen_kernel(size):
    if size == 3:
        return np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
    delta = np.zeros((size, size), dtype=np.float32)
    delta[size // 2, size // 2] = 1.0
    return delta - laplacian_kernel(size)


# ----------------------------------------------------------------- effects
def conv(img, kernel):
    return cv2.filter2D(img, -1, kernel, borderType=BORDER)


def smooth_normalize(x, key, peaks):
    p = float(x.max())
    peaks[key] = 0.85 * peaks.get(key, p) + 0.15 * p
    return np.clip(x / max(peaks[key], 1e-4), 0.0, 1.0)


def apply_effect(bgr, effect, size, peaks):
    original = bgr.astype(np.float32) / 255.0
    if effect == "Gaussian Blur":
        filtered = cv2.GaussianBlur(original, (size, size), sigmaX=max(1.0, size / 3.0), borderType=BORDER)
    elif effect == "Sobel X":
        filtered = smooth_normalize(np.abs(conv(original, sobel_x_kernel(size))), effect, peaks)
    elif effect == "Sobel Y":
        filtered = smooth_normalize(np.abs(conv(original, sobel_y_kernel(size))), effect, peaks)
    elif effect == "Sobel Magnitude":
        gx = conv(original, sobel_x_kernel(size))
        gy = conv(original, sobel_y_kernel(size))
        filtered = smooth_normalize(np.sqrt(gx * gx + gy * gy), effect, peaks)
    elif effect == "Laplacian":
        filtered = smooth_normalize(np.abs(conv(original, laplacian_kernel(size))), effect, peaks)
    elif effect == "Sharpen":
        filtered = np.clip(conv(original, sharpen_kernel(size)), 0.0, 1.0)
    else:
        return bgr
    return np.clip(filtered * 255.0, 0, 255).astype(np.uint8)


# ----------------------------------------------------------------- gestures
def dist(lm, a, b):
    return math.hypot(lm[a].x - lm[b].x, lm[a].y - lm[b].y)


def pdist(lm, a, b, w, h):
    return math.hypot((lm[a].x - lm[b].x) * w, (lm[a].y - lm[b].y) * h)


def classify_gesture(lm):
    tips, pips = [8, 12, 16, 20], [6, 10, 14, 18]
    fingers = sum(1 for t, p in zip(tips, pips) if dist(lm, 0, t) > dist(lm, 0, p) * 1.02)
    thumb_out = dist(lm, 4, 17) > dist(lm, 3, 17) * 1.05
    if fingers == 0:
        return "Laplacian"
    if fingers == 1:
        return "Sobel X"
    if fingers == 2:
        return "Sobel Y"
    if fingers == 3:
        return "Sobel Magnitude"
    return "Gaussian Blur" if thumb_out else "Sharpen"


def is_cancel_pose(hands, w, h):
    a, b = hands
    palm = 0.5 * (pdist(a, 0, 9, w, h) + pdist(b, 0, 9, w, h))
    wrist_gap = math.hypot((a[0].x - b[0].x) * w, (a[0].y - b[0].y) * h)
    return wrist_gap < 0.8 * max(palm, 1.0)


# ----------------------------------------------------------------- hand slots
def assign_slots(hands, slots, w, h):
    if len(hands) == 2:
        return [0, 1]
    wx, wy = hands[0][0].x * w, hands[0][0].y * h
    if slots[0] is not None and slots[1] is not None:
        d = [math.hypot(wx - s["wrist"][0], wy - s["wrist"][1]) for s in slots]
        return [int(np.argmin(d))]
    return [0 if wx < w / 2 else 1]


def update_slot(slots, i, lm, w, h):
    pts = np.array([[lm[4].x * w, lm[4].y * h], [lm[8].x * w, lm[8].y * h]], dtype=np.float32)
    wrist = np.array([lm[0].x * w, lm[0].y * h], dtype=np.float32)
    if slots[i] is not None:
        pts = (1 - SMOOTH) * slots[i]["pts"] + SMOOTH * pts
    slots[i] = {"pts": pts, "wrist": wrist}


def order_quad(slots):
    pts = np.vstack([slots[0]["pts"], slots[1]["pts"]])
    c = pts.mean(axis=0)
    ang = np.arctan2(pts[:, 1] - c[1], pts[:, 0] - c[0])
    return pts[np.argsort(ang)]


# ----------------------------------------------------------------- drawing
def draw_hand(img, lm, w, h):
    pts = [(int(p.x * w), int(p.y * h)) for p in lm]
    for a, b in HAND_CONNECTIONS:
        cv2.line(img, pts[a], pts[b], (255, 255, 255), 1, cv2.LINE_AA)


def calculate_kernel_size(poly):
    x0, y0 = poly.min(axis=0)
    x1, y1 = poly.max(axis=0)
    diag = math.hypot(x1 - x0, y1 - y0)
    t = float(np.clip((diag - 100) / (600 - 100), 0, 1))
    return SIZES[min(int(t * len(SIZES)), len(SIZES) - 1)]


def render_region(frame, out, poly, effect, size, peaks):
    h, w = frame.shape[:2]
    if cv2.contourArea(poly.astype(np.float32)) < 200:
        return

    pad = size // 2 + 2
    x0, y0 = poly.min(axis=0)
    x1, y1 = poly.max(axis=0)
    px0, py0 = max(int(x0) - pad, 0), max(int(y0) - pad, 0)
    px1, py1 = min(int(x1) + pad + 1, w), min(int(y1) + pad + 1, h)
    if px1 - px0 < 4 or py1 - py0 < 4:
        return

    crop = frame[py0:py1, px0:px1]
    patch = apply_effect(crop, effect, size, peaks)
    local = np.round(poly - np.array([px0, py0])).astype(np.int32)
    mask = np.zeros(crop.shape[:2], dtype=np.float32)
    cv2.fillPoly(mask, [local], 1.0, lineType=cv2.LINE_AA)
    mask = cv2.GaussianBlur(mask, (0, 0), 1.5)[:, :, None]
    blended = crop.astype(np.float32) * (1 - mask) + patch.astype(np.float32) * mask
    out[py0:py1, px0:px1] = blended.astype(np.uint8)


def draw_shape(img, poly, color, label):
    p = np.round(poly).astype(np.int32)
    cv2.polylines(img, [p], True, color, 2, cv2.LINE_AA)
    for q in p:
        cv2.circle(img, tuple(int(v) for v in q), 6, color, -1, cv2.LINE_AA)
    top = p[np.argmin(p[:, 1])]
    cv2.putText(img, label, (int(top[0]) + 8, max(int(top[1]) - 8, 16)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)


# ----------------------------------------------------------------- camera
class Camera:
    def __init__(self, width=1280, height=720):
        self.cap = None
        for idx in (0, 1, 2):
            cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW) if os.name == "nt" else cv2.VideoCapture(idx)
            if not cap.isOpened():
                cap.release()
                cap = cv2.VideoCapture(idx)
            if cap.isOpened():
                self.cap = cap
                break
            cap.release()
        if self.cap is None:
            raise RuntimeError("Could not open any webcam (tried indices 0-2).")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.frame = None
        self.lock = threading.Lock()
        self.running = True
        threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self):
        while self.running:
            ok, f = self.cap.read()
            if ok:
                with self.lock:
                    self.frame = f

    def read(self):
        with self.lock:
            f, self.frame = self.frame, None
        return f

    def release(self):
        self.running = False
        time.sleep(0.05)
        self.cap.release()


# ----------------------------------------------------------------- main
def main():
    ensure_model()
    options = vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.6,
        min_hand_presence_confidence=0.6,
        min_tracking_confidence=0.6,
    )

    cam = Camera(width=1280, height=720)
    window_name = "Two-Hand Free-Form Convolution"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    peaks = {}
    history = deque(maxlen=6)
    effect = "Gaussian Blur"
    slots = [None, None]
    layers = []  # Stack of committed layers: list of dicts {poly, effect, size}
    size = 7
    cancel_since = None
    cancel_latch = False
    fps, prev = 0.0, time.time()
    start, last_ts = time.time(), -1

    with vision.HandLandmarker.create_from_options(options) as landmarker:
        while True:
            frame = cam.read()
            if frame is None:
                if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                    break
                continue

            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]
            rgb = np.ascontiguousarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            ts = max(int((time.time() - start) * 1000), last_ts + 1)
            last_ts = ts
            res = landmarker.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), ts)

            hands = list(res.hand_landmarks) if res.hand_landmarks else []
            hands.sort(key=lambda lm: lm[0].x)
            progress = 0.0
            cancel_pose = len(hands) == 2 and is_cancel_pose(hands, w, h)

            if cancel_pose:
                if not cancel_latch:
                    cancel_since = cancel_since or time.time()
                    progress = (time.time() - cancel_since) / CANCEL_HOLD_SECONDS
                    if progress >= 1.0:
                        slots = [None, None]
                        layers.clear()
                        cancel_latch = True
                        cancel_since = None
                        history.clear()
                        progress = 0.0
            else:
                cancel_since = None
                cancel_latch = cancel_latch and len(hands) == 2
                if hands:
                    history.append(classify_gesture(hands[0]))
                    if len(history) >= 3:
                        effect = Counter(history).most_common(1)[0][0]
                    for lm, i in zip(hands, assign_slots(hands, slots, w, h)):
                        update_slot(slots, i, lm, w, h)

            # Start rendering from raw camera frame
            canvas = frame.copy()

            # 1. Apply existing locked layers in sequence
            for idx, layer in enumerate(layers):
                render_region(canvas, canvas, layer["poly"], layer["effect"], layer["size"], peaks)

            # 2. Draw live interactive layer on top
            status = f"stack: {len(layers)} layer(s)"
            live_poly = None
            if slots[0] is not None and slots[1] is not None:
                live_poly = order_quad(slots)
                size = calculate_kernel_size(live_poly)
                render_region(canvas, canvas, live_poly, effect, size, peaks)
                status = f"stack: {len(layers)} | active quad ready"

            out = canvas.copy()

            # 3. Draw outlines
            for idx, layer in enumerate(layers):
                draw_shape(out, layer["poly"], (255, 140, 40), f"L{idx + 1}: {layer['effect']}")

            if live_poly is not None:
                live = len(hands) == 2
                color = (60, 200, 255) if live else (80, 200, 80)
                draw_shape(out, live_poly, color, f"Live: {effect} {size}x{size}")

            for lm in hands:
                draw_hand(out, lm, w, h)

            now = time.time()
            dt = now - prev
            prev = now
            if dt > 0:
                fps = 0.9 * fps + 0.1 / dt if fps else 1.0 / dt

            # HUD
            cv2.rectangle(out, (8, 8), (530, 102), (20, 20, 20), -1)
            cv2.putText(out, f"Filter: {effect}  Kernel: {size}x{size}  [Layers: {len(layers)}]",
                        (18, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (80, 240, 255), 2)
            cv2.putText(out, f"Status: {status}  FPS: {fps:.0f}",
                        (18, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (255, 255, 255), 1)
            cv2.putText(out, "L: lock/push layer | U: undo layer | C / Palms: clear all",
                        (18, 76), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (180, 255, 180), 1)
            cv2.putText(out, "Q / ESC: quit",
                        (18, 96), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (180, 255, 180), 1)

            if progress > 0:
                cv2.rectangle(out, (8, 106), (8 + int(522 * min(progress, 1.0)), 114), (60, 60, 255), -1)

            cv2.imshow(window_name, out)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
            if key == ord("l"):
                # Commit current quad to stack and reset hand tracking slots
                if slots[0] is not None and slots[1] is not None:
                    p = order_quad(slots)
                    k_size = calculate_kernel_size(p)
                    layers.append({"poly": p, "effect": effect, "size": k_size})
                    slots = [None, None]
            if key in (ord("u"), ord("z")):
                # Undo topmost stacked layer
                if layers:
                    layers.pop()
            if key == ord("c"):
                # Clear all stacked layers & reset
                slots = [None, None]
                layers.clear()
                history.clear()

    cam.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()