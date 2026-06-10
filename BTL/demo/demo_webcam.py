"""
demo_webcam.py — v2 (Nhanh hơn, dùng MediaPipe Face Detection thay MTCNN)
--------------------------------------------------------------------------
Demo nhận diện trạng thái tương tác từ webcam realtime.

Cách dùng (local Windows, có webcam):
    py -3.12 BTL/demo/demo_webcam.py --model landmarks
    py -3.12 BTL/demo/demo_webcam.py --model efficientnet

Bấm 'q' để thoát.

Cải tiến v2:
- MediaPipe Face Detection thay MTCNN (nhanh ~10×, ổn định)
- Tasks API mới cho landmarks (không bị bug solutions)
- Smoothing với moving average → predict ổn định, ít nhảy
- Resize frame trước khi detect → giảm tải CPU
"""

import argparse
import os
import sys
import time
import threading
import urllib.request
from collections import deque
from queue import Queue, Empty

import numpy as np
import cv2
import tensorflow as tf

# ── Tắt log TF ──────────────────────────────────────────────────────────────
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.get_logger().setLevel('ERROR')

# ── Constants ───────────────────────────────────────────────────────────────
LABEL_NAMES = [
    'Focused', 'Confused', 'Frustrated',
    'Bored', 'Drowsy', 'Looking Away'
]
ENGAGED = {0, 1, 2}

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

DEFAULT_WEIGHTS = {
    'efficientnet': 'BTL/weights/efficientnet/efficientnet_phase2.weights.h5',
    'landmarks':    'BTL/weights/efficientnet_landmark/efficientnet_lm_phase2.weights.h5',
}

# Tốc độ
PROCESS_WIDTH = 480    # Resize frame xuống 480 trước khi xử lý
SMOOTH_WINDOW = 10     # Trung bình hóa 10 predictions gần nhất (~3 giây)


def build_model(model_type):
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'models'))
    if model_type == 'landmarks':
        from efficientnet_landmarks import build_model as build_lm
        return build_lm(num_classes=6)
    else:
        from efficientnet import build_model as build_ef
        return build_ef(num_classes=6)


def preprocess(face_rgb):
    """Resize + normalize ImageNet."""
    img224 = cv2.resize(face_rgb, (224, 224))
    img = img224.astype(np.float32) / 255.0
    img = (img - IMAGENET_MEAN) / IMAGENET_STD
    return img, img224


def init_face_detector():
    """MediaPipe Face Detection — siêu nhanh, ~5-10ms/frame trên CPU."""
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision

    MODEL_URL  = "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite"
    MODEL_PATH = "face_detector.tflite"
    if not os.path.exists(MODEL_PATH):
        print("Đang tải MediaPipe Face Detector...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)

    base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.FaceDetectorOptions(
        base_options=base_options,
        min_detection_confidence=0.5,
    )
    return vision.FaceDetector.create_from_options(options), mp


def init_face_landmarker():
    """MediaPipe Face Landmarker — cho 478 điểm landmark."""
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision

    MODEL_URL  = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
    MODEL_PATH = "face_landmarker.task"
    if not os.path.exists(MODEL_PATH):
        print("Đang tải MediaPipe Landmarker...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)

    base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.FaceLandmarkerOptions(
        base_options=base_options, num_faces=1
    )
    return vision.FaceLandmarker.create_from_options(options)


def detect_face_mp(detector, mp, img_rgb, margin=0.10):
    """Phát hiện khuôn mặt bằng MediaPipe + SQUARE crop để match training."""
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
    results = detector.detect(mp_image)
    if not results.detections:
        return None, None

    best = max(results.detections, key=lambda d: d.categories[0].score)
    bbox = best.bounding_box
    x, y, w, h = bbox.origin_x, bbox.origin_y, bbox.width, bbox.height

    # Square crop: lấy cạnh dài làm size
    side = max(w, h)
    cx, cy = x + w // 2, y + h // 2

    # Thêm margin 10% (match training)
    side = int(side * (1 + 2 * margin))

    H, W = img_rgb.shape[:2]
    x1 = max(0, cx - side // 2); y1 = max(0, cy - side // 2)
    x2 = min(W, cx + side // 2); y2 = min(H, cy + side // 2)
    if x2 <= x1 or y2 <= y1:
        return None, None
    return img_rgb[y1:y2, x1:x2], (x1, y1, x2, y2)


def extract_landmarks(detector, mp, img_rgb_224):
    """Trích landmark vector 956-dim."""
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb_224)
    results = detector.detect(mp_image)
    if not results.face_landmarks:
        return np.zeros(956, dtype=np.float32)
    lm = results.face_landmarks[0]
    vec = np.array([[p.x, p.y] for p in lm[:478]], dtype=np.float32).flatten()
    if vec.shape[0] < 956:
        vec = np.pad(vec, (0, 956 - vec.shape[0]))
    elif vec.shape[0] > 956:
        vec = vec[:956]
    return vec.astype(np.float32)


def draw_label(frame_bgr, bbox, label_str, group_str, color, confidence):
    """Vẽ bounding box + nhãn chính."""
    x1, y1, x2, y2 = bbox
    cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), color, 2)
    bar_h = 50
    cv2.rectangle(frame_bgr, (x1, max(0, y1 - bar_h)), (x2, y1), color, -1)
    cv2.putText(
        frame_bgr, f'{label_str} {confidence*100:.0f}%',
        (x1 + 5, y1 - 28),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2
    )
    cv2.putText(
        frame_bgr, group_str,
        (x1 + 5, y1 - 8),
        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1
    )


def draw_top3_panel(frame_bgr, avg_probs):
    """Vẽ panel Top-3 prediction ở góc phải dưới."""
    H, W = frame_bgr.shape[:2]
    panel_w = 260
    panel_h = 130
    px = W - panel_w - 10
    py = H - panel_h - 10

    # Background đen mờ
    overlay = frame_bgr.copy()
    cv2.rectangle(overlay, (px, py), (px + panel_w, py + panel_h), (30, 30, 30), -1)
    cv2.addWeighted(overlay, 0.7, frame_bgr, 0.3, 0, frame_bgr)

    cv2.putText(frame_bgr, 'Top 3 predictions:', (px + 8, py + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    top3_idx = np.argsort(avg_probs)[-3:][::-1]
    for i, idx in enumerate(top3_idx):
        prob = avg_probs[idx]
        name = LABEL_NAMES[idx]
        color = (0, 220, 0) if idx in ENGAGED else (50, 100, 240)
        y = py + 45 + i * 28

        # Tên + %
        cv2.putText(frame_bgr, f'{name}', (px + 8, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        cv2.putText(frame_bgr, f'{prob*100:.0f}%', (px + 130, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        # Bar
        bar_x = px + 170
        bar_w_max = 80
        bar_w = int(bar_w_max * prob)
        cv2.rectangle(frame_bgr,
                      (bar_x, y - 10), (bar_x + bar_w_max, y - 2),
                      (60, 60, 60), -1)
        cv2.rectangle(frame_bgr,
                      (bar_x, y - 10), (bar_x + bar_w, y - 2),
                      color, -1)


class InferenceWorker(threading.Thread):
    """
    Thread chạy face detection + model prediction.
    UI thread chỉ đẩy frame vào queue → không bị block.
    """
    def __init__(self, model, model_type, face_detector, landmarker, mp_module):
        super().__init__(daemon=True)
        self.model = model
        self.model_type = model_type
        self.face_detector = face_detector
        self.landmarker = landmarker
        self.mp = mp_module

        self.frame_queue = Queue(maxsize=1)
        self.stop_flag = threading.Event()
        # Shared state — UI đọc, worker ghi
        self.lock = threading.Lock()
        self.last_bbox_small = None
        self.last_probs = None
        self.version = 0  # Tăng mỗi khi có prediction mới

    def submit(self, rgb_small):
        """Đẩy frame mới. Drop frame cũ nếu queue đầy."""
        if not self.frame_queue.full():
            self.frame_queue.put(rgb_small, block=False)

    def get_state(self):
        with self.lock:
            return self.last_bbox_small, self.last_probs, self.version

    def stop(self):
        self.stop_flag.set()

    def run(self):
        while not self.stop_flag.is_set():
            try:
                rgb_small = self.frame_queue.get(timeout=0.1)
            except Empty:
                continue

            face_crop, bbox = detect_face_mp(self.face_detector, self.mp, rgb_small)
            if bbox is None:
                with self.lock:
                    self.last_bbox_small = None
                continue

            img_norm, img_224 = preprocess(face_crop)
            img_batch = np.expand_dims(img_norm, 0)

            if self.model_type == 'landmarks':
                lm = extract_landmarks(self.landmarker, self.mp, img_224)
                probs = self.model.predict(
                    [img_batch, np.expand_dims(lm, 0)], verbose=0
                )[0]
            else:
                probs = self.model.predict(img_batch, verbose=0)[0]

            with self.lock:
                self.last_bbox_small = bbox
                self.last_probs = probs
                self.version += 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=['efficientnet', 'landmarks'],
                        default='efficientnet')
    parser.add_argument('--weights', default=None)
    parser.add_argument('--camera', type=int, default=0)
    args = parser.parse_args()

    print(f'Loading model "{args.model}"...')
    model = build_model(args.model)
    weights_path = args.weights or DEFAULT_WEIGHTS[args.model]
    if not os.path.exists(weights_path):
        print(f'Lỗi: weights {weights_path} không tồn tại')
        return
    model.load_weights(weights_path)
    print(f'Đã load: {weights_path}')

    face_detector, mp = init_face_detector()
    print('MediaPipe Face Detector sẵn sàng.')

    landmarker = None
    if args.model == 'landmarks':
        landmarker = init_face_landmarker()
        print('MediaPipe Landmarker sẵn sàng.')

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f'Không mở được camera {args.camera}')
        return
    print('Camera sẵn sàng. Bấm "q" để thoát.')

    # Khởi động inference thread
    worker = InferenceWorker(model, args.model, face_detector, landmarker, mp)
    worker.start()

    # Trạng thái UI
    frame_count = 0
    smooth_probs = deque(maxlen=SMOOTH_WINDOW)
    last_seen_version = -1
    fps_t0 = time.time()
    fps = 0.0
    submit_every = 2  # Đẩy frame vào worker mỗi 2 frame (tránh queue overload)

    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break
        frame_count += 1

        # Resize cho worker
        h, w = frame_bgr.shape[:2]
        if w > PROCESS_WIDTH:
            scale = PROCESS_WIDTH / w
            small_bgr = cv2.resize(frame_bgr, (PROCESS_WIDTH, int(h * scale)))
        else:
            scale = 1.0
            small_bgr = frame_bgr

        # Submit frame cho worker async
        if frame_count % submit_every == 0:
            small_rgb = cv2.cvtColor(small_bgr, cv2.COLOR_BGR2RGB)
            worker.submit(small_rgb)

        # Lấy kết quả mới nhất từ worker — CHỈ append khi version mới
        bbox_small, probs, version = worker.get_state()
        if probs is not None and version != last_seen_version:
            smooth_probs.append(probs)
            last_seen_version = version

        # Vẽ
        if bbox_small is not None and smooth_probs:
            x1, y1, x2, y2 = bbox_small
            bbox_full = (int(x1/scale), int(y1/scale),
                         int(x2/scale), int(y2/scale))

            avg_probs = np.mean(smooth_probs, axis=0)
            pred_idx = int(np.argmax(avg_probs))
            confidence = float(avg_probs[pred_idx])
            label = LABEL_NAMES[pred_idx]
            group = 'Engaged' if pred_idx in ENGAGED else 'Not Engaged'
            color = (0, 200, 0) if pred_idx in ENGAGED else (0, 80, 220)

            draw_label(frame_bgr, bbox_full, label, group, color, confidence)
            draw_top3_panel(frame_bgr, avg_probs)

        # FPS
        if frame_count % 30 == 0:
            now = time.time()
            fps = 30.0 / (now - fps_t0)
            fps_t0 = now

        # Overlay
        cv2.putText(frame_bgr, f'FPS: {fps:.1f}',
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (50, 220, 50), 2)
        cv2.putText(frame_bgr, f'Model: {args.model}',
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        cv2.putText(frame_bgr, "Bam 'q' de thoat",
                    (10, frame_bgr.shape[0] - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

        cv2.imshow('Student Engagement — Realtime Demo', frame_bgr)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    worker.stop()
    cap.release()
    cv2.destroyAllWindows()
    print('Bye.')


if __name__ == '__main__':
    main()
