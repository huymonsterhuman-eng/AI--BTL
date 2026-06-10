"""
demo_upload.py
--------------
Demo nhận diện trạng thái tương tác từ ảnh tĩnh.

Cách dùng (local Windows):
    pip install tensorflow opencv-python mtcnn mediapipe
    python demo_upload.py --image path/to/photo.jpg

    # Dùng model EfficientNet+Landmarks (mặc định):
    python demo_upload.py --image photo.jpg --model landmarks

    # Hoặc model EfficientNet thuần (đơn giản, nhanh):
    python demo_upload.py --image photo.jpg --model efficientnet

Input  : 1 file ảnh
Output : nhãn trạng thái + độ tin cậy, đồng thời hiển thị ảnh
         có bounding box và nhãn lên màn hình.
"""

import argparse
import os
import sys
import numpy as np
import cv2
import tensorflow as tf

# ── Constants ───────────────────────────────────────────────────────────────
LABEL_NAMES = [
    'Focused', 'Confused', 'Frustrated',
    'Bored', 'Drowsy', 'Looking Away'
]
ENGAGED = {0, 1, 2}    # 3 lớp đầu = Engaged
NOT_ENGAGED = {3, 4, 5}

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# Đường dẫn weights — đổi nếu cần
DEFAULT_WEIGHTS = {
    'efficientnet': 'BTL/weights/efficientnet/efficientnet_phase2.weights.h5',
    'landmarks':    'BTL/weights/efficientnet_landmark/efficientnet_lm_phase2.weights.h5',
}


def detect_and_crop_face(img_rgb, margin=0.10):
    """MTCNN detect + crop khuôn mặt. Trả về (cropped, bbox) hoặc (None, None)."""
    from mtcnn import MTCNN
    detector = MTCNN()
    results = detector.detect_faces(img_rgb)
    if not results:
        return None, None

    best = max(results, key=lambda r: r['confidence'])
    x, y, w, h = best['box']
    x, y = max(0, x), max(0, y)
    H, W = img_rgb.shape[:2]
    mx, my = int(w * margin), int(h * margin)
    x1 = max(0, x - mx); y1 = max(0, y - my)
    x2 = min(W, x + w + mx); y2 = min(H, y + h + my)
    return img_rgb[y1:y2, x1:x2], (x1, y1, x2, y2)


def extract_landmarks(img_rgb_224):
    """Trích landmark vector 956-dim từ MediaPipe Face Landmarker (Tasks API)."""
    import urllib.request
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision

    MODEL_URL  = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
    MODEL_PATH = "face_landmarker.task"
    if not os.path.exists(MODEL_PATH):
        print("Đang tải MediaPipe model...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)

    base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.FaceLandmarkerOptions(base_options=base_options, num_faces=1)
    detector = vision.FaceLandmarker.create_from_options(options)

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


def preprocess_image(face_rgb):
    """Resize 224 + normalize ImageNet."""
    resized = cv2.resize(face_rgb, (224, 224))
    img = resized.astype(np.float32) / 255.0
    img = (img - IMAGENET_MEAN) / IMAGENET_STD
    return img, resized  # trả cả phiên bản đã normalize và phiên bản 224 raw


def build_model(model_type):
    """Build kiến trúc model phù hợp."""
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'models'))
    if model_type == 'landmarks':
        from efficientnet_landmarks import build_model as build_lm
        return build_lm(num_classes=6)
    else:
        from efficientnet import build_model as build_ef
        return build_ef(num_classes=6)


def predict(model, model_type, face_rgb):
    """Dự đoán trạng thái từ ảnh khuôn mặt đã crop."""
    img_norm, img_224 = preprocess_image(face_rgb)
    img_batch = np.expand_dims(img_norm, axis=0)

    if model_type == 'landmarks':
        lm = extract_landmarks(img_224)
        lm_batch = np.expand_dims(lm, axis=0)
        probs = model.predict([img_batch, lm_batch], verbose=0)[0]
    else:
        probs = model.predict(img_batch, verbose=0)[0]

    pred_idx = int(np.argmax(probs))
    return pred_idx, float(probs[pred_idx]), probs


def visualize(img_rgb, bbox, pred_idx, confidence, all_probs):
    """Vẽ bounding box + label lên ảnh và hiển thị."""
    img_show = img_rgb.copy()
    label = LABEL_NAMES[pred_idx]
    group = 'Engaged' if pred_idx in ENGAGED else 'Not Engaged'
    color = (0, 255, 0) if pred_idx in ENGAGED else (0, 100, 255)

    if bbox is not None:
        x1, y1, x2, y2 = bbox
        cv2.rectangle(img_show, (x1, y1), (x2, y2), color, 2)
        text = f'{label} ({confidence*100:.1f}%)'
        cv2.putText(img_show, text, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.putText(img_show, group, (x1, y2 + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # Bảng xác suất
    print('\n' + '=' * 50)
    print(f'  DỰ ĐOÁN: {label}  →  {group}')
    print(f'  Confidence: {confidence*100:.2f}%')
    print('=' * 50)
    print(f'\n{"Label":<15} {"Prob":>8}')
    print('-' * 25)
    for i, name in enumerate(LABEL_NAMES):
        marker = ' ←' if i == pred_idx else ''
        print(f'{name:<15} {all_probs[i]*100:>7.2f}%{marker}')

    # Hiển thị OpenCV
    img_bgr = cv2.cvtColor(img_show, cv2.COLOR_RGB2BGR)
    cv2.imshow('Demo Prediction', img_bgr)
    print('\nBấm phím bất kỳ để đóng cửa sổ...')
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description='Demo upload ảnh predict engagement')
    parser.add_argument('--image', required=True, help='Đường dẫn ảnh đầu vào')
    parser.add_argument('--model', choices=['efficientnet', 'landmarks'],
                        default='landmarks', help='Loại model dùng cho demo')
    parser.add_argument('--weights', default=None,
                        help='Đường dẫn weights (mặc định lấy từ BTL/weights/)')
    args = parser.parse_args()

    # Load ảnh
    img_bgr = cv2.imread(args.image)
    if img_bgr is None:
        print(f'Lỗi: không đọc được {args.image}')
        return
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    print(f'Đã load: {args.image} — kích thước {img_rgb.shape}')

    # MTCNN detect khuôn mặt
    print('Đang phát hiện khuôn mặt...')
    face, bbox = detect_and_crop_face(img_rgb)
    if face is None:
        print('Không phát hiện được khuôn mặt — sẽ dự đoán trên toàn ảnh.')
        face = img_rgb
        bbox = None

    # Build model
    print(f'Đang load model "{args.model}"...')
    model = build_model(args.model)
    weights_path = args.weights or DEFAULT_WEIGHTS[args.model]
    if not os.path.exists(weights_path):
        print(f'Lỗi: không tìm thấy weights {weights_path}')
        return
    model.load_weights(weights_path)
    print(f'Đã load weights: {weights_path}')

    # Predict
    pred_idx, confidence, all_probs = predict(model, args.model, face)

    # Hiển thị
    visualize(img_rgb, bbox, pred_idx, confidence, all_probs)


if __name__ == '__main__':
    main()
