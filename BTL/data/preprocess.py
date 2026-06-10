"""
preprocess.py
-------------
Các hàm tiền xử lý ảnh dùng chung cho cả 2 dataset.

⚠️ THỨ TỰ PIPELINE (v2 — fix mode collapse):
    load → decode → resize/crop → [augment nếu train] → normalize_imagenet

    → load_affectnet_image / load_and_crop_student trả ảnh raw [0, 1]
    → augment_train hoạt động trên [0, 1]
    → normalize_imagenet được gọi CUỐI CÙNG ở dataset pipeline

Hàm xuất:
    init_mtcnn()                       — khởi tạo MTCNN 1 lần
    load_affectnet_image(path)         — ảnh AffectNet, [0, 1]
    load_and_crop_student(path)        — ảnh Student với MTCNN crop, [0, 1]
    normalize_imagenet(image)          — chuẩn hóa ImageNet ở bước cuối
"""

import numpy as np
import tensorflow as tf
from pathlib import Path

# ── ImageNet statistics ──────────────────────────────────────────────────────
_IMAGENET_MEAN = tf.constant([0.485, 0.456, 0.406], dtype=tf.float32)
_IMAGENET_STD  = tf.constant([0.229, 0.224, 0.225], dtype=tf.float32)

TARGET_SIZE = 224  # pixel

# ── MTCNN (lazy init) ────────────────────────────────────────────────────────
_detector = None


def init_mtcnn():
    """
    Khởi tạo MTCNN detector. Gọi 1 lần trước khi dùng load_and_crop_student().
    Yêu cầu: pip install mtcnn
    """
    global _detector
    if _detector is None:
        try:
            from mtcnn import MTCNN
            _detector = MTCNN()
            print("[preprocess] MTCNN khởi tạo thành công.")
        except ImportError:
            print("[preprocess] WARN: mtcnn chưa cài. Chạy: pip install mtcnn")
            _detector = None
    return _detector


# ── Normalize ImageNet (gọi CUỐI CÙNG sau khi augment) ──────────────────────

def normalize_imagenet(image: tf.Tensor) -> tf.Tensor:
    """
    Chuẩn hóa ảnh về ImageNet mean/std.
    Gọi ở bước CUỐI CÙNG của pipeline (sau augment).

    Input : tf.Tensor float32, giá trị [0, 1], shape (..., H, W, 3)
    Output: tf.Tensor float32, shape giống vào, đã normalize ImageNet
    """
    return (image - _IMAGENET_MEAN) / _IMAGENET_STD


# ── AffectNet: load + resize (KHÔNG normalize ở đây) ────────────────────────

def load_affectnet_image(image_path: str) -> tf.Tensor:
    """
    Đọc ảnh AffectNet → resize 224×224. KHÔNG normalize.

    Args:
        image_path: đường dẫn tuyệt đối đến file .png

    Returns:
        tf.Tensor shape (224, 224, 3), float32, giá trị trong [0, 1].
        (Augment + normalize được gọi sau ở dataset pipeline.)
    """
    raw = tf.io.read_file(image_path)
    image = tf.image.decode_png(raw, channels=3)
    image = tf.image.resize(image, [TARGET_SIZE, TARGET_SIZE],
                            method=tf.image.ResizeMethod.BICUBIC)
    image = tf.cast(image, tf.float32) / 255.0
    # Resize bicubic có thể tạo giá trị vượt [0,1] nhẹ → clip lại cho an toàn
    image = tf.clip_by_value(image, 0.0, 1.0)
    return image


# ── Student Dataset: load + MTCNN crop (KHÔNG normalize ở đây) ──────────────

def load_and_crop_student(image_path: str, margin: float = 0.10) -> np.ndarray:
    """
    Đọc ảnh Student Dataset → MTCNN crop khuôn mặt → resize 224×224.
    KHÔNG normalize.

    Args:
        image_path : đường dẫn tuyệt đối đến file .jpg
        margin     : tỉ lệ mở rộng bounding box (default 10%)

    Returns:
        np.ndarray shape (224, 224, 3), float32, giá trị trong [0, 1].
        Nếu không detect được mặt → fallback resize toàn ảnh.
    """
    raw   = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(raw, channels=3)
    img_np = image.numpy()  # uint8

    face_crop = _try_mtcnn_crop(img_np, margin)
    if face_crop is None:
        face_crop = img_np   # fallback

    resized = tf.image.resize(
        tf.cast(face_crop, tf.float32),
        [TARGET_SIZE, TARGET_SIZE],
        method=tf.image.ResizeMethod.BICUBIC
    )
    # Đổi từ [0, 255] về [0, 1] + clip
    resized = tf.clip_by_value(resized / 255.0, 0.0, 1.0)
    return resized.numpy()


def _try_mtcnn_crop(img_np: np.ndarray, margin: float):
    """Thử detect face. Trả vùng crop (uint8) hoặc None."""
    global _detector
    if _detector is None:
        init_mtcnn()
    if _detector is None:
        return None

    try:
        results = _detector.detect_faces(img_np)
    except Exception as e:
        print(f"[preprocess] MTCNN error: {e}")
        return None

    if not results:
        return None

    best = max(results, key=lambda r: r["confidence"])
    x, y, w, h = best["box"]
    x, y = max(0, x), max(0, y)

    H, W = img_np.shape[:2]
    mx = int(w * margin)
    my = int(h * margin)
    x1 = max(0, x - mx); y1 = max(0, y - my)
    x2 = min(W, x + w + mx); y2 = min(H, y + h + my)

    cropped = img_np[y1:y2, x1:x2]
    if cropped.size == 0:
        return None
    return cropped


# ── Aliases để code cũ vẫn import được (deprecated) ─────────────────────────
# Giữ tương thích ngược nếu ai đó import resize_affectnet / detect_and_crop_face
# nhưng KHUYẾN KHÍCH chuyển sang tên mới.

def resize_affectnet(image_path: str) -> tf.Tensor:
    """⚠️ DEPRECATED — dùng load_affectnet_image() + normalize ở cuối pipeline."""
    return load_affectnet_image(image_path)


def detect_and_crop_face(image_path: str, margin: float = 0.10) -> np.ndarray:
    """⚠️ DEPRECATED — dùng load_and_crop_student() + normalize ở cuối pipeline."""
    return load_and_crop_student(image_path, margin)
