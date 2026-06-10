"""
augmentation.py
---------------
Augmentation pipeline cho cả 2 dataset.

⚠️ INPUT/OUTPUT (v2 — fix mode collapse):
    Cả augment_train và augment_val nhận ảnh raw [0, 1] và trả ảnh raw [0, 1].
    Việc normalize_imagenet được gọi RIÊNG ở dataset pipeline (sau augment).

Lý do thay đổi v2:
    Phiên bản cũ augment ảnh đã normalize ImageNet [-2.1, +2.6] → tf.image.random_hue
    và random_brightness/contrast tạo ra giá trị ngoài distribution mà
    pretrained backbone expect → model bị mode collapse.

Augmentation mới (nhẹ hơn, phù hợp khuôn mặt):
    1. Random horizontal flip (50%)
    2. Random brightness ±0.1     (nhẹ, tránh mất chi tiết khuôn mặt)
    3. Random contrast [0.9, 1.1] (nhẹ)
    4. Random crop sau pad 5%     (mô phỏng dịch chuyển khuôn mặt nhẹ)

Bỏ so với v1:
    ✗ Random hue       — gây nhiễu cho biểu cảm khuôn mặt, model học sai pattern
    ✗ Random rotation  — có thể làm mất context biểu cảm khi nghiêng nhiều
"""

import tensorflow as tf

TARGET_SIZE = 224


def augment_train(image: tf.Tensor) -> tf.Tensor:
    """
    Augmentation cho tập train. Hoạt động trên ảnh raw [0, 1].

    Args:
        image: tf.Tensor float32, shape (224, 224, 3), giá trị [0, 1]

    Returns:
        tf.Tensor float32, shape (224, 224, 3), giá trị vẫn [0, 1].
    """
    # 1. Lật ngang ngẫu nhiên 50%
    image = tf.image.random_flip_left_right(image)

    # 2. Độ sáng ngẫu nhiên ±0.1 (nhẹ)
    image = tf.image.random_brightness(image, max_delta=0.1)

    # 3. Độ tương phản ngẫu nhiên [0.9, 1.1] (nhẹ)
    image = tf.image.random_contrast(image, lower=0.9, upper=1.1)

    # Clip sau brightness/contrast để giữ trong [0, 1]
    image = tf.clip_by_value(image, 0.0, 1.0)

    # 4. Random crop sau khi pad 5% (mô phỏng dịch chuyển khuôn mặt)
    image = _random_crop_after_pad(image, pad_ratio=0.05)

    return image


def augment_val(image: tf.Tensor) -> tf.Tensor:
    """
    Không augment — trả ảnh gốc cho val/test để đánh giá công bằng.

    Args:
        image: tf.Tensor float32, shape (224, 224, 3), giá trị [0, 1]

    Returns:
        tf.Tensor float32, không thay đổi.
    """
    return image


# ── Helper ──────────────────────────────────────────────────────────────────

def _random_crop_after_pad(image: tf.Tensor, pad_ratio: float = 0.05) -> tf.Tensor:
    """
    Pad ảnh thêm pad_ratio rồi crop ngẫu nhiên về TARGET_SIZE×TARGET_SIZE.
    Dùng REFLECT padding để tránh viền đen.

    pad_ratio=0.05 nghĩa là pad ~11 pixel mỗi phía → dịch chuyển khuôn mặt
    tối đa 11 pixel (5% kích thước ảnh) — đủ để học robustness về vị trí
    nhưng không làm mất context khuôn mặt.
    """
    pad_size = int(TARGET_SIZE * pad_ratio)
    padded = tf.pad(
        image,
        paddings=[[pad_size, pad_size], [pad_size, pad_size], [0, 0]],
        mode="REFLECT"
    )
    cropped = tf.image.random_crop(padded, size=[TARGET_SIZE, TARGET_SIZE, 3])
    return cropped
