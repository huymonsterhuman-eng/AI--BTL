"""
dataset_affectnet.py
--------------------
tf.data pipeline cho AffectNet (Phase 1 — Pre-training).

⚠️ PIPELINE THỨ TỰ (v2 — fix mode collapse):
    load_affectnet_image (resize → [0,1])
        → augment_train (nếu train, vẫn [0,1])
        → normalize_imagenet (CUỐI cùng, ra range [-2.1, +2.6])

Hàm chính:
    build_affectnet_dataset(split, data_dir, batch_size, subset_per_class)
        → (tf.data.Dataset, class_weights_dict)

Cách dùng (Colab):
    sys.path.append('/content/drive/MyDrive/ProjectBTL/BTL/data')
    from dataset_affectnet import build_affectnet_dataset

    train_ds, class_weights = build_affectnet_dataset(
        split='train', data_dir=AFFECTNET_DIR,
        batch_size=32, subset_per_class=1500
    )
    test_ds, _ = build_affectnet_dataset('test', AFFECTNET_DIR, batch_size=32)

    model.fit(train_ds, validation_data=test_ds, class_weight=class_weights, ...)
"""

import random
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight

from preprocess import load_affectnet_image, normalize_imagenet
from augmentation import augment_train, augment_val

# ── Label map (alphabetical) ────────────────────────────────────────────────
LABEL_MAP = {
    "anger":    0,
    "contempt": 1,
    "disgust":  2,
    "fear":     3,
    "happy":    4,
    "neutral":  5,
    "sad":      6,
    "surprise": 7,
}
NUM_CLASSES = len(LABEL_MAP)

# Test set có 2 folder viết hoa: Anger, Contempt → đã được normalize lowercase
_SPLIT_DIRS = {
    "train": "Train",
    "test":  "Test",
}


def _scan_images(split_dir: Path, subset_per_class: int | None) -> tuple[list, list]:
    """
    Duyệt thư mục split → (filepaths, int_labels).
    Normalize tên class về lowercase trước khi map vào LABEL_MAP.
    """
    filepaths, labels = [], []

    for class_dir in sorted(split_dir.iterdir()):
        if not class_dir.is_dir():
            continue
        class_name = class_dir.name.lower()
        if class_name not in LABEL_MAP:
            print(f"[dataset_affectnet] Bỏ qua thư mục: {class_dir.name}")
            continue

        label_idx = LABEL_MAP[class_name]
        imgs = [
            str(p) for p in class_dir.iterdir()
            if p.suffix.lower() in (".png", ".jpg", ".jpeg")
        ]

        if subset_per_class is not None and len(imgs) > subset_per_class:
            random.seed(42)
            imgs = random.sample(imgs, subset_per_class)

        filepaths.extend(imgs)
        labels.extend([label_idx] * len(imgs))

    return filepaths, labels


def _compute_class_weights(labels: list) -> dict:
    """Tính class weights theo phương pháp balanced."""
    classes = np.arange(NUM_CLASSES)
    weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=np.array(labels)
    )
    return {int(c): float(w) for c, w in zip(classes, weights)}


def _load_preprocess(filepath: tf.Tensor, label: tf.Tensor, is_train: bool):
    """
    PIPELINE:
        1. Load ảnh raw [0, 1] qua py_function
        2. Augment (nếu train) — vẫn [0, 1]
        3. Normalize ImageNet — ra range [-2.1, +2.6]
        4. One-hot label
    """
    # Bước 1: Load + resize (trả ảnh [0, 1])
    image = tf.py_function(
        func=lambda p: load_affectnet_image(p.numpy().decode("utf-8")),
        inp=[filepath],
        Tout=tf.float32
    )
    image.set_shape([224, 224, 3])

    # Bước 2: Augment trên ảnh [0, 1]
    if is_train:
        image = augment_train(image)
    else:
        image = augment_val(image)

    # Bước 3: Normalize ImageNet ở CUỐI cùng
    image = normalize_imagenet(image)

    # Bước 4: One-hot label
    label_onehot = tf.one_hot(label, NUM_CLASSES)
    return image, label_onehot


def build_affectnet_dataset(
    split: str,
    data_dir: str,
    batch_size: int = 32,
    subset_per_class: int | None = None,
    shuffle_buffer: int = 20000,
) -> tuple[tf.data.Dataset, dict]:
    """
    Xây dựng tf.data.Dataset cho AffectNet.

    Args:
        split           : 'train' hoặc 'test'
        data_dir        : đường dẫn đến thư mục AffectNet/
        batch_size      : kích thước batch (default 32)
        subset_per_class: giới hạn ảnh/class (None = lấy tất cả)
        shuffle_buffer  : buffer shuffle (chỉ cho 'train')

    Returns:
        (tf.data.Dataset, class_weights_dict)
            Dataset: batch (image, one_hot_label)
                     image shape: (batch, 224, 224, 3), normalized ImageNet
                     label shape: (batch, 8)
            class_weights: dict {0..7: weight} dùng cho model.fit
    """
    assert split in ("train", "test"), f"split = 'train'/'test', nhận: {split}"

    data_path = Path(data_dir) / _SPLIT_DIRS[split]
    if not data_path.exists():
        raise FileNotFoundError(f"Không tìm thấy: {data_path}")

    filepaths, labels = _scan_images(data_path, subset_per_class)
    print(f"[dataset_affectnet] {split}: {len(filepaths)} ảnh")

    class_weights = _compute_class_weights(labels)

    is_train = (split == "train")
    ds = tf.data.Dataset.from_tensor_slices(
        (filepaths, tf.cast(labels, tf.int32))
    )

    if is_train:
        # Shuffle ở mức filepath (trước khi load) — rẻ và hiệu quả
        ds = ds.shuffle(
            buffer_size=min(shuffle_buffer, len(filepaths)),
            seed=42,
            reshuffle_each_iteration=True
        )

    ds = ds.map(
        lambda fp, lb: _load_preprocess(fp, lb, is_train),
        num_parallel_calls=tf.data.AUTOTUNE
    )
    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.AUTOTUNE)

    return ds, class_weights
