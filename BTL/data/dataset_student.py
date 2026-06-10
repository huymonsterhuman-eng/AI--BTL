"""
dataset_student.py
------------------
tf.data pipeline cho Student Engagement Dataset (Phase 2 — Fine-tuning).

⚠️ PIPELINE THỨ TỰ (v2 — fix mode collapse):
    load_and_crop_student (MTCNN crop → resize → [0,1])
        → augment_train (nếu train, vẫn [0,1])
        → normalize_imagenet (CUỐI cùng, ra range [-2.1, +2.6])

YÊU CẦU: chạy split_student.py trước để có 3 file CSV.

Hàm chính:
    build_student_dataset(split, data_dir, batch_size)
        → tf.data.Dataset

Cách dùng (Colab):
    sys.path.append('/content/drive/MyDrive/ProjectBTL/BTL/data')

    from preprocess import init_mtcnn
    init_mtcnn()

    from dataset_student import build_student_dataset
    train_ds = build_student_dataset('train', DATA_DIR, batch_size=16)
    val_ds   = build_student_dataset('val',   DATA_DIR, batch_size=16)
    test_ds  = build_student_dataset('test',  DATA_DIR, batch_size=16)
"""

from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

from preprocess import load_and_crop_student, normalize_imagenet
from augmentation import augment_train, augment_val

# ── Label map — 6 trạng thái engagement ─────────────────────────────────────
LABEL_MAP = {
    "Focused":      0,
    "Confused":     1,
    "Frustrated":   2,
    "Bored":        3,
    "Drowsy":       4,
    "Looking Away": 5,
}
NUM_CLASSES = len(LABEL_MAP)

_CSV_FILES = {
    "train": "student_train.csv",
    "val":   "student_val.csv",
    "test":  "student_test.csv",
}


def _load_csv(data_dir: str, split: str) -> pd.DataFrame:
    csv_path = Path(data_dir) / _CSV_FILES[split]
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Không tìm thấy {csv_path}. Hãy chạy split_student.py trước."
        )
    df = pd.read_csv(csv_path)
    print(f"[dataset_student] {split}: {len(df)} ảnh")
    print(df["label"].value_counts().to_string())
    return df


def _generator(filepaths: list, labels: list, is_train: bool):
    """
    Python generator: load ảnh + MTCNN crop + augment + normalize.

    PIPELINE THỨ TỰ:
        1. load_and_crop_student → ảnh [0, 1]
        2. augment_train (nếu train) → vẫn [0, 1]
        3. normalize_imagenet → [-2.1, +2.6]
    """
    for fp, lb in zip(filepaths, labels):
        try:
            # Bước 1: Load + MTCNN crop + resize (trả [0, 1])
            image = load_and_crop_student(fp)
            image = tf.constant(image, dtype=tf.float32)
            image.set_shape([224, 224, 3])

            # Bước 2: Augment trên ảnh [0, 1]
            if is_train:
                image = augment_train(image)
            else:
                image = augment_val(image)

            # Bước 3: Normalize ImageNet ở CUỐI cùng
            image = normalize_imagenet(image)

            label_onehot = tf.one_hot(lb, NUM_CLASSES)
            yield image, label_onehot

        except Exception as e:
            print(f"[dataset_student] Bỏ qua {fp}: {e}")
            continue


def build_student_dataset(
    split: str,
    data_dir: str,
    batch_size: int = 16,
    shuffle_buffer: int = 1000,
    cache: bool = False,
) -> tf.data.Dataset:
    """
    Xây dựng tf.data.Dataset cho Student Engagement Dataset.

    Args:
        split         : 'train', 'val', hoặc 'test'
        data_dir      : đường dẫn đến thư mục chứa 3 CSV
        batch_size    : kích thước batch (default 16 — dataset nhỏ)
        shuffle_buffer: buffer shuffle cho 'train'
        cache         : True = cache sau preprocess (cần ~2 GB RAM)

    Returns:
        tf.data.Dataset, mỗi item là (image, one_hot_label)
            image shape: (batch, 224, 224, 3), normalized ImageNet
            label shape: (batch, 6)
    """
    assert split in ("train", "val", "test"), \
        f"split = 'train'/'val'/'test', nhận: {split}"

    df = _load_csv(data_dir, split)

    # Map label string → int
    df["label_idx"] = df["label"].map(LABEL_MAP)
    missing = df["label_idx"].isna().sum()
    if missing > 0:
        print(f"[dataset_student] WARN: {missing} nhãn lạ, bỏ qua.")
        df = df.dropna(subset=["label_idx"])

    filepaths = df["filepath"].tolist()
    labels    = df["label_idx"].astype(int).tolist()

    is_train = (split == "train")

    ds = tf.data.Dataset.from_generator(
        generator=lambda: _generator(filepaths, labels, is_train),
        output_signature=(
            tf.TensorSpec(shape=(224, 224, 3), dtype=tf.float32),
            tf.TensorSpec(shape=(NUM_CLASSES,), dtype=tf.float32),
        )
    )

    if cache:
        ds = ds.cache()

    if is_train:
        ds = ds.shuffle(
            buffer_size=min(shuffle_buffer, len(filepaths)),
            seed=42,
            reshuffle_each_iteration=True
        )

    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.AUTOTUNE)

    return ds
