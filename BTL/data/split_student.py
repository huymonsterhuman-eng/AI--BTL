"""
split_student.py
----------------
Stratified 70/15/15 split cho Student Engagement Dataset.
Chạy file này MỘT LẦN để tạo ra 3 CSV dùng cho training.

Output:
    student_train.csv  (~1,484 ảnh)
    student_val.csv    (~318 ảnh)
    student_test.csv   (~318 ảnh)

Mỗi CSV có 2 cột: filepath, label (6 lớp chi tiết)

Cách dùng (local):
    python split_student.py

Cách dùng (Colab):
    STUDENT_DIR = '/content/drive/MyDrive/ProjectBTL/Student Dataset/Student-engagement-dataset'
    OUT_DIR     = '/content/drive/MyDrive/ProjectBTL/BTL/data'
    python split_student.py --student_dir STUDENT_DIR --out_dir OUT_DIR
"""

import os
import argparse
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

# ── Cấu hình mặc định (local Windows) ──────────────────────────────────────
DEFAULT_STUDENT_DIR = r"D:\HK2025-2026\HK2\AI\ProjectBTL\Student Dataset\Student-engagement-dataset"
DEFAULT_OUT_DIR     = r"D:\HK2025-2026\HK2\AI\ProjectBTL\BTL\data"

# ── Mapping tên thư mục → tên nhãn chuẩn ────────────────────────────────────
# Giữ đúng tên để dataset_student.py dùng label_map nhất quán
LABEL_MAP = {
    # Engaged
    "Confused":      "Confused",
    "Focused":       "Focused",
    "Frustrated":    "Frustrated",
    # Not Engaged
    "Bored":         "Bored",
    "Drowsy":        "Drowsy",
    "Looking Away":  "Looking Away",
}


def collect_samples(student_dir: str) -> pd.DataFrame:
    """Duyệt toàn bộ thư mục, thu thập (filepath, label)."""
    records = []
    root = Path(student_dir)

    for group_dir in root.iterdir():          # Engaged / Not Engaged
        if not group_dir.is_dir():
            continue
        for class_dir in group_dir.iterdir(): # Confused, Focused, ...
            if not class_dir.is_dir():
                continue
            label = LABEL_MAP.get(class_dir.name)
            if label is None:
                print(f"[WARN] Bỏ qua thư mục không nhận diện được: {class_dir}")
                continue
            for img_path in class_dir.iterdir():
                if img_path.suffix.lower() in (".jpg", ".jpeg", ".png"):
                    records.append({"filepath": str(img_path), "label": label})

    df = pd.DataFrame(records)
    print(f"Tổng số ảnh thu thập: {len(df)}")
    print(df["label"].value_counts().to_string())
    return df


def split_and_save(df: pd.DataFrame, out_dir: str):
    """Stratified 70 / 15 / 15 split, lưu CSV."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Bước 1: tách 70% train vs 30% còn lại
    train_df, temp_df = train_test_split(
        df, test_size=0.30, stratify=df["label"], random_state=42
    )

    # Bước 2: tách 30% còn lại thành 15% val + 15% test (tức 50/50)
    val_df, test_df = train_test_split(
        temp_df, test_size=0.50, stratify=temp_df["label"], random_state=42
    )

    # Lưu file
    train_df.to_csv(out / "student_train.csv", index=False)
    val_df.to_csv(out / "student_val.csv",   index=False)
    test_df.to_csv(out / "student_test.csv",  index=False)

    print(f"\nSplit hoàn tất:")
    print(f"  Train : {len(train_df):>5} ảnh  → {out / 'student_train.csv'}")
    print(f"  Val   : {len(val_df):>5} ảnh  → {out / 'student_val.csv'}")
    print(f"  Test  : {len(test_df):>5} ảnh  → {out / 'student_test.csv'}")

    # Kiểm tra phân bố Drowsy (lớp nhỏ nhất)
    print("\nPhân bố Drowsy qua các split:")
    for name, d in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
        n = (d["label"] == "Drowsy").sum()
        print(f"  {name}: {n} ảnh")


def main():
    parser = argparse.ArgumentParser(description="Split Student Engagement Dataset")
    parser.add_argument("--student_dir", default=DEFAULT_STUDENT_DIR)
    parser.add_argument("--out_dir",     default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    df = collect_samples(args.student_dir)
    split_and_save(df, args.out_dir)


if __name__ == "__main__":
    main()
