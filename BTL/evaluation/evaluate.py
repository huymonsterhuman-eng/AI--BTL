"""
evaluate.py
-----------
Tổng hợp kết quả 4 model từ các file metrics.json đã có,
in bảng so sánh và xuất biểu đồ tổng quan.

Cách dùng:
    cd "D:/HK2025-2026/HK2/AI/ProjectBTL"
    python BTL/evaluation/evaluate.py

Yêu cầu: pip install matplotlib numpy

Output:
    - In bảng so sánh ra màn hình
    - Lưu biểu đồ tổng hợp: BTL/results/comparison_chart.png
"""

import json
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# ── Cấu hình đường dẫn ──────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent  # ProjectBTL/
RESULTS_DIR  = PROJECT_ROOT / "BTL" / "results"

MODEL_PATHS = {
    "MobileNetV3":              RESULTS_DIR / "mobilenetv3"          / "mobilenetv3_metrics.json",
    "ResNet50":                 RESULTS_DIR / "resnet50"             / "resnet50_metrics.json",
    "EfficientNet-B0":          RESULTS_DIR / "efficientnet"         / "efficientnet_metrics.json",
    "EfficientNet-B0+Landmarks": RESULTS_DIR / "efficientnet_landmark"/ "efficientnet_lm_metrics.json",
}

PARAMS_M = {
    "MobileNetV3":               2.5,
    "ResNet50":                 23.6,
    "EfficientNet-B0":           5.3,
    "EfficientNet-B0+Landmarks": 5.9,
}

LABEL_NAMES = [
    "Focused", "Confused", "Frustrated",
    "Bored", "Drowsy", "Looking Away",
]


# ── Load metrics ────────────────────────────────────────────────────────────

def load_all_metrics():
    """Đọc 4 file metrics.json. Trả về dict {model_name: metrics_dict}."""
    metrics = {}
    for name, path in MODEL_PATHS.items():
        if not path.exists():
            print(f"⚠️  Bỏ qua {name}: không tìm thấy {path}")
            continue
        with open(path, encoding="utf-8") as f:
            metrics[name] = json.load(f)
    return metrics


# ── In bảng so sánh ─────────────────────────────────────────────────────────

def print_summary_table(metrics):
    """In bảng so sánh tổng quan ra màn hình."""
    print()
    print("=" * 90)
    print(f"{'BẢNG SO SÁNH 4 MÔ HÌNH':^90}")
    print("=" * 90)

    header = f"{'Model':<28} {'Params':>9} {'Acc':>8} {'Macro F1':>10} {'Precision':>11} {'Recall':>9}"
    print(header)
    print("-" * 90)

    for name, m in metrics.items():
        params = f"{PARAMS_M.get(name, 0):.1f}M"
        acc = f"{m['phase2_accuracy']*100:.2f}%"
        f1  = f"{m['macro_f1']:.4f}"
        pr  = f"{m['precision']:.4f}"
        rc  = f"{m['recall']:.4f}"
        print(f"{name:<28} {params:>9} {acc:>8} {f1:>10} {pr:>11} {rc:>9}")

    # Best model
    best = max(metrics.items(), key=lambda kv: kv[1]["macro_f1"])
    print("-" * 90)
    print(f"🏆 Model tốt nhất theo Macro F1: {best[0]} (F1 = {best[1]['macro_f1']:.4f})")
    print("=" * 90)


def print_per_class_table(metrics):
    """In bảng F1-score chi tiết theo từng class."""
    print()
    print("=" * 90)
    print(f"{'F1-SCORE THEO TỪNG LỚP':^90}")
    print("=" * 90)

    header = f"{'Class':<15}"
    for name in metrics.keys():
        # Rút gọn tên cho gọn cột
        short = name.replace("EfficientNet-B0", "EffNet").replace("+Landmarks", "+LM")
        header += f" {short:>16}"
    print(header)
    print("-" * 90)

    for label in LABEL_NAMES:
        row = f"{label:<15}"
        for name, m in metrics.items():
            f1 = m["per_class_f1"].get(label, 0)
            marker = ""
            # Đánh dấu giá trị cao nhất hàng
            best_value = max(mm["per_class_f1"].get(label, 0) for mm in metrics.values())
            if abs(f1 - best_value) < 1e-6:
                marker = "★"
            row += f" {f1:>14.4f}{marker:<2}"
        print(row)
    print("=" * 90)


# ── Vẽ biểu đồ ──────────────────────────────────────────────────────────────

def plot_comparison_chart(metrics, save_path):
    """
    Biểu đồ 2 hàng:
    - Hàng 1: Accuracy + Macro F1 (bar chart so sánh)
    - Hàng 2: Per-class F1 (grouped bar chart)
    """
    fig, axes = plt.subplots(2, 1, figsize=(13, 9))
    models = list(metrics.keys())

    # ── Hàng 1: Accuracy + Macro F1 ─────────────────────────────────────────
    acc_vals = [metrics[m]["phase2_accuracy"] for m in models]
    f1_vals  = [metrics[m]["macro_f1"]        for m in models]

    x = np.arange(len(models))
    width = 0.35

    bars1 = axes[0].bar(x - width/2, acc_vals, width, label="Accuracy",
                         color="#3498db", edgecolor="white")
    bars2 = axes[0].bar(x + width/2, f1_vals,  width, label="Macro F1",
                         color="#e74c3c", edgecolor="white")

    axes[0].set_xticks(x)
    axes[0].set_xticklabels(models, rotation=10, ha="right", fontsize=9)
    axes[0].set_ylabel("Giá trị")
    axes[0].set_ylim(0.85, 1.01)
    axes[0].set_title("So sánh Accuracy & Macro F1 (Phase 2 — Student Engagement)",
                       fontsize=11, fontweight="bold")
    axes[0].legend(loc="lower right")
    axes[0].grid(axis="y", alpha=0.3)

    for bar in bars1:
        h = bar.get_height()
        axes[0].text(bar.get_x() + bar.get_width()/2, h + 0.002,
                      f"{h:.3f}", ha="center", va="bottom", fontsize=8)
    for bar in bars2:
        h = bar.get_height()
        axes[0].text(bar.get_x() + bar.get_width()/2, h + 0.002,
                      f"{h:.3f}", ha="center", va="bottom", fontsize=8)

    # ── Hàng 2: Per-class F1 ────────────────────────────────────────────────
    n_classes = len(LABEL_NAMES)
    n_models = len(models)
    width2 = 0.8 / n_models
    x2 = np.arange(n_classes)

    colors = ["#9b59b6", "#3498db", "#1abc9c", "#e67e22"]
    for i, model in enumerate(models):
        vals = [metrics[model]["per_class_f1"].get(c, 0) for c in LABEL_NAMES]
        offset = (i - n_models/2 + 0.5) * width2
        axes[1].bar(x2 + offset, vals, width2, label=model,
                     color=colors[i % len(colors)], edgecolor="white")

    axes[1].set_xticks(x2)
    axes[1].set_xticklabels(LABEL_NAMES, rotation=15, ha="right", fontsize=9)
    axes[1].set_ylabel("F1-score")
    axes[1].set_ylim(0.75, 1.02)
    axes[1].set_title("F1-score theo từng lớp", fontsize=11, fontweight="bold")
    axes[1].legend(loc="lower left", fontsize=8)
    axes[1].grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✅ Đã lưu biểu đồ: {save_path}")


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    metrics = load_all_metrics()
    if not metrics:
        print("Không tìm thấy file metrics nào.")
        sys.exit(1)

    print_summary_table(metrics)
    print_per_class_table(metrics)

    save_path = RESULTS_DIR / "comparison_chart.png"
    plot_comparison_chart(metrics, save_path)


if __name__ == "__main__":
    main()
