"""
efficientnet.py
---------------
Định nghĩa kiến trúc EfficientNet-B0 cho bài toán FER + Student Engagement.

Đây là PROPOSED MODEL chính của dự án — cân bằng tốt giữa accuracy và
chi phí tính toán nhờ Compound Scaling.

Kiến trúc:
    Input (224×224×3)
        ↓
    EfficientNet-B0 (pretrained ImageNet, include_top=False)  ~4M params
        ↓
    GlobalAveragePooling2D                  (1280-dim feature vector)
        ↓
    Dropout(0.3)
        ↓
    Dense(256, activation='relu')
        ↓
    Dropout(0.2)
        ↓
    Dense(num_classes, activation='softmax')

Hàm export:
    build_model(num_classes, input_shape)
    freeze_backbone(model)
    unfreeze_backbone(model)
    load_phase1_weights_for_phase2(model_p2, p1_weights_path)
"""

import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import EfficientNetB0


# ── Tên backbone (default name của Keras EfficientNetB0) ────────────────────
BACKBONE_NAME = "efficientnetb0"


def build_model(num_classes: int, input_shape: tuple = (224, 224, 3)) -> Model:
    """
    Xây dựng EfficientNet-B0 + classification head.

    ⚠️ FIX QUAN TRỌNG:
        Keras EfficientNet-B0 có BUILT-IN preprocessing (Rescaling /255 + Normalization).
        Pipeline ta đã normalize ImageNet → cần DENORMALIZE input về [0, 255]
        trước khi vào backbone để built-in preprocessing hoạt động đúng.

        Công thức ngược: x_raw_255 = (x_normalized * std + mean) * 255

    Args:
        num_classes : 8 cho Phase 1 (AffectNet) hoặc 6 cho Phase 2 (Student)
        input_shape : (H, W, C) mặc định (224, 224, 3)

    Returns:
        tf.keras.Model — chưa compile
    """
    inputs = layers.Input(shape=input_shape, name="input_image")

    # ── Denormalize: ImageNet normalized → [0, 255] raw ─────────────────────
    # Dùng Rescaling thay Lambda để JSON serialize được (fix save_weights error).
    # Toán: y = (x * std + mean) * 255 = x * (std * 255) + (mean * 255)
    x_for_backbone = layers.Rescaling(
        scale=[0.229 * 255.0, 0.224 * 255.0, 0.225 * 255.0],
        offset=[0.485 * 255.0, 0.456 * 255.0, 0.406 * 255.0],
        name="denorm_to_0_255",
    )(inputs)

    # Backbone — pretrained ImageNet
    backbone = EfficientNetB0(
        input_shape=input_shape,
        include_top=False,
        weights="imagenet",
    )

    # Feature extraction (backbone tự normalize lại bên trong)
    x = backbone(x_for_backbone, training=False)    # (None, 7, 7, 1280)
    x = layers.GlobalAveragePooling2D(name="gap")(x)  # (None, 1280)

    # Classification head
    x = layers.Dropout(0.3, name="dropout_1")(x)
    x = layers.Dense(256, activation="relu", name="dense_256")(x)
    x = layers.Dropout(0.2, name="dropout_2")(x)
    outputs = layers.Dense(
        num_classes, activation="softmax", name="predictions"
    )(x)

    model = Model(inputs, outputs, name=f"EfficientNetB0_FER_{num_classes}cls")
    return model


def freeze_backbone(model: Model) -> None:
    """Đóng băng backbone — chỉ train classification head."""
    backbone = model.get_layer(BACKBONE_NAME)
    backbone.trainable = False
    n = sum(tf.size(w).numpy() for w in model.trainable_weights)
    print(f"[efficientnet] Backbone FROZEN — trainable params: {n:,}")


def unfreeze_backbone(model: Model) -> None:
    """Mở khóa backbone — train cả backbone + head với lr nhỏ."""
    backbone = model.get_layer(BACKBONE_NAME)
    backbone.trainable = True
    n = sum(tf.size(w).numpy() for w in model.trainable_weights)
    print(f"[efficientnet] Backbone UNFROZEN — trainable params: {n:,}")


def load_phase1_weights_for_phase2(model_p2: Model, p1_weights_path: str) -> None:
    """
    Transfer weights từ Phase 1 (8 class) sang Phase 2 (6 class).
    Copy: backbone + dense_256
    Bỏ qua: predictions (8 → 6 class mismatch)
    """
    model_p1 = build_model(num_classes=8)
    model_p1.load_weights(p1_weights_path)
    print(f"[efficientnet] Loaded Phase 1 weights từ {p1_weights_path}")

    for layer_name in [BACKBONE_NAME, "dense_256"]:
        weights = model_p1.get_layer(layer_name).get_weights()
        model_p2.get_layer(layer_name).set_weights(weights)
        print(f"[efficientnet]   Copied weights: {layer_name}")

    print(f"[efficientnet] Layer 'predictions' (6 class) giữ random init.")


# ── Quick test ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    m = build_model(num_classes=8)
    print(f"Total params: {m.count_params():,}")
    freeze_backbone(m)
    unfreeze_backbone(m)
