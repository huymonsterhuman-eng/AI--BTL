"""
resnet50.py
-----------
Định nghĩa kiến trúc ResNet50 cho bài toán FER + Student Engagement.

Kiến trúc:
    Input (224×224×3)
        ↓
    ResNet50 (pretrained ImageNet, include_top=False)  ~23.6M params
        ↓
    GlobalAveragePooling2D                  (2048-dim feature vector)
        ↓
    Dropout(0.3)
        ↓
    Dense(256, activation='relu')
        ↓
    Dropout(0.2)
        ↓
    Dense(num_classes, activation='softmax')

Lưu ý kỹ thuật:
- Pipeline Người 1 chuẩn hóa theo PyTorch-style (chia 255, ImageNet mean/std).
  Keras ResNet50 gốc dùng caffe-style nhưng transfer learning vẫn hoạt động tốt
  vì backbone sẽ được fine-tune lại.
- ResNet50 có BatchNormalization → khi unfreeze cần dùng lr nhỏ (1e-5) để tránh
  phá hỏng BN statistics đã học từ ImageNet.

Hàm export:
    build_model(num_classes, input_shape)
    freeze_backbone(model)
    unfreeze_backbone(model)
    load_phase1_weights_for_phase2(model_p2, p1_weights_path)
"""

import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import ResNet50


# ── Tên backbone (default name của Keras ResNet50) ──────────────────────────
BACKBONE_NAME = "resnet50"


def build_model(num_classes: int, input_shape: tuple = (224, 224, 3)) -> Model:
    """
    Xây dựng ResNet50 + classification head.

    Args:
        num_classes : 8 cho Phase 1 (AffectNet) hoặc 6 cho Phase 2 (Student)
        input_shape : (H, W, C) mặc định (224, 224, 3)

    Returns:
        tf.keras.Model — chưa compile
    """
    inputs = layers.Input(shape=input_shape, name="input_image")

    # Backbone — pretrained ImageNet, bỏ classification head gốc
    backbone = ResNet50(
        input_shape=input_shape,
        include_top=False,
        weights="imagenet",
    )

    # Feature extraction
    x = backbone(inputs, training=False)            # (None, 7, 7, 2048)
    x = layers.GlobalAveragePooling2D(name="gap")(x)  # (None, 2048)

    # Classification head
    x = layers.Dropout(0.3, name="dropout_1")(x)
    x = layers.Dense(256, activation="relu", name="dense_256")(x)
    x = layers.Dropout(0.2, name="dropout_2")(x)
    outputs = layers.Dense(
        num_classes, activation="softmax", name="predictions"
    )(x)

    model = Model(inputs, outputs, name=f"ResNet50_FER_{num_classes}cls")
    return model


def freeze_backbone(model: Model) -> None:
    """Đóng băng backbone — chỉ train classification head."""
    backbone = model.get_layer(BACKBONE_NAME)
    backbone.trainable = False
    n = sum(tf.size(w).numpy() for w in model.trainable_weights)
    print(f"[resnet50] Backbone FROZEN — trainable params: {n:,}")


def unfreeze_backbone(model: Model) -> None:
    """Mở khóa backbone — train cả backbone + head với lr nhỏ."""
    backbone = model.get_layer(BACKBONE_NAME)
    backbone.trainable = True
    n = sum(tf.size(w).numpy() for w in model.trainable_weights)
    print(f"[resnet50] Backbone UNFROZEN — trainable params: {n:,}")


def load_phase1_weights_for_phase2(model_p2: Model, p1_weights_path: str) -> None:
    """
    Transfer weights từ Phase 1 (8 class) sang Phase 2 (6 class).
    Copy: backbone + dense_256
    Bỏ qua: predictions (8→6 class mismatch)
    """
    model_p1 = build_model(num_classes=8)
    model_p1.load_weights(p1_weights_path)
    print(f"[resnet50] Loaded Phase 1 weights từ {p1_weights_path}")

    for layer_name in [BACKBONE_NAME, "dense_256"]:
        weights = model_p1.get_layer(layer_name).get_weights()
        model_p2.get_layer(layer_name).set_weights(weights)
        print(f"[resnet50]   Copied weights: {layer_name}")

    print(f"[resnet50] Layer 'predictions' (6 class) giữ random init.")


# ── Quick test ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    m = build_model(num_classes=8)
    print(f"Total params: {m.count_params():,}")
    freeze_backbone(m)
    unfreeze_backbone(m)
