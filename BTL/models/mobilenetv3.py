"""
mobilenetv3.py
--------------
Định nghĩa kiến trúc MobileNetV3Small cho bài toán FER + Student Engagement.

Kiến trúc:
    Input (224×224×3)
        ↓
    MobileNetV3Small (pretrained ImageNet, include_top=False)  ~2.5M params
        ↓
    GlobalAveragePooling2D                  (576-dim feature vector)
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
from tensorflow.keras.applications import MobileNetV3Small


# ── Tên backbone (đúng casing mặc định của Keras MobileNetV3Small) ──────────
BACKBONE_NAME = "MobileNetV3Small"


def build_model(num_classes: int, input_shape: tuple = (224, 224, 3)) -> Model:
    """
    Xây dựng MobileNetV3Small + classification head.

    Args:
        num_classes : 8 cho Phase 1 (AffectNet) hoặc 6 cho Phase 2 (Student)
        input_shape : (H, W, C) mặc định (224, 224, 3)

    Returns:
        tf.keras.Model — chưa compile
    """
    inputs = layers.Input(shape=input_shape, name="input_image")

    # Backbone — pretrained ImageNet, bỏ classification head gốc
    # Default name = "MobileNetV3Small" (Keras tự đặt)
    backbone = MobileNetV3Small(
        input_shape=input_shape,
        include_top=False,
        weights="imagenet",
        include_preprocessing=False,  # Ảnh đã được normalize ImageNet ở pipeline
    )

    # Feature extraction
    x = backbone(inputs, training=False)            # (None, 7, 7, 576)
    x = layers.GlobalAveragePooling2D(name="gap")(x)  # (None, 576)

    # Classification head
    x = layers.Dropout(0.3, name="dropout_1")(x)
    x = layers.Dense(256, activation="relu", name="dense_256")(x)
    x = layers.Dropout(0.2, name="dropout_2")(x)
    outputs = layers.Dense(
        num_classes, activation="softmax", name="predictions"
    )(x)

    model = Model(inputs, outputs, name=f"MobileNetV3Small_FER_{num_classes}cls")
    return model


def freeze_backbone(model: Model) -> None:
    """
    Đóng băng toàn bộ backbone — chỉ train classification head.
    Dùng ở stage 1 của mỗi phase (Freeze stage).
    """
    backbone = model.get_layer(BACKBONE_NAME)
    backbone.trainable = False
    print(f"[mobilenetv3] Backbone FROZEN — trainable params: "
          f"{count_trainable_params(model):,}")


def unfreeze_backbone(model: Model) -> None:
    """
    Mở khóa toàn bộ backbone — train cả backbone + head với lr nhỏ.
    Dùng ở stage 2 của mỗi phase (Unfreeze stage).
    """
    backbone = model.get_layer(BACKBONE_NAME)
    backbone.trainable = True
    print(f"[mobilenetv3] Backbone UNFROZEN — trainable params: "
          f"{count_trainable_params(model):,}")


def load_phase1_weights_for_phase2(model_p2: Model, p1_weights_path: str) -> None:
    """
    Load weights từ Phase 1 (8 class) vào model Phase 2 (6 class).

    Vì số lượng class khác nhau ở output layer, ta KHÔNG load nguyên model.
    Thay vào đó:
    1. Load model Phase 1 đầy đủ
    2. Copy weights backbone + dense head trung gian sang model Phase 2
    3. Bỏ qua layer output cuối (predictions) — giữ random init cho 6 class

    Args:
        model_p2        : model Phase 2 mới build (num_classes=6)
        p1_weights_path : đường dẫn file .h5 chứa weights Phase 1
    """
    # Build temporary model Phase 1 để load weights
    model_p1 = build_model(num_classes=8)
    model_p1.load_weights(p1_weights_path)
    print(f"[mobilenetv3] Loaded Phase 1 weights từ {p1_weights_path}")

    # Copy weights từng layer (trừ output predictions)
    layers_to_transfer = [BACKBONE_NAME, "dense_256"]
    for layer_name in layers_to_transfer:
        weights = model_p1.get_layer(layer_name).get_weights()
        model_p2.get_layer(layer_name).set_weights(weights)
        print(f"[mobilenetv3]   Copied weights: {layer_name}")

    # Layer 'predictions' giữ nguyên random init (output 6 class)
    print(f"[mobilenetv3] Layer 'predictions' (6 class) giữ random init.")


def count_trainable_params(model: Model) -> int:
    """Đếm số tham số trainable của model."""
    return sum(
        tf.size(w).numpy() for w in model.trainable_weights
    )


# ── Quick test ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Test Phase 1
    m1 = build_model(num_classes=8)
    print(f"Phase 1 model — Total params: {m1.count_params():,}")
    freeze_backbone(m1)
    unfreeze_backbone(m1)

    # Test Phase 2
    m2 = build_model(num_classes=6)
    print(f"Phase 2 model — Total params: {m2.count_params():,}")
