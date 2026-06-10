"""
efficientnet_landmarks.py
-------------------------
EfficientNet-B0 + Facial Landmarks (Fusion Architecture).

Đây là MODEL CẢI TIẾN — kết hợp:
- Deep features từ EfficientNet-B0 (1280-dim) — đặc trưng thị giác
- Geometric features từ MediaPipe Face Mesh (956-dim) — đặc trưng hình học

Kiến trúc:
    Image (224×224×3) ──→ Rescaling → EfficientNet-B0 → GAP → 1280-dim
                                                                  │
    Landmarks (956,)  ──→ Dense(128) → Dropout → Dense(64) → 64   │
                                                                  │
                                          Concatenate (1344-dim)
                                                  ↓
                                          Dropout(0.3)
                                                  ↓
                                          Dense(256, relu)
                                                  ↓
                                          Dropout(0.2)
                                                  ↓
                                          Dense(num_classes, softmax)

Landmarks: MediaPipe Face Mesh 478 điểm × 2 toạ độ (x, y) = 956-dim vector,
chuẩn hóa về [0, 1].

Hàm xuất:
    build_model(num_classes, input_shape, landmark_dim)
    freeze_backbone(model)
    unfreeze_backbone(model)
    load_phase1_weights_for_phase2(model_p2, p1_weights_path)
"""

import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import EfficientNetB0


BACKBONE_NAME = "efficientnetb0"
LANDMARK_DIM = 956  # 478 điểm × 2 (x, y) từ MediaPipe Face Mesh


def build_model(
    num_classes: int,
    input_shape: tuple = (224, 224, 3),
    landmark_dim: int = LANDMARK_DIM,
) -> Model:
    """
    Xây dựng EfficientNet-B0 + Landmarks Fusion model.

    Args:
        num_classes  : 8 cho Phase 1 (AffectNet) hoặc 6 cho Phase 2 (Student)
        input_shape  : kích thước ảnh đầu vào
        landmark_dim : số chiều vector landmark (default 956 = 478×2)

    Returns:
        tf.keras.Model — 2 input (image, landmarks), 1 output
    """
    # ── Image branch ────────────────────────────────────────────────────────
    image_input = layers.Input(shape=input_shape, name="image_input")

    # Denormalize → [0, 255] cho EfficientNet built-in preprocessing
    x_img = layers.Rescaling(
        scale=[0.229 * 255.0, 0.224 * 255.0, 0.225 * 255.0],
        offset=[0.485 * 255.0, 0.456 * 255.0, 0.406 * 255.0],
        name="denorm_to_0_255",
    )(image_input)

    backbone = EfficientNetB0(
        input_shape=input_shape, include_top=False, weights="imagenet"
    )
    x_img = backbone(x_img, training=False)
    x_img = layers.GlobalAveragePooling2D(name="gap")(x_img)  # (None, 1280)

    # ── Landmark branch ─────────────────────────────────────────────────────
    landmark_input = layers.Input(shape=(landmark_dim,), name="landmark_input")
    x_lm = layers.Dense(128, activation="relu", name="lm_dense_1")(landmark_input)
    x_lm = layers.Dropout(0.3, name="lm_dropout")(x_lm)
    x_lm = layers.Dense(64, activation="relu", name="lm_dense_2")(x_lm)  # (None, 64)

    # ── Fusion ──────────────────────────────────────────────────────────────
    x = layers.Concatenate(name="fusion")([x_img, x_lm])  # (None, 1344)
    x = layers.Dropout(0.3, name="dropout_1")(x)
    x = layers.Dense(256, activation="relu", name="dense_256")(x)
    x = layers.Dropout(0.2, name="dropout_2")(x)
    outputs = layers.Dense(
        num_classes, activation="softmax", name="predictions"
    )(x)

    return Model(
        inputs=[image_input, landmark_input],
        outputs=outputs,
        name=f"EfficientNetB0_Landmarks_FER_{num_classes}cls",
    )


def freeze_backbone(model: Model) -> None:
    """Đóng băng EfficientNet backbone."""
    backbone = model.get_layer(BACKBONE_NAME)
    backbone.trainable = False
    n = sum(tf.size(w).numpy() for w in model.trainable_weights)
    print(f"[efficientnet_lm] Backbone FROZEN — trainable params: {n:,}")


def unfreeze_backbone(model: Model) -> None:
    """Mở khóa EfficientNet backbone."""
    backbone = model.get_layer(BACKBONE_NAME)
    backbone.trainable = True
    n = sum(tf.size(w).numpy() for w in model.trainable_weights)
    print(f"[efficientnet_lm] Backbone UNFROZEN — trainable params: {n:,}")


def load_phase1_weights_for_phase2(model_p2: Model, p1_weights_path: str) -> None:
    """
    Transfer weights từ Phase 1 (8 class) sang Phase 2 (6 class).
    Copy: backbone + lm_dense_1 + lm_dense_2 + dense_256
    Bỏ qua: predictions (8 → 6 class)
    """
    model_p1 = build_model(num_classes=8)
    model_p1.load_weights(p1_weights_path)
    print(f"[efficientnet_lm] Loaded Phase 1 weights từ {p1_weights_path}")

    layers_to_transfer = [BACKBONE_NAME, "lm_dense_1", "lm_dense_2", "dense_256"]
    for layer_name in layers_to_transfer:
        weights = model_p1.get_layer(layer_name).get_weights()
        model_p2.get_layer(layer_name).set_weights(weights)
        print(f"[efficientnet_lm]   Copied weights: {layer_name}")

    print(f"[efficientnet_lm] Layer 'predictions' (6 class) giữ random init.")


# ── Quick test ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    m = build_model(num_classes=8)
    print(f"Total params: {m.count_params():,}")
    freeze_backbone(m)
    unfreeze_backbone(m)
    print("Inputs:", [t.name for t in m.inputs])
    print("Output:", m.output.name)
