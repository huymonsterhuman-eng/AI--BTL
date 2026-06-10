# weights/

Chứa model weights đã train. **KHÔNG đưa lên Git** (file quá nặng).

## Quy ước đặt tên

```
<tên_model>_phase<1|2>.h5
```

| File | Phase | Người tạo |
|------|-------|----------|
| `mobilenetv3_phase1.h5` | Pre-train AffectNet (8 class) | Người 2 |
| `mobilenetv3_phase2.h5` | Fine-tune Student (6 class) | Người 2 |
| `resnet50_phase1.h5` | Pre-train AffectNet (8 class) | Người 3 |
| `resnet50_phase2.h5` | Fine-tune Student (6 class) | Người 3 |
| `efficientnet_phase1.h5` | Pre-train AffectNet (8 class) | Người 4 |
| `efficientnet_phase2.h5` | Fine-tune Student (6 class) | Người 4 |
| `efficientnet_lm_phase1.h5` | Pre-train AffectNet (8 class) | Người 5 |
| `efficientnet_lm_phase2.h5` | Fine-tune Student (6 class) | Người 5 |

## Cách lưu weights (trong notebook)

```python
# Lưu model tốt nhất sau training
model.save('/content/drive/MyDrive/ProjectBTL/BTL/weights/mobilenetv3_phase2.h5')

# Hoặc dùng ModelCheckpoint callback
checkpoint = tf.keras.callbacks.ModelCheckpoint(
    filepath='/content/drive/MyDrive/ProjectBTL/BTL/weights/mobilenetv3_phase2.h5',
    monitor='val_accuracy',
    save_best_only=True,
    verbose=1
)
```
