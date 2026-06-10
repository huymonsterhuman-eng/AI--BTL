# models/

Chứa định nghĩa kiến trúc từng model. Mỗi người viết 1 file riêng.

| File | Người phụ trách | Nội dung |
|------|----------------|---------|
| `mobilenetv3.py` | Người 2 | MobileNetV3Small + classification head |
| `resnet50.py` | Người 3 | ResNet50 + classification head |
| `efficientnet.py` | Người 4 | EfficientNet-B0 + classification head |
| `efficientnet_landmarks.py` | Người 5 | EfficientNet-B0 + Dlib Landmarks fusion |

## Cấu trúc mỗi file

Mỗi file cần export 1 hàm `build_model(num_classes, input_shape=(224,224,3))` trả về `tf.keras.Model`.

```python
# Ví dụ
def build_model(num_classes=8, input_shape=(224, 224, 3)):
    base = tf.keras.applications.MobileNetV3Small(
        input_shape=input_shape, include_top=False, weights='imagenet'
    )
    ...
    return model
```
