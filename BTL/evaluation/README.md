# evaluation/

Chứa script đánh giá và tổng hợp kết quả. Người 5 phụ trách.

| File | Nội dung |
|------|---------|
| `evaluate.py` | Load weights, tính accuracy / macro F1 / confusion matrix |
| `visualize.py` | Vẽ biểu đồ so sánh 4 model, accuracy/loss curves |

## Quy ước lưu kết quả

Mỗi model sau khi train xong lưu kết quả vào `../results/` theo format:

```
results/
├── mobilenetv3_metrics.json
├── resnet50_metrics.json
├── efficientnet_metrics.json
├── efficientnet_lm_metrics.json
├── mobilenetv3_confusion_matrix.png
├── resnet50_confusion_matrix.png
├── efficientnet_confusion_matrix.png
└── efficientnet_lm_confusion_matrix.png
```

## Format file JSON

```json
{
  "model": "MobileNetV3",
  "phase1_accuracy": 0.72,
  "phase2_accuracy": 0.85,
  "macro_f1": 0.83,
  "precision": 0.84,
  "recall": 0.83,
  "per_class_f1": {
    "Focused": 0.90,
    "Confused": 0.81,
    "Frustrated": 0.82,
    "Bored": 0.84,
    "Drowsy": 0.79,
    "Looking Away": 0.88
  }
}
```
