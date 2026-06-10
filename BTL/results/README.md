# results/

Chứa kết quả thực nghiệm sau khi train xong. Người 2–5 tự điền.

## Files cần có

```
results/
├── mobilenetv3_metrics.json          ← Người 2
├── mobilenetv3_confusion_matrix.png  ← Người 2
├── resnet50_metrics.json             ← Người 3
├── resnet50_confusion_matrix.png     ← Người 3
├── efficientnet_metrics.json         ← Người 4
├── efficientnet_confusion_matrix.png ← Người 4
├── efficientnet_lm_metrics.json      ← Người 5
└── efficientnet_lm_confusion_matrix.png ← Người 5
```

## Cách xuất kết quả (template)

```python
import json
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# ── Tính metrics ──────────────────────────────────────────────────
y_true, y_pred = [], []
for images, labels in test_student:
    preds = model.predict(images)
    y_true.extend(np.argmax(labels.numpy(), axis=1))
    y_pred.extend(np.argmax(preds, axis=1))

report = classification_report(y_true, y_pred, output_dict=True,
    target_names=['Focused','Confused','Frustrated','Bored','Drowsy','Looking Away'])

# ── Lưu JSON ──────────────────────────────────────────────────────
metrics = {
    "model": "MobileNetV3",          # THAY TÊN MODEL
    "phase2_accuracy": report["accuracy"],
    "macro_f1":        report["macro avg"]["f1-score"],
    "precision":       report["macro avg"]["precision"],
    "recall":          report["macro avg"]["recall"],
    "per_class_f1": {k: report[k]["f1-score"]
                     for k in ['Focused','Confused','Frustrated',
                                'Bored','Drowsy','Looking Away']}
}
RESULTS_DIR = '/content/drive/MyDrive/ProjectBTL/BTL/results'
with open(f'{RESULTS_DIR}/mobilenetv3_metrics.json', 'w') as f:  # THAY TÊN
    json.dump(metrics, f, indent=2)

# ── Lưu Confusion Matrix ──────────────────────────────────────────
cm = confusion_matrix(y_true, y_pred)
labels = ['Focused','Confused','Frustrated','Bored','Drowsy','Looking Away']

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=labels, yticklabels=labels)
plt.title('Confusion Matrix — MobileNetV3')   # THAY TÊN
plt.ylabel('Thực tế'); plt.xlabel('Dự đoán')
plt.tight_layout()
plt.savefig(f'{RESULTS_DIR}/mobilenetv3_confusion_matrix.png', dpi=150)  # THAY TÊN
plt.show()
```
