# training/

Chứa script và config huấn luyện dùng chung cho cả 4 model.

| File | Người phụ trách | Nội dung |
|------|----------------|---------|
| `config.py` | Người 4 | Hyperparameters chung (lr, epochs, batch_size, ...) |
| `train_phase1.py` | Người 4 | Script train Phase 1 — AffectNet (8 class) |
| `train_phase2.py` | Người 4 | Script fine-tune Phase 2 — Student Dataset (6 class) |

## Cách dùng

```python
# Trong notebook của mỗi người:
import sys
sys.path.append('/content/drive/MyDrive/ProjectBTL/BTL/training')
from config import PHASE1_CONFIG, PHASE2_CONFIG

# PHASE1_CONFIG = {
#     'batch_size': 32,
#     'epochs_freeze': 5,
#     'epochs_unfreeze': 15,
#     'lr_freeze': 1e-3,
#     'lr_unfreeze': 1e-5,
#     'subset_per_class': 1500,
# }
```
