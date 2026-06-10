# Chỉ mục dự án — BTL IS54A

> Cập nhật: tất cả code và artifacts nằm trong folder `BTL/`.
> Status: ✅ = đã có | 📋 = README hướng dẫn, chờ Người phụ trách

---

## Cấu trúc thư mục

```
ProjectBTL/
│
├── 📁 AffectNet/                        # Dataset gốc (không push lên Git)
│   ├── Train/
│   │   ├── anger/        (1,500 ảnh)
│   │   ├── contempt/     (1,559 ảnh)
│   │   ├── disgust/      (1,229 ảnh)
│   │   ├── fear/         (1,512 ảnh)
│   │   ├── happy/        (2,340 ảnh)
│   │   ├── neutral/      (2,758 ảnh)
│   │   ├── sad/          (3,091 ảnh)
│   │   └── surprise/     (2,119 ảnh)
│   ├── Test/
│   │   └── [8 folder tương tự]          (14,518 ảnh)
│   └── labels.csv                       (pth, label, relFCs)
│
├── 📁 Student Dataset/                  # Dataset gốc (không push lên Git)
│   └── Student-engagement-dataset/
│       ├── Engaged/
│       │   ├── Confused/     (369 ảnh)
│       │   ├── Focused/      (347 ảnh)
│       │   └── Frustrated/   (360 ảnh)
│       └── Not Engaged/
│           ├── Bored/        (358 ảnh)
│           ├── Drowsy/       (263 ảnh)
│           └── Looking Away/ (423 ảnh)
│
├── 📁 BTL/                              # ⭐ Toàn bộ code dự án nằm trong đây
│   │
│   ├── 📁 data/                ✅ Pipeline xử lý dữ liệu (Người 1)
│   │   ├── preprocess.py                # Face detection, crop, resize, normalize
│   │   ├── augmentation.py              # augment_train / augment_val
│   │   ├── dataset_affectnet.py         # build_affectnet_dataset() + class_weights
│   │   ├── dataset_student.py           # build_student_dataset() từ CSV
│   │   ├── split_student.py             # Stratified 70/15/15 split
│   │   ├── student_train.csv            # ← tạo trên Colab
│   │   ├── student_val.csv              # ← tạo trên Colab
│   │   └── student_test.csv             # ← tạo trên Colab
│   │
│   ├── 📁 models/              📋 Định nghĩa kiến trúc (Người 2–5)
│   │   ├── README.md                    # Hướng dẫn quy ước
│   │   ├── mobilenetv3.py               # Người 2
│   │   ├── resnet50.py                  # Người 3
│   │   ├── efficientnet.py              # Người 4
│   │   └── efficientnet_landmarks.py    # Người 5
│   │
│   ├── 📁 training/            ❌ KHÔNG dùng — logic đã trong từng notebook
│   │   └── README.md                    # (placeholder, không có script .py)
│   │
│   ├── 📁 evaluation/          📋 Đánh giá (Người 5)
│   │   ├── README.md
│   │   ├── evaluate.py                  # Metrics, confusion matrix
│   │   └── visualize.py                 # Biểu đồ so sánh 4 model
│   │
│   ├── 📁 demo/                📋 Ứng dụng demo (Người 5)
│   │   ├── README.md
│   │   ├── demo_upload.py               # Upload ảnh predict
│   │   └── demo_webcam.py               # Webcam realtime
│   │
│   ├── 📁 weights/             📋 Model weights (không push Git)
│   │   ├── README.md
│   │   ├── mobilenetv3_phase1.h5        # Người 2
│   │   ├── mobilenetv3_phase2.h5        # Người 2
│   │   ├── resnet50_phase1.h5           # Người 3
│   │   ├── resnet50_phase2.h5           # Người 3
│   │   ├── efficientnet_phase1.h5       # Người 4
│   │   ├── efficientnet_phase2.h5       # Người 4
│   │   ├── efficientnet_lm_phase1.h5    # Người 5
│   │   └── efficientnet_lm_phase2.h5    # Người 5
│   │
│   ├── 📁 results/             📋 Kết quả thực nghiệm
│   │   ├── README.md
│   │   ├── mobilenetv3_metrics.json
│   │   ├── mobilenetv3_confusion_matrix.png
│   │   ├── resnet50_metrics.json
│   │   ├── resnet50_confusion_matrix.png
│   │   ├── efficientnet_metrics.json
│   │   ├── efficientnet_confusion_matrix.png
│   │   ├── efficientnet_lm_metrics.json
│   │   └── efficientnet_lm_confusion_matrix.png
│   │
│   └── 📁 notebooks/           ✅ Google Colab notebooks
│       ├── 01_data_pipeline.ipynb       # Người 1 — demo data pipeline
│       ├── 02_mobilenetv3.ipynb         # Người 2
│       ├── 03_resnet50.ipynb            # Người 3
│       ├── 04_efficientnet.ipynb        # Người 4
│       └── 05_efficientnet_landmarks.ipynb  # Người 5
│
├── 📄 Bao_cao_BTL_IS54A_..._v6.docx    # Báo cáo Word (cập nhật khi viết xong)
├── 📄 datadescription.md               # Mô tả chi tiết 2 dataset
├── 📄 work.md                          # Phân công công việc + deadline
├── 📄 progress.md                      # Nhật ký tiến độ
└── 📄 index.md                         # File này — chỉ mục dự án
```

---

## Trạng thái từng folder

| Folder | Trạng thái | Người phụ trách |
|--------|-----------|----------------|
| `BTL/data/` | ✅ Hoàn thành (5 file .py) | Người 1 |
| `BTL/notebooks/` | ✅ 1/5 notebook (data pipeline) | Người 1–5 |
| `BTL/models/` | 📋 Chỉ có README | Người 2–5 |
| `BTL/training/` | ❌ KHÔNG dùng (logic đã trong notebook) | — |
| `BTL/evaluation/` | 📋 Chỉ có README | Người 5 |
| `BTL/demo/` | 📋 Chỉ có README | Người 5 |
| `BTL/weights/` | 📋 Chỉ có README | Người 2–5 |
| `BTL/results/` | 📋 Chỉ có README | Người 2–5 |

---

## Tổng quan các file quan trọng

| File | Người tạo | Mục đích |
|------|-----------|---------|
| `BTL/data/preprocess.py` | Người 1 | Face detection, resize, normalize — dùng chung |
| `BTL/data/dataset_*.py` | Người 1 | DataLoader cho cả 2 dataset |
| `BTL/models/*.py` | Người 2–5 | Định nghĩa từng model |
| `BTL/training/config.py` | Người 4 | Hyperparameters chung cả nhóm |
| `BTL/training/train_phase1.py` | Người 4 | Script train Phase 1 |
| `BTL/training/train_phase2.py` | Người 4 | Script fine-tune Phase 2 |
| `BTL/evaluation/evaluate.py` | Người 5 | Tổng hợp metrics 4 model |
| `BTL/demo/demo_*.py` | Người 5 | Demo upload ảnh + webcam |
| `BTL/results/*.json` | Người 2–5 | Kết quả từng model |
| `datadescription.md` | — | Thống kê 2 dataset |
| `work.md` | — | Phân công + deadline |
| `progress.md` | Cả nhóm | Nhật ký cập nhật tiến độ |

---

## Lưu ý

- Thư mục `BTL/weights/`, `AffectNet/`, `Student Dataset/` **không đưa lên Git** (quá nặng)
- Tên file weights đặt đúng format `<model>_phase<1|2>.h5` để `evaluate.py` tự động load
- Folder `BTL/` được upload lên Google Drive chung — đường dẫn Colab: `/content/drive/MyDrive/ProjectBTL/BTL/`
