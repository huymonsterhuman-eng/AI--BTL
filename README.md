# BTL IS54A — Ứng dụng Computer Vision phân tích cảm xúc học sinh trong lớp học trực tuyến

> Bài tập lớn cuối kỳ — Môn Trí tuệ nhân tạo (IS54A), năm học 2025-2026

Hệ thống nhận diện trạng thái tương tác của học sinh trong môi trường học trực tuyến thông qua phân tích biểu cảm khuôn mặt, sử dụng Transfer Learning hai giai đoạn trên các kiến trúc deep learning (MobileNetV3, ResNet50, EfficientNet-B0, EfficientNet-B0 + Landmarks).

---

## 📊 Kết quả

| Mô hình | Phase 2 Accuracy | Macro F1 | Params |
|---|---|---|---|
| MobileNetV3Small | 93,00% | 0,92 | 2,5M |
| ResNet50 | 96,86% | 0,95 | 23,6M |
| **EfficientNet-B0** | **97,00%** | **0,96** | **5,3M** |
| **EfficientNet-B0 + Landmarks** | **96,86%** | **0,96** | 5,9M |

Model được lựa chọn cho demo: **EfficientNet-B0 + Landmarks** (cải thiện rõ trên các lớp khó: Bored, Frustrated).

---

## 🗂 Cấu trúc thư mục

```
ProjectBTL/
├── BTL/
│   ├── data/          # Pipeline tiền xử lý (Người 1)
│   ├── models/        # Định nghĩa kiến trúc 4 model
│   ├── notebooks/     # 5 Jupyter notebook training
│   ├── demo/          # Ứng dụng demo upload + webcam
│   ├── evaluation/    # Script tổng hợp kết quả
│   ├── weights/       # Model weights (KHÔNG push Git — tải từ Drive)
│   └── results/       # Metrics JSON + confusion matrix PNG
├── datadescription.md # Mô tả 2 dataset
├── work.md            # Phân công công việc
├── progress.md        # Nhật ký tiến độ
└── index.md           # Chỉ mục dự án
```

---

## 🚀 Cài đặt và chạy demo

### Yêu cầu hệ thống

- Python 3.12
- Windows / Linux / macOS
- Webcam (cho demo realtime)

### Cài đặt thư viện

```bash
pip install tensorflow opencv-python mtcnn mediapipe numpy matplotlib
```

### Tải weights (không có trong Git)

Tải file `efficientnet_lm_phase2.weights.h5` từ Google Drive của nhóm về:

```
BTL/weights/efficientnet_landmark/efficientnet_lm_phase2.weights.h5
```

### Chạy demo

**Demo upload ảnh:**
```bash
py -3.12 BTL/demo/demo_upload.py --image path/to/image.jpg --model landmarks
```

**Demo webcam realtime:**
```bash
py -3.12 BTL/demo/demo_webcam.py --model landmarks
```

Bấm `q` để thoát demo webcam.

**Tổng hợp kết quả 4 model:**
```bash
py -3.12 BTL/evaluation/evaluate.py
```

---

## 🧠 Kiến trúc tổng quát

### Transfer Learning hai giai đoạn

1. **Phase 1 — Pre-training trên AffectNet** (30.626 ảnh, 8 lớp cảm xúc): học đặc trưng khuôn mặt tổng quát
2. **Phase 2 — Fine-tuning trên Student Engagement Dataset** (2.120 ảnh, 6 lớp trạng thái): thích nghi domain học sinh

### 6 lớp đầu ra

| # | Nhãn | Nhóm |
|---|------|------|
| 0 | Focused | Engaged |
| 1 | Confused | Engaged |
| 2 | Frustrated | Engaged |
| 3 | Bored | Not Engaged |
| 4 | Drowsy | Not Engaged |
| 5 | Looking Away | Not Engaged |

---

## 👥 Nhóm thực hiện

| STT | Phụ trách |
|-----|-----------|
| 1 | Data & Preprocessing pipeline + Chương 1 & 2 |
| 2 | MobileNetV3Small |
| 3 | ResNet50 |
| 4 | EfficientNet-B0 + Khung Chương 3 |
| 5 | EfficientNet-B0 + Landmarks + Demo + Chương 4 |

---

## 📜 Tài liệu

- `Bao_cao_BTL_IS54A_Cam_xuc_hoc_sinh_v6.docx` — Báo cáo đầy đủ
- `datadescription.md` — Mô tả chi tiết 2 dataset
- `work.md` — Phân công công việc và tiến độ
