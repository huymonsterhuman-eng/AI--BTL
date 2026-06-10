# Phân công công việc — Nhóm BTL IS54A

> Đề tài: Ứng dụng Computer Vision phân tích cảm xúc học sinh trong lớp học trực tuyến
> Môn: Trí tuệ nhân tạo (IS54A) — Năm học 2025-2026
> Báo cáo: 10/6/2026

---

## Thành viên

| STT | Họ tên | Phụ trách |
|-----|--------|-----------|
| 1 | *(điền tên)* | Data & Preprocessing |
| 2 | *(điền tên)* | MobileNetV3 |
| 3 | *(điền tên)* | ResNet50 |
| 4 | *(điền tên)* | EfficientNet-B0 + Khung Chương 3 |
| 5 | *(điền tên)* | EfficientNet-B0 + Landmarks + Demo |

---

## Phân công chi tiết

### Người 1 — Data & Preprocessing
**Deadline: 23/5 | Trạng thái: ✅ HOÀN THÀNH (22/5)**

**Code (`BTL/data/`):**
- [x] Tải và kiểm tra cấu trúc AffectNet + Student Engagement Dataset
- [x] Pipeline face detection, crop, resize về 224×224 (MTCNN)
- [x] Data augmentation (flip, rotation, color jitter, random crop)
- [x] Class weights cho AffectNet (xử lý imbalanced)
- [x] Shared DataLoader / `tf.data.Dataset` pipeline dùng chung cho 4 model
- [x] Script split Student Dataset: 70/15/15 stratified
- [x] Notebook demo `01_data_pipeline.ipynb` — đã verify trên Colab
- [x] Upload `BTL/` lên Google Drive chung

**Word:**
- [x] Chương 1: 1.1 Giới thiệu bài toán
- [x] Chương 1: 1.2 Bài toán AI
- [x] Chương 1: 1.3 Phạm vi thực hiện
- [x] Chương 1: 1.4 Khảo sát nghiên cứu liên quan (8 paper + sản phẩm thương mại)
- [x] Chương 2: 2.1 Dataset AffectNet (Pre-training)
- [x] Chương 2: 2.2 Dataset Student Engagement (Fine-tuning)
- [x] Chương 2: 2.3 Phân tích và phát hiện vấn đề dữ liệu
- [x] Chương 2: 2.4.1 Phát hiện và crop khuôn mặt
- [x] Chương 2: 2.4.2 Xử lý mất cân bằng nhãn
- [x] Chương 2: 2.4.3 Tăng cường dữ liệu
- [x] Chương 2: 2.4.4 Chuẩn hóa

---

### Người 2 — MobileNetV3
**Trạng thái: ✅ HOÀN THÀNH CODE (9/6) — Accuracy 93%, Macro F1 0.92**

**Code (`models/`, `notebooks/`):**
- [x] Xây dựng model MobileNetV3Small (pretrained ImageNet)
- [x] Phase 1: Train trên AffectNet (best val_acc 40.7%)
- [x] Phase 2: Fine-tune trên Student Dataset (val_acc 93%)
- [x] Lưu weights `mobilenetv3_phase1.h5` và `mobilenetv3_phase2.h5`
- [x] Xuất metrics, confusion matrix, training curves
- [x] Gửi số liệu cho Người 5

**Word (còn lại):**
- [ ] Mục 3.4.1: Mô tả MobileNetV3 (Depthwise Separable Conv, Hard-Swish)
- [ ] Phân tích confusion matrix → mục 4.2.2

---

### Người 3 — ResNet50
**Trạng thái: ✅ HOÀN THÀNH CODE (9/6) — Accuracy 96.86%, Macro F1 0.95**

**Code (`models/`, `notebooks/`):**
- [x] Xây dựng model ResNet50 (pretrained ImageNet)
- [x] Phase 1: Train trên AffectNet (best val_acc 66.15%)
- [x] Phase 2: Fine-tune trên Student Dataset (val_acc 96.86%)
- [x] Lưu weights `resnet50_phase1.weights.h5` và `resnet50_phase2.h5`
- [x] Xuất metrics, confusion matrix, training curves
- [x] Gửi số liệu cho Người 5

**Word (còn lại):**
- [ ] Mục 3.4.2: Mô tả ResNet50 (Residual/Skip Connection)
- [ ] Phân tích confusion matrix → mục 4.2.2

---

### Người 4 — EfficientNet-B0 + Khung Chương 3
**Trạng thái: ✅ HOÀN THÀNH CODE (9/6) — Accuracy 97%, Macro F1 0.96 🥇**

**Code (`models/`, `notebooks/`):**
- [x] Xây dựng model EfficientNet-B0 (pretrained ImageNet)
- [x] Phase 1: Train trên AffectNet (best val_acc 56.52%)
- [x] Phase 2: Fine-tune trên Student Dataset (val_acc 97%)
- [x] Lưu weights `efficientnet_phase1.weights.h5` và `efficientnet_phase2.weights.h5`
- [x] Xuất metrics, confusion matrix, training curves
- [x] Gửi số liệu cho Người 5

**Word (còn lại):**
- [ ] Mục 3.1: Đặc trưng dữ liệu và trích chọn đặc trưng
- [ ] Mục 3.2: Kiến trúc Pipeline (2 giai đoạn)
- [ ] Mục 3.3: Phân chia dữ liệu
- [ ] Mục 3.4.3: Mô tả EfficientNet-B0 (Compound Scaling)
- [ ] Mục 3.5: Cài đặt thực nghiệm
- [ ] Mục 3.7: Framework và công cụ

---

### Người 5 — EfficientNet-B0 + Landmarks + Demo + Chương 4
**Trạng thái: 🟡 ĐANG CHẠY (9/6) — Cell 4 pre-extract landmarks**

**Code (`models/`, `demo/`):**
- [x] Xây dựng model EfficientNet-B0 + MediaPipe Face Mesh Landmarks (đổi từ Dlib 68 → MediaPipe 478 để cài đặt dễ hơn trên Colab)
- [ ] Phase 1: Train trên AffectNet (đang chạy)
- [ ] Phase 2: Fine-tune trên Student Dataset
- [ ] Lưu weights `efficientnet_lm_phase1.weights.h5` và `efficientnet_lm_phase2.weights.h5`
- [ ] Demo upload ảnh (đã viết code `demo/demo_upload.py`)
- [ ] Demo webcam realtime (đã viết code `demo/demo_webcam.py`)

**Word (sau khi có kết quả):**
- [ ] Mục 3.4.4: Mô tả EfficientNet-B0 + Landmarks (sơ đồ fusion)
- [ ] Mục 4.1: Độ đo đánh giá
- [ ] Mục 4.2.1: Bảng so sánh 4 mô hình
- [ ] Mục 4.2.2: Phân tích Confusion Matrix
- [ ] Mục 4.2.3: Phân tích lỗi
- [ ] Mục 4.3: Công nghệ xây dựng demo
- [ ] Mục 4.4: Mô tả sản phẩm demo (ảnh chụp màn hình)
- [ ] Kết luận (đã làm được / hạn chế / hướng phát triển)
- [ ] Tài liệu tham khảo (tổng hợp toàn nhóm)

---

## 🏆 Tổng kết kết quả 4 model (cập nhật 9/6)

| Model | Phase 2 Accuracy | Macro F1 | Params | Vị trí |
|-------|------------------|----------|--------|--------|
| MobileNetV3Small | 93% | 0.92 | 2.5M | Nhẹ nhất |
| ResNet50 | 96.86% | 0.95 | 23.6M | Deep baseline |
| **EfficientNet-B0** | **97%** | **0.96** | **5.3M** | 🥇 **Đứng đầu** |
| EfficientNet+Landmarks | ⏳ | ⏳ | ~6M | Đang train |

---

## Timeline tổng hợp

| Mốc | Ngày | Nội dung |
|-----|------|---------|
| Data pipeline xong | 23/5 | Người 1 bàn giao DataLoader cho cả nhóm |
| Models + kết quả | 3/6 | Người 2, 3, 4 gửi số liệu cho Người 5 |
| Word Chương 3 | 4/6 | Người 4 hoàn thiện khung Chương 3 |
| Demo + Chương 4 | 6/6 | Người 5 hoàn thiện |
| Review toàn bộ | 6–8/6 | Cả nhóm kiểm tra, format, citation |
| Buffer | 8–9/6 | Sửa lỗi, in ấn |
| **Nộp báo cáo** | **10/6** | 🎯 |

---

## Quy ước chung

- **Weights**: lưu vào `weights/`, đặt tên theo format `<model>_phase<1|2>.h5`
- **Notebook**: mỗi người tạo file `notebooks/<tên_mình>.ipynb` trên Google Colab
- **Kết quả**: xuất ra `results/<model>_metrics.json` + ảnh confusion matrix PNG
- **Word**: chỉ chỉnh sửa phần mình phụ trách, không đụng phần người khác
- **Giao tiếp**: báo ngay nếu bị block (Colab hết quota, lỗi data, v.v.)
