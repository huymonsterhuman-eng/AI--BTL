# Progress Log — BTL IS54A

> Cập nhật tiến độ theo ngày. Mỗi người tự điền phần của mình.
> Format: `[YYYY-MM-DD] Người X — Nội dung đã làm / vấn đề gặp phải`

---

## Tuần 1 (18/5 – 23/5) — Data Pipeline ✅ HOÀN THÀNH

### Người 1 — Checklist code (`BTL/data/`) ✅
- [x] Tạo thư mục `BTL/data/` và `BTL/notebooks/`
- [x] `split_student.py` — stratified 70/15/15, xuất 3 CSV
- [x] `preprocess.py` — resize (AffectNet) + MTCNN detect/crop (Student)
- [x] `augmentation.py` — augment_train / augment_val bằng tf.image
- [x] `dataset_affectnet.py` — build_affectnet_dataset() + class_weights
- [x] `dataset_student.py` — build_student_dataset() từ CSV
- [x] `notebooks/01_data_pipeline.ipynb` — demo end-to-end, đã chạy thành công trên Colab
- [x] Chạy `split_student.py` thực tế trên Colab → verify 3 CSV
- [x] Test cả 2 pipeline trên Colab, chụp màn hình output
- [x] Upload toàn bộ `BTL/` lên Google Drive chung

### Người 1 — Checklist Word (Chương 1 & 2) ✅
- [x] Chương 1.1: bối cảnh học online + số liệu UNESCO + vấn đề mất tập trung
- [x] Chương 1.2: bài toán AI, pipeline 2 giai đoạn, input/output 6 lớp
- [x] Chương 1.3: phạm vi thực hiện (trong/ngoài scope, nền tảng)
- [x] Chương 1.4: khảo sát 4 nhóm nghiên cứu liên quan + sản phẩm thương mại
- [x] Chương 2.1: AffectNet — nguồn gốc, cấu trúc, thống kê, imbalance
- [x] Chương 2.2: Student Dataset — cấu trúc, 6 lớp, phân bố, đặc điểm ảnh
- [x] Chương 2.3: phân tích vấn đề dữ liệu (imbalance, nhãn sai, naming bug, Drowsy nhỏ)
- [x] Chương 2.4.1: phát hiện + crop khuôn mặt (MTCNN cho Student, chỉ resize cho AffectNet)
- [x] Chương 2.4.2: class weighting — công thức, bảng w_i, lý do không áp dụng cho Student
- [x] Chương 2.4.3: data augmentation — 4 nhóm phép biến đổi
- [x] Chương 2.4.4: chuẩn hóa ImageNet mean/std
- [x] Thêm citation [3]–[10] vào danh mục tài liệu tham khảo

### Nhật ký
- `[2026-05-22]` Người 1 — Hoàn thành toàn bộ code pipeline và nội dung Word Chương 1 & 2. Pipeline đã chạy thành công trên Colab. Bàn giao DataLoader cho nhóm.

---

## Tuần 2 (24/5 – 30/5) — Training Phase 1 & 2

*(chưa có cập nhật)*

---

## Tuần 3 (31/5 – 6/6) — Kết quả + Word + Demo

*(chưa có cập nhật)*

---

## Tuần 4 (7/6 – 10/6) — Review & Nộp

*(chưa có cập nhật)*

---

## Vấn đề đang mở (Blockers)

*(liệt kê nếu bị kẹt, cần cả nhóm hỗ trợ)*

---

## Kết quả thực nghiệm (điền khi có)

| Mô hình | Phase 1 Acc (AffectNet) | Phase 2 Acc (Student) | Macro F1 | Ghi chú |
|---------|------------------------|----------------------|----------|---------|
| MobileNetV3 | — | — | — | |
| ResNet50 | — | — | — | |
| EfficientNet-B0 | — | — | — | |
| EfficientNet-B0 + LM | — | — | — | |
