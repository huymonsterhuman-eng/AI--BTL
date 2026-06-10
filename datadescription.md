# Mô tả Dataset

## 1. AffectNet

**Nguồn gốc:** AffectNet là một trong những bộ dữ liệu nhận diện cảm xúc khuôn mặt lớn nhất hiện nay, được thu thập từ các ảnh trên Internet thông qua các từ khóa liên quan đến cảm xúc. Dataset này được sử dụng rộng rãi trong các nghiên cứu về nhận dạng biểu cảm khuôn mặt (Facial Expression Recognition - FER).

**Mục đích sử dụng trong dự án:** Huấn luyện mô hình nhận diện cảm xúc cơ bản (8 lớp cảm xúc), làm nền tảng để phân tích trạng thái học sinh.

### Cấu trúc thư mục

```
AffectNet/
├── Train/
│   ├── anger/
│   ├── contempt/
│   ├── disgust/
│   ├── fear/
│   ├── happy/
│   ├── neutral/
│   ├── sad/
│   └── surprise/
├── Test/
│   ├── Anger/
│   ├── Contempt/
│   ├── disgust/
│   ├── fear/
│   ├── happy/
│   ├── neutral/
│   ├── sad/
│   └── surprise/
└── labels.csv
```

### Thống kê dữ liệu

| Nhãn      | Train  | Test   | Tổng   |
|-----------|--------|--------|--------|
| Anger     | 1,500  | 1,718  | 3,218  |
| Contempt  | 1,559  | 1,312  | 2,871  |
| Disgust   | 1,229  | 1,248  | 2,477  |
| Fear      | 1,512  | 1,664  | 3,176  |
| Happy     | 2,340  | 2,704  | 5,044  |
| Neutral   | 2,758  | 2,368  | 5,126  |
| Sad       | 3,091  | 1,584  | 4,675  |
| Surprise  | 2,119  | 1,920  | 4,039  |
| **Tổng**  | **16,108** | **14,518** | **30,626** |

### Đặc điểm ảnh

- **Định dạng:** PNG
- **Kích thước:** 96 × 96 pixels
- **Màu sắc:** RGB
- **Nội dung:** Ảnh khuôn mặt người, được crop và căn chỉnh

### File nhãn (`labels.csv`)

File CSV đi kèm chứa các cột:
- `pth`: đường dẫn tương đối đến file ảnh
- `label`: nhãn cảm xúc thực tế
- `relFCs`: độ tin cậy của nhãn (confidence score)

---

## 2. Student Engagement Dataset

**Nguồn gốc:** Bộ dữ liệu ảnh khuôn mặt học sinh trong môi trường học tập, được gán nhãn theo mức độ tập trung và trạng thái cảm xúc khi học.

**Mục đích sử dụng trong dự án:** Phân loại trực tiếp mức độ tương tác của học sinh trong lớp học — phân biệt học sinh đang tập trung (Engaged) và không tập trung (Not Engaged) thông qua 6 trạng thái cụ thể.

### Cấu trúc thư mục

```
Student Dataset/
└── Student-engagement-dataset/
    ├── Engaged/
    │   ├── Confused/
    │   ├── Focused/
    │   └── Frustrated/
    └── Not Engaged/
        ├── Bored/
        ├── Drowsy/
        └── Looking Away/
```

### Thống kê dữ liệu

| Nhóm        | Trạng thái    | Số ảnh |
|-------------|---------------|--------|
| Engaged     | Confused      | 369    |
| Engaged     | Focused       | 347    |
| Engaged     | Frustrated    | 360    |
| Not Engaged | Bored         | 358    |
| Not Engaged | Drowsy        | 263    |
| Not Engaged | Looking Away  | 423    |
| **Tổng**    |               | **2,120** |

**Phân bố theo nhóm:**
- Engaged (tập trung): 1,076 ảnh (~50.8%)
- Not Engaged (không tập trung): 1,044 ảnh (~49.2%)

### Đặc điểm ảnh

- **Định dạng:** JPG
- **Kích thước:** 1280 × 720 pixels
- **Màu sắc:** RGB
- **Nội dung:** Ảnh học sinh chụp trong môi trường học tập thực tế

### Mô tả các nhãn

| Nhãn         | Nhóm        | Mô tả |
|--------------|-------------|-------|
| Focused      | Engaged     | Học sinh tập trung chú ý, nhìn thẳng vào bài |
| Confused     | Engaged     | Học sinh đang cố hiểu bài, thể hiện sự bối rối |
| Frustrated   | Engaged     | Học sinh đang cố gắng nhưng cảm thấy khó khăn |
| Bored        | Not Engaged | Học sinh không quan tâm, thờ ơ |
| Drowsy       | Not Engaged | Học sinh buồn ngủ, mất tỉnh táo |
| Looking Away | Not Engaged | Học sinh mất tập trung, nhìn chỗ khác |

---

## So sánh tổng quan

| Tiêu chí         | AffectNet               | Student Engagement Dataset     |
|------------------|-------------------------|--------------------------------|
| Tổng số ảnh      | 30,626                  | 2,120                          |
| Số lớp           | 8                       | 6 (2 nhóm chính)               |
| Kích thước ảnh   | 96 × 96 px              | 1280 × 720 px                  |
| Định dạng        | PNG                     | JPG                            |
| Domain           | Cảm xúc chung           | Trạng thái học tập             |
| Mục tiêu         | Nhận diện cảm xúc cơ bản | Đánh giá mức độ tập trung học sinh |
