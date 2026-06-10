# demo/

Chứa ứng dụng demo. Người 5 phụ trách.

| File | Nội dung |
|------|---------|
| `demo_upload.py` | Upload ảnh → predict trạng thái engagement + confidence score |
| `demo_webcam.py` | Webcam realtime → bounding box + nhãn engagement |

## Yêu cầu

- Model weights Phase 2 đã được train xong (lấy từ `../weights/`)
- Dùng model EfficientNet-B0 hoặc EfficientNet-B0 + Landmarks (model tốt nhất)

## Cách chạy (Colab)

```python
# demo_upload.py
!python demo_upload.py --model_path /content/drive/.../weights/efficientnet_phase2.h5 \
                       --image_path /content/sample.jpg

# demo_webcam.py — chạy local (không chạy được trên Colab)
python demo_webcam.py --model_path weights/efficientnet_phase2.h5
```
