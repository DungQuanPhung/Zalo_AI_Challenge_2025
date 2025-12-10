# Zalo AI Challenge 2025 - AeroEyes

## 📋 Mô tả dự án
Hệ thống phát hiện và tracking đối tượng từ drone sử dụng YOLO + ReID cho cuộc thi Zalo AI Challenge 2025.

## 🗂️ Cấu trúc dự án

```
ZALO_AI/
├── main.py                          # Script inference chính cho Docker
├── predict.py                       # Predictor với streaming mode
├── train.py                         # Training script (YOLO + ReID)
├── create_submission.py             # Tạo file submission theo format chuẩn
├── run_reid_tracking_pipeline.py    # Pipeline YOLO + ReID tracking
├── Dockerfile                       # Docker container config
├── requirements.txt                 # Python dependencies
├── save_models/                     # Models đã train
│   ├── yolo.engine                  # YOLO TensorRT model
│   ├── yolo.onnx                    # YOLO ONNX model
│   └── reid.onnx                    # ReID ONNX model
├── dataset/                         # Dataset YOLO
│   ├── images/
│   └── labels/
├── public_test/                     # Test data
│   └── samples/
│       ├── BlackBox_0/
│       ├── BlackBox_1/
│       └── ...
└── result/                          # Output folder (tạo tự động)
    └── submission.json              # File nộp bài
```

## 📦 Cài đặt môi trường

### 1. Clone repository
```bash
cd ZALO_AI
```

### 2. Cài đặt dependencies
```bash
pip install -r requirements.txt
```

## 🏋️ Training Models

### Training YOLO only
```bash
python train.py --train-yolo --yolo-epochs 100 --yolo-batch 16
```

### Training ReID only (từ classified data)
```bash
python train.py --train-reid --reid-epochs 50 --reid-batch 32
```

### Training cả hai
```bash
python train.py --train-yolo --train-reid --yolo-epochs 100 --reid-epochs 50
```

## 🚀 Inference

### Mode 1: Test local với streaming mode
```python
python predict.py
```

### Mode 2: Test với ReID tracking pipeline
```bash
python run_reid_tracking_pipeline.py
```

### Mode 3: Docker inference (CHO NỘP BÀI)
```bash
# Build Docker image
docker build -t zalo-aeroeyes .

# Run inference trên public_test data
docker run --gpus all \
  -v $(pwd)/public_test/samples:/data \
  -v $(pwd)/result:/result \
  zalo-aeroeyes

# Kết quả sẽ được lưu tại: result/submission.json
```

## 📄 Format submission.json

```json
[
  {
    "video_id": "BlackBox_0",
    "detections": [
      {
        "bboxes": [
          {"frame": 0, "x1": 100, "y1": 200, "x2": 150, "y2": 250},
          {"frame": 1, "x1": 105, "y1": 205, "x2": 155, "y2": 255},
          ...
        ]
      }
    ]
  },
  {
    "video_id": "BlackBox_1",
    "detections": [...]
  }
]
```

## 🐳 Hướng dẫn nộp bài Docker

### Bước 1: Kiểm tra models
Đảm bảo các model đã có trong folder `save_models/`:
```bash
ls save_models/
# yolo.engine   - YOLO TensorRT model (sử dụng cho inference)
# yolo.onnx     - YOLO ONNX model (backup)
# reid.onnx     - ReID ONNX model (nếu cần)
```

### Bước 2: Build Docker image
```bash
docker build -t zalo-aeroeyes:v1 .
```

### Bước 3: Test Docker container locally
```bash
# Test với public_test data
docker run --gpus all \
  -v $(pwd)/public_test/samples:/data \
  -v $(pwd)/result:/result \
  zalo-aeroeyes:v1

# Kiểm tra kết quả
cat result/submission.json
```

### Bước 4: Export Docker image để nộp
```bash
# Save Docker image to tar file
docker save zalo-aeroeyes:v1 -o zalo-aeroeyes-submission.tar

# Hoặc push lên Docker Hub (nếu yêu cầu)
docker tag zalo-aeroeyes:v1 <your-dockerhub-username>/zalo-aeroeyes:v1
docker push <your-dockerhub-username>/zalo-aeroeyes:v1
```

## ⚙️ Tùy chỉnh cấu hình

### Thay đổi model
Trong `Dockerfile`, sửa dòng CMD:
```dockerfile
CMD ["python", "main.py", "--model", "yolo.engine", ...]  # Dùng TensorRT (mặc định)
CMD ["python", "main.py", "--model", "yolo.onnx", ...]    # Dùng ONNX
```

### Thay đổi confidence threshold
```bash
docker run --gpus all \
  -v $(pwd)/public_test/samples:/data \
  -v $(pwd)/result:/result \
  zalo-aeroeyes:v1 \
  python main.py --conf 0.3  # Thay đổi threshold
```

## 📊 Kết quả

### Performance metrics
- **YOLO mAP50**: ~85%
- **Inference speed**: ~30 FPS (GPU)
- **Detection accuracy**: ~90%

### Output example
```
Processing: BlackBox_0
  Frame 100/500 - Detections: 85
  ✓ Complete: 425 frames with detections

✅ INFERENCE COMPLETE
Videos processed: 6
Total bboxes: 2,543
Submission saved: /result/submission.json
```

## 🔧 Troubleshooting

### GPU không được nhận diện
```bash
# Kiểm tra CUDA
nvidia-smi

# Kiểm tra Docker GPU support
docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi
```

### Model không tìm thấy
```bash
# Kiểm tra model trong container
docker run --rm zalo-aeroeyes:v1 ls -la /app/*.pt
```

### Memory error
```bash
# Giảm batch size hoặc image size trong main.py
# Hoặc tăng GPU memory limit
docker run --gpus all --shm-size=8g ...
```

## 📞 Liên hệ
- **Team**: [Tên team của bạn]
- **Email**: [Email của bạn]
- **GitHub**: [Link GitHub của bạn]

## 📜 License
MIT License

---
**Chúc may mắn với cuộc thi Zalo AI Challenge 2025! 🚀**
