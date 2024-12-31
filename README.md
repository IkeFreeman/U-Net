# U-Net Semantic Segmentation

Phần mềm sử dụng mô hình U-Net để thực hiện phân vùng ngữ nghĩa (semantic segmentation) trên ảnh.

## Tính năng

- Tải và tiền xử lý ảnh (hỗ trợ PNG, JPG)
- Huấn luyện mô hình U-Net
- Dự đoán và hiển thị kết quả phân vùng
- Giao diện người dùng thân thiện với Streamlit

## Cài đặt

1. Cài đặt các thư viện cần thiết:
```bash
pip install -r requirements.txt
```

2. Chạy ứng dụng:
```bash
streamlit run app.py
```

## Cấu trúc thư mục

```
project/
├── src/
│   ├── data/
│   │   └── data_loader.py
│   ├── model/
│   │   └── unet.py
│   └── inference/
│       └── predictor.py
├── app.py
├── requirements.txt
└── README.md
```

## Sử dụng

### Chế độ Inference
1. Chọn "Inference" trong sidebar
2. Tải lên ảnh cần phân vùng
3. Nhấn "Segment Image" để xem kết quả

### Chế độ Training
1. Chọn "Training" trong sidebar
2. Cấu hình các tham số huấn luyện
3. Tải lên dữ liệu huấn luyện (ảnh và mask)
4. Bắt đầu huấn luyện

## Yêu cầu hệ thống

- Python 3.7+
- CUDA (tùy chọn, để tăng tốc GPU)
