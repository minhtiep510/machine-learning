# Football Match Result Prediction System (ML & Deep Learning)

Hệ thống ứng dụng các thuật toán Học máy (Machine Learning) và Mạng nơ-ron nhân tạo (Neural Network) để dự đoán kết quả trận đấu bóng đá (Ngoại Hạng Anh) trước khi trận đấu diễn ra, dựa trên dữ liệu lịch sử thi đấu từ Football-Data.co.uk.

## 📌 Các tính năng chính
- **Data Pipeline:** Hợp nhất và chuẩn hóa dữ liệu 5 mùa giải gần nhất (phân chia Train/Test theo trình tự thời gian 80/20 để tránh rò rỉ dữ liệu tương lai).
- **Feature Engineering (40 đặc trưng):**
  - Phong độ ngắn hạn (Rolling 5 trận gần nhất): điểm số, bàn thắng/thua, hiệu số.
  - Chỉ số chuyên môn trượt: Sút, sút trúng đích, phạt góc, phạm lỗi, thẻ phạt.
  - Thành tích thi đấu sân nhà / sân khách riêng biệt.
  - Lịch sử đối đầu trực tiếp (Head-to-Head - H2H).
  - Tỷ lệ xác suất ngụ ý từ tỷ lệ cược thị trường (Odds).
- **6 Mô hình Huấn luyện & Đối chứng:**
  - Logistic Regression
  - Random Forest Classifier
  - Support Vector Machine (SVM - RBF Kernel)
  - XGBoost Classifier
  - Neural Network (Multi-Layer Perceptron - MLP)
  - **Weighted Ensemble (XGBoost + Neural Network):** Mô hình kết hợp xác suất (Soft Voting) giữa cây quyết định và mạng nơ-ron.
- **Đánh giá & Trực quan hóa:** Accuracy, F1-macro, Log-Loss, Ma trận nhầm lẫn (Confusion Matrix) và biểu đồ so sánh hiệu năng.
- **Module Dự đoán (Inference):** Cho phép nhập 2 đội bóng bất kỳ để dự báo xác suất kết quả (Chủ nhà thắng / Hòa / Khách thắng).

## 📂 Cấu trúc thư mục
```text
├── data/               # Dữ liệu lịch sử các mùa giải Ngoại Hạng Anh (.csv)
├── models/             # Lưu trữ các mô hình đã huấn luyện (.joblib) và danh sách đặc trưng
├── results/            # Bảng tổng hợp kết quả (.csv) và các biểu đồ đánh giá (.png)
├── src/
│   ├── data_loader.py  # Đọc và hợp nhất dữ liệu thô
│   ├── features.py     # Trích xuất 40 đặc trưng chuyên môn
│   ├── train.py        # Huấn luyện, đánh giá và lưu 6 mô hình
│   └── predict.py      # Dự đoán xác suất trận đấu mới
└── main.py             # Giao diện dòng lệnh (CLI) điều khiển hệ thống