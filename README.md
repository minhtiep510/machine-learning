# Football Match Result Prediction System (Machine Learning)

Hệ thống ứng dụng các thuật toán Học máy (Machine Learning) để dự đoán kết quả trận đấu bóng đá (Ngoại Hạng Anh) trước khi trận đấu diễn ra, dựa trên dữ liệu lịch sử thi đấu từ Football-Data.co.uk.

## 📌 Các tính năng chính
- **Data Pipeline:** Hợp nhất và chuẩn hóa dữ liệu 5 mùa giải gần nhất.
- **Feature Engineering (40 đặc trưng):**
  - Phong độ ngắn hạn (Rolling 5 trận gần nhất): điểm số, bàn thắng/thua, hiệu số.
  - Chỉ số chuyên môn trượt: Sút, sút trúng đích, phạt góc, phạm lỗi, thẻ phạt.
  - Thành tích thi đấu sân nhà / sân khách riêng biệt.
  - Lịch sử đối đầu trực tiếp (Head-to-Head - H2H).
  - Tỷ lệ xác suất ngụ ý từ tỷ lệ cược thị trường (Odds).
- **4 Mô hình Machine Learning trọng tâm:**
  - Logistic Regression
  - Random Forest Classifier
  - Support Vector Machine (SVM - RBF Kernel)
  - XGBoost Classifier
- **Đánh giá & Trực quan hóa:** Accuracy, F1-macro, Log-Loss, Ma trận nhầm lẫn (Confusion Matrix).
- **Module Dự đoán (Inference):** Cho phép nhập 2 đội bóng bất kỳ để dự báo xác suất kết quả.

## 🚀 Hướng dẫn cài đặt & Chạy

### 1. Cài đặt thư viện
```bash
pip install pandas numpy scikit-learn xgboost matplotlib seaborn joblib
```

### 2. Chạy ứng dụng
Mở terminal và chạy lệnh:
```bash
python main.py
```
- Chọn **`1`** để huấn luyện lại và so sánh hiệu năng 4 mô hình.
- Chọn **`2`** để dự đoán kết quả trận đấu giữa 2 câu lạc bộ cụ thể.
