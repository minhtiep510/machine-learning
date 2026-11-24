import os
import sys

# Thiết lập UTF-8 cho console Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
        sys.stdin.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Đảm bảo nhận diện đúng thư mục gốc project
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.train import train_and_compare_models
from src.predict import predict_upcoming_match


def main():
    while True:
        print("\n" + "=" * 60)
        print("   HỆ THỐNG DỰ ĐOÁN KẾT QUẢ BÓNG ĐÁ (ML & DEEP LEARNING)")
        print("=" * 60)
        print("1. Huấn luyện và so sánh 6 mô hình (Train & Benchmark)")
        print("2. Dự đoán kết quả một trận đấu cụ thể (Inference)")
        print("3. Thoát")
        print("=" * 60)

        choice = input("Nhập lựa chọn của bạn (1/2/3): ").strip()

        if choice == "1":
            print("\n[THÔNG BÁO] Bắt đầu quá trình huấn luyện và đánh giá 6 mô hình...")
            train_and_compare_models()
            print("\n[THÀNH CÔNG] Toàn bộ kết quả và biểu đồ đã được lưu tại 'results/'!")

        elif choice == "2":
            print("\n--- NHẬP THÔNG TIN TRẬN ĐẤU CẦN DỰ ĐOÁN ---")
            print("Gợi ý tên các đội bóng: Arsenal, Chelsea, Liverpool, Man City, Man United, Tottenham, Newcastle, Aston Villa, Everton...")
            home = input("Nhập tên Đội chủ nhà (HomeTeam): ").strip()
            away = input("Nhập tên Đội khách (AwayTeam): ").strip()

            if not home or not away:
                print("[LỖI] Tên đội bóng không được để trống!")
                continue

            print("\nChọn mô hình muốn dùng để dự đoán:")
            print("  [1] Ensemble: XGBoost + Neural Network (Khuyên dùng - 51.59%)")
            print("  [2] Logistic Regression (51.44%)")
            print("  [3] Random Forest (51.15%)")
            print("  [4] XGBoost (50.14%)")
            print("  [5] Neural Network (MLP)")
            print("  [6] Support Vector Machine (SVM)")
            m_choice = input("Chọn mô hình (1-6, bấm Enter mặc định là 1): ").strip()

            model_map = {
                "1": "ensemble_xgb_plus_mlp",
                "2": "logistic_regression",
                "3": "random_forest",
                "4": "xgboost",
                "5": "neural_network_mlp",
                "6": "svm_rbf_kernel",
                "": "ensemble_xgb_plus_mlp",
            }
            selected_model = model_map.get(m_choice, "ensemble_xgb_plus_mlp")

            predict_upcoming_match(home, away, selected_model)

        elif choice == "3":
            print("\nCảm ơn bạn đã sử dụng hệ thống. Tạm biệt!")
            break
        else:
            print("\n[LỖI] Lựa chọn không hợp lệ, vui lòng nhập 1, 2 hoặc 3!")


if __name__ == "__main__":
    main()