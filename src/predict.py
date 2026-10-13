import json
import os
import sys
import joblib
import numpy as np
import pandas as pd
from src.data_loader import load_and_merge_data
from src.features import build_feature_dataset

# Cấu hình UTF-8 cho console Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass


def predict_upcoming_match(
    home_team: str, away_team: str, model_name: str = "random_forest"
):
    """Trích xuất phong độ mới nhất của 2 đội và đưa ra dự đoán kết quả trận đấu sắp diễn ra."""
    # 1. Đọc dữ liệu lịch sử và tính toàn bộ features
    df = load_and_merge_data("data")
    all_teams = sorted(df["HomeTeam"].unique().tolist())

    if home_team not in all_teams:
        print(f"\n[LỖI] Đội chủ nhà '{home_team}' không tồn tại trong tập dữ liệu!")
        print(f"Danh sách các câu lạc bộ hợp lệ: {', '.join(all_teams)}")
        return

    if away_team not in all_teams:
        print(f"\n[LỖI] Đội khách '{away_team}' không tồn tại trong tập dữ liệu!")
        print(f"Danh sách các câu lạc bộ hợp lệ: {', '.join(all_teams)}")
        return

    df_feat, feature_cols = build_feature_dataset(df, window=5)

    # 2. Lấy vector đặc trưng mới nhất của HomeTeam và AwayTeam
    home_last_match = df_feat[df_feat["HomeTeam"] == home_team].iloc[-1]
    away_last_match = df_feat[df_feat["AwayTeam"] == away_team].iloc[-1]

    # Tạo vector đặc trưng dự đoán cho cặp đấu
    feat_vector = {}
    for col in feature_cols:
        if col.startswith("Home_"):
            feat_vector[col] = home_last_match[col]
        elif col.startswith("Away_"):
            feat_vector[col] = away_last_match[col]
        elif col.startswith("Diff_"):
            base = col.replace("Diff_", "")
            h_col = f"Home_{base}"
            a_col = f"Away_{base}"
            if h_col in feat_vector and a_col in feat_vector:
                feat_vector[col] = feat_vector[h_col] - feat_vector[a_col]
            else:
                feat_vector[col] = 0.0
        elif "H2H" in col:
            # Lấy H2H mới nhất giữa 2 đội
            h2h_matches = df_feat[
                ((df_feat["HomeTeam"] == home_team) & (df_feat["AwayTeam"] == away_team))
                | ((df_feat["HomeTeam"] == away_team) & (df_feat["AwayTeam"] == home_team))
            ]
            if len(h2h_matches) > 0:
                feat_vector[col] = h2h_matches.iloc[-1][col]
            else:
                feat_vector[col] = 1.3 if "Points" in col else 0.0
        elif "Odds_" in col:
            feat_vector[col] = 0.33  # Trung tính nếu không có kèo trước trận

    input_df = pd.DataFrame([feat_vector])[feature_cols]

    # 3. Load Model và Dự đoán
    model_path = f"models/{model_name}.joblib"
    if not os.path.exists(model_path):
        print(f"[LỖI] Chưa tìm thấy mô hình đã lưu tại '{model_path}'. Vui lòng chạy huấn luyện trước!")
        return

    model = joblib.load(model_path)

    probs = model.predict_proba(input_df)[0]
    pred_idx = np.argmax(probs)

    labels = [
        f"Đội chủ nhà thắng: {home_team} (Home Win - H)",
        "Hai đội hòa (Draw - D)",
        f"Đội khách thắng: {away_team} (Away Win - A)",
    ]

    print("\n" + "=" * 60)
    print(f"   KẾT QUẢ DỰ ĐOÁN: {home_team.upper()} vs {away_team.upper()}")
    print(f"   Mô hình tính toán: {model_name}")
    print("=" * 60)
    print(f" • Xác suất {home_team} thắng (Chủ nhà - H):  {probs[0]*100:.2f}%")
    print(f" • Xác suất kết quả Hòa (Hòa - D):          {probs[1]*100:.2f}%")
    print(f" • Xác suất {away_team} thắng (Đội khách - A): {probs[2]*100:.2f}%")
    print("-" * 60)
    print(f" => DỰ BÁO XÁC SUẤT CAO NHẤT: {labels[pred_idx]}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    h = sys.argv[1] if len(sys.argv) > 1 else "Arsenal"
    a = sys.argv[2] if len(sys.argv) > 2 else "Chelsea"
    m = sys.argv[3] if len(sys.argv) > 3 else "xgboost"
    predict_upcoming_match(h, a, m)
