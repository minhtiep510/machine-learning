import json
import os
import sys
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import xgboost as xgb

from src.data_loader import load_and_merge_data
from src.features import build_feature_dataset

# Cấu hình UTF-8 cho console Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass


def train_and_compare_models():
    # 1. Tải và xử lý dữ liệu
    df = load_and_merge_data("data")
    df_feat, feature_cols = build_feature_dataset(df, window=5)

    X = df_feat[feature_cols].copy()
    y = df_feat["Target"].copy()

    # 2. Phân chia Train/Test theo chuỗi thời gian (Time-Series)
    # 80% thời gian đầu dùng để huấn luyện, 20% gần nhất dùng để kiểm thử
    split_idx = int(len(df_feat) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    print(
        f"[THÔNG BÁO] Số mẫu huấn luyện (Train): {len(X_train)} trận (Đến ngày {df_feat.iloc[split_idx]['Date'].strftime('%d/%m/%Y')})"
    )
    print(
        f"[THÔNG BÁO] Số mẫu kiểm thử (Test):     {len(X_test)} trận (Các trận tiếp theo trong tương lai)"
    )

    # 3. Khởi tạo 4 mô hình trọng tâm
    # Target: 0: Chủ nhà thắng (H), 1: Hòa (D), 2: Đội khách thắng (A)
    models = {
        "Logistic Regression": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        max_iter=1000,
                        random_state=42,
                    ),
                ),
            ]
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=6,
            min_samples_split=5,
            random_state=42,
            n_jobs=-1,
        ),
        "SVM (RBF Kernel)": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", SVC(kernel="rbf", C=1.0, probability=True, random_state=42)),
            ]
        ),
        "XGBoost": xgb.XGBClassifier(
            n_estimators=150,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="multi:softprob",
            eval_metric="mlogloss",
            random_state=42,
            n_jobs=-1,
        ),
    }

    results = []
    class_names = ["Chủ nhà (H)", "Hòa (D)", "Khách (A)"]

    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    # 4. Huấn luyện và đánh giá từng mô hình
    print("\n" + "=" * 68)
    print(f"{'MÔ HÌNH':<24} | {'ĐỘ CHÍNH XÁC':<13} | {'F1-MACRO':<10} | {'LOG-LOSS':<10}")
    print("=" * 68)

    for name, model in models.items():
        # Fit
        model.fit(X_train, y_train)

        # Predict
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)

        acc = accuracy_score(y_test, y_pred)
        f1_macro = f1_score(y_test, y_pred, average="macro")
        loss = log_loss(y_test, y_proba)

        results.append(
            {
                "Model": name,
                "Accuracy": round(acc, 4),
                "F1_Macro": round(f1_macro, 4),
                "Log_Loss": round(loss, 4),
            }
        )

        print(f"{name:<24} | {acc * 100:>10.2f}% | {f1_macro:>10.4f} | {loss:>10.4f}")

        # Lưu model
        clean_name = name.lower().replace(" ", "_").replace("(", "").replace(")", "")
        joblib.dump(model, f"models/{clean_name}.joblib")

        # Vẽ Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(5.5, 4.5))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=class_names,
            yticklabels=class_names,
        )
        plt.title(f"Ma trận nhầm lẫn - {name}")
        plt.ylabel("Thực tế")
        plt.xlabel("Dự đoán")
        plt.tight_layout()
        plt.savefig(f"results/confusion_matrix_{clean_name}.png", dpi=200)
        plt.close()

    print("=" * 68 + "\n")

    # Lưu kết quả bảng so sánh
    results_df = pd.DataFrame(results)
    results_df.to_csv("results/model_comparison.csv", index=False)
    print("[THÀNH CÔNG] Đã lưu bảng kết quả so sánh tại 'results/model_comparison.csv'")

    # Lưu feature list
    with open("models/feature_columns.json", "w", encoding="utf-8") as f:
        json.dump(feature_cols, f, indent=4)

    # Vẽ biểu đồ so sánh Accuracy & F1-Score
    plt.figure(figsize=(9, 5))
    x_pos = np.arange(len(results_df))
    width = 0.35

    plt.bar(x_pos - width / 2, results_df["Accuracy"], width, label="Độ chính xác (Accuracy)", color="#1f77b4")
    plt.bar(x_pos + width / 2, results_df["F1_Macro"], width, label="F1-Macro", color="#2ca02c")

    plt.xlabel("Mô hình Machine Learning")
    plt.ylabel("Điểm đánh giá")
    plt.title("So sánh hiệu năng 4 mô hình Machine Learning")
    plt.xticks(x_pos, results_df["Model"])
    plt.ylim(0, 1.0)
    plt.legend()
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig("results/model_benchmark_barchart.png", dpi=200)
    plt.close()

    return results_df


if __name__ == "__main__":
    train_and_compare_models()
