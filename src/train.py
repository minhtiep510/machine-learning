import json
import os
import sys

# Đảm bảo nhận diện đúng thư mục gốc dự án
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
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
    # 1. Tải và xử lý toàn bộ dữ liệu 10 mùa giải
    df = load_and_merge_data()
    df_feat, feature_cols = build_feature_dataset(df, window=5)

    X = df_feat[feature_cols].copy()
    y = df_feat["Target"].copy()

    # 2. Phân chia Train/Test theo chuỗi thời gian (Time-Series) chuẩn xác
    split_idx = int(len(df_feat) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    print(
        f"[THÔNG BÁO] Số mẫu huấn luyện (Train): {len(X_train)} trận (Đến ngày {df_feat.iloc[split_idx]['Date'].strftime('%d/%m/%Y')})"
    )
    print(
        f"[THÔNG BÁO] Số mẫu kiểm thử (Test):     {len(X_test)} trận (Các trận tiếp theo trong tương lai)"
    )

    # 3. Khởi tạo 5 mô hình đại diện độc lập: 4 Machine Learning + 1 Deep Learning (Mạng nơ-ron)
    models = {
        "Logistic Regression": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        max_iter=1000, C=0.05, random_state=42
                    ),
                ),
            ]
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=7,
            min_samples_split=6,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
        ),
        "Support Vector Machine (SVM)": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    SVC(kernel="rbf", C=1.0, probability=True, random_state=42),
                ),
            ]
        ),
        "XGBoost": xgb.XGBClassifier(
            n_estimators=100,
            max_depth=3,
            learning_rate=0.03,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.5,
            reg_lambda=1.0,
            objective="multi:softprob",
            eval_metric="mlogloss",
            random_state=42,
            n_jobs=-1,
        ),
        "Neural Network (MLP)": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "mlp",
                    MLPClassifier(
                        hidden_layer_sizes=(16, 8),
                        activation="relu",
                        alpha=0.5,
                        max_iter=600,
                        early_stopping=True,
                        n_iter_no_change=15,
                        random_state=42,
                    ),
                ),
            ]
        ),
    }

    results = []
    class_names = ["Chủ nhà (H)", "Hòa (D)", "Khách (A)"]

    os.makedirs(os.path.join(BASE_DIR, "models"), exist_ok=True)
    os.makedirs(os.path.join(BASE_DIR, "results"), exist_ok=True)

    # 4. Huấn luyện và đánh giá từng mô hình
    print("\n" + "=" * 68)
    print(
        f"{'MÔ HÌNH':<30} | {'ĐỘ CHÍNH XÁC':<13} | {'F1-MACRO':<10} | {'LOG-LOSS':<10}"
    )
    print("=" * 68)

    for name, model in models.items():
        model.fit(X_train, y_train)

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

        print(
            f"{name:<30} | {acc * 100:>10.2f}% | {f1_macro:>10.4f} | {loss:>10.4f}"
        )

        # Lưu model
        clean_name = (
            name.lower()
            .replace(" ", "_")
            .replace("(", "")
            .replace(")", "")
            .replace("[", "")
            .replace("]", "")
        )
        joblib.dump(
            model, os.path.join(BASE_DIR, "models", f"{clean_name}.joblib")
        )

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
        plt.savefig(
            os.path.join(
                BASE_DIR, "results", f"confusion_matrix_{clean_name}.png"
            ),
            dpi=200,
        )
        plt.close()

    print("=" * 68 + "\n")

    # Lưu kết quả bảng so sánh
    results_df = pd.DataFrame(results)
    csv_path = os.path.join(BASE_DIR, "results", "model_comparison.csv")
    results_df.to_csv(csv_path, index=False)
    print(f"[THÀNH CÔNG] Đã lưu bảng kết quả so sánh tại '{csv_path}'")

    # Lưu feature list
    with open(
        os.path.join(BASE_DIR, "models", "feature_columns.json"),
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(feature_cols, f, indent=4)

    # Vẽ biểu đồ Dashboard đôi so sánh đa chiều giữa 5 mô hình
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    models_names = results_df["Model"]
    x_pos = np.arange(len(models_names))
    width = 0.35

    # Subplot 1: Accuracy & F1-Score
    ax1.bar(
        x_pos - width / 2,
        results_df["Accuracy"] * 100,
        width,
        label="Độ chính xác - Accuracy (%)",
        color="#0284c7",
    )
    ax1.bar(
        x_pos + width / 2,
        results_df["F1_Macro"] * 100,
        width,
        label="F1-Macro (x100)",
        color="#10b981",
    )
    ax1.set_title(
        "So sánh Độ chính xác (Accuracy) & F1-Score",
        fontsize=12,
        fontweight="bold",
    )
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(models_names, rotation=18, ha="right", fontsize=9.5)
    ax1.set_ylabel("Tỷ lệ (%)", fontsize=10)
    ax1.set_ylim(0, 65)
    ax1.grid(axis="y", linestyle="--", alpha=0.6)
    ax1.legend(loc="upper left")

    for i, v in enumerate(results_df["Accuracy"]):
        ax1.text(
            i - width / 2,
            v * 100 + 0.7,
            f"{v * 100:.1f}%",
            ha="center",
            fontsize=9,
            fontweight="bold",
        )
    for i, v in enumerate(results_df["F1_Macro"]):
        ax1.text(
            i + width / 2,
            v * 100 + 0.7,
            f"{v * 100:.1f}%",
            ha="center",
            fontsize=9,
            fontweight="bold",
        )

    # Subplot 2: Log-Loss (Càng thấp càng tốt)
    min_loss = results_df["Log_Loss"].min()
    colors = [
        "#22c55e" if v == min_loss else "#0ea5e9"
        for v in results_df["Log_Loss"]
    ]
    bars = ax2.bar(x_pos, results_df["Log_Loss"], color=colors, width=0.55)
    ax2.set_title(
        "So sánh Log-Loss (Cột xanh lá là tối ưu nhất)",
        fontsize=12,
        fontweight="bold",
    )
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(models_names, rotation=18, ha="right", fontsize=9.5)
    ax2.set_ylabel("Log-Loss", fontsize=10)
    ax2.set_ylim(min_loss * 0.95, results_df["Log_Loss"].max() * 1.05)
    ax2.grid(axis="y", linestyle="--", alpha=0.6)

    for bar in bars:
        yval = bar.get_height()
        ax2.text(
            bar.get_x() + bar.get_width() / 2.0,
            yval + 0.002,
            f"{yval:.4f}",
            ha="center",
            fontsize=9,
            fontweight="bold",
        )

    plt.tight_layout()
    chart_path = os.path.join(
        BASE_DIR, "results", "model_benchmark_barchart.png"
    )
    plt.savefig(chart_path, dpi=200)
    plt.close()

    return results_df


if __name__ == "__main__":
    train_and_compare_models()
