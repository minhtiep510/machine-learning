import json
import os
import sys
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.base import BaseEstimator, ClassifierMixin
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


# Wrapper tương thích Scikit-Learn để gộp xác suất (Soft Voting) giữa XGBoost và Neural Network
class WeightedEnsembleClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, models_weights):
        """
        models_weights: list các tuple (tên_mô_hình, đối_tượng_mô_hình, trọng_số)
        """
        self.models_weights = models_weights

    def fit(self, X, y):
        # Các mô hình con đã được huấn luyện sẵn trong vòng lặp chính nên không cần fit lại
        self.classes_ = np.unique(y)
        return self

    def predict_proba(self, X):
        total_proba = None
        for _, model, weight in self.models_weights:
            proba = model.predict_proba(X)
            if total_proba is None:
                total_proba = weight * proba
            else:
                total_proba += weight * proba
        return total_proba

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)


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

    # 3. Khởi tạo 5 mô hình cơ sở (thêm Neural Network MLP)
    # Target: 0: Chủ nhà thắng (H), 1: Hòa (D), 2: Đội khách thắng (A)
    mlp_pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "clf",
                MLPClassifier(
                    hidden_layer_sizes=(64, 32),  # 2 lớp ẩn gọn nhẹ chống overfitting
                    activation="relu",
                    solver="adam",
                    alpha=0.1,  # L2 regularization mạnh để giảm nhiễu dữ liệu bóng đá
                    batch_size=32,
                    learning_rate_init=0.001,
                    max_iter=300,
                    early_stopping=True,  # Tự dừng nếu validation loss không giảm
                    validation_fraction=0.15,
                    n_iter_no_change=15,
                    random_state=42,
                ),
            ),
        ]
    )

    xgb_model = xgb.XGBClassifier(
        n_estimators=150,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="multi:softprob",
        eval_metric="mlogloss",
        random_state=42,
        n_jobs=-1,
    )

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
        "XGBoost": xgb_model,
        "Neural Network (MLP)": mlp_pipeline,
        # Mô hình thứ 6: Gộp xác suất 70% XGBoost + 30% Neural Network
        "Ensemble (XGB + MLP)": WeightedEnsembleClassifier(
            models_weights=[
                ("XGBoost", xgb_model, 0.7),
                ("Neural Network", mlp_pipeline, 0.3),
            ]
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
        # Fit (Với Ensemble, xgb_model và mlp_pipeline đã được fit ở các vòng lặp trước đó)
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
        clean_name = (
            name.lower()
            .replace(" ", "_")
            .replace("(", "")
            .replace(")", "")
            .replace("+", "plus")
        )
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

    # Vẽ biểu đồ Dashboard so sánh đa chiều (Accuracy, F1-Score & Log-Loss)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    models_names = results_df["Model"]
    x_pos = np.arange(len(models_names))
    width = 0.35

    # Subplot 1: Accuracy & F1-Score
    ax1.bar(x_pos - width / 2, results_df["Accuracy"] * 100, width, label="Độ chính xác - Accuracy (%)", color="#0284c7")
    ax1.bar(x_pos + width / 2, results_df["F1_Macro"] * 100, width, label="F1-Macro (x100)", color="#10b981")
    ax1.set_title("So sánh Accuracy & F1-Score", fontsize=12, fontweight="bold")
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(models_names, rotation=22, ha="right", fontsize=9)
    ax1.set_ylabel("Tỷ lệ (%)", fontsize=10)
    ax1.set_ylim(0, 60)
    ax1.grid(axis="y", linestyle="--", alpha=0.6)
    ax1.legend(loc="upper left")

    for i, v in enumerate(results_df["Accuracy"]):
        ax1.text(i - width / 2, v * 100 + 0.7, f"{v * 100:.1f}%", ha="center", fontsize=8.5, fontweight="bold")
    for i, v in enumerate(results_df["F1_Macro"]):
        ax1.text(i + width / 2, v * 100 + 0.7, f"{v * 100:.1f}%", ha="center", fontsize=8.5, fontweight="bold")

    # Subplot 2: Log-Loss (Càng thấp càng tốt)
    min_loss = results_df["Log_Loss"].min()
    colors = ["#22c55e" if v == min_loss else "#0ea5e9" for v in results_df["Log_Loss"]]
    bars = ax2.bar(x_pos, results_df["Log_Loss"], color=colors, width=0.55)
    ax2.set_title("So sánh Log-Loss (Cột xanh lá là tối ưu nhất)", fontsize=12, fontweight="bold")
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(models_names, rotation=22, ha="right", fontsize=9)
    ax2.set_ylabel("Log-Loss", fontsize=10)
    ax2.set_ylim(min_loss * 0.9, results_df["Log_Loss"].max() * 1.05)
    ax2.grid(axis="y", linestyle="--", alpha=0.6)

    for bar in bars:
        yval = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2.0, yval + 0.003, f"{yval:.4f}", ha="center", fontsize=8.5, fontweight="bold")

    plt.tight_layout()
    plt.savefig("results/model_benchmark_barchart.png", dpi=200)
    plt.close()

    return results_df


if __name__ == "__main__":
    train_and_compare_models()