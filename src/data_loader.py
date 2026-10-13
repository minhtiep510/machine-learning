import glob
import os
import sys
import pandas as pd

# Thiết lập UTF-8 cho console Windows để hiển thị tiếng Việt có dấu
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass


def load_and_merge_data(data_dir: str = "data") -> pd.DataFrame:
    """Đọc và hợp nhất tất cả các file CSV 5 mùa giải từ thư mục data/.

    Chuẩn hóa thời gian và sắp xếp theo trình tự diễn ra trận đấu.
    """
    csv_files = sorted(glob.glob(os.path.join(data_dir, "*.csv")))
    if not csv_files:
        raise FileNotFoundError(
            f"Không tìm thấy file csv nào trong thư mục '{data_dir}'"
        )

    dfs = []
    for f in csv_files:
        df = pd.read_csv(f).copy()
        # Ghi nhận mùa giải dựa trên tên file
        season_name = os.path.splitext(os.path.basename(f))[0]
        df = df.assign(Season=season_name)
        dfs.append(df)

    combined = pd.concat(dfs, ignore_index=True)

    # Loại bỏ hàng trống (nếu có cuối file)
    combined = combined.dropna(subset=["HomeTeam", "AwayTeam", "FTR"])

    # Chuẩn hóa cột ngày tháng
    combined["Date"] = pd.to_datetime(combined["Date"], format="%d/%m/%Y")

    # Sắp xếp theo ngày thi đấu (giữ tính nhân quả time-series, chống rò rỉ dữ liệu)
    combined = combined.sort_values(by=["Date"]).reset_index(drop=True)

    print(
        f"[THÔNG BÁO] Đã hợp nhất thành công {len(combined)} trận đấu từ {len(csv_files)} mùa giải."
    )
    return combined


if __name__ == "__main__":
    df = load_and_merge_data()
    print("Dữ liệu mẫu:")
    print(df[["Date", "HomeTeam", "AwayTeam", "FTR", "FTHG", "FTAG"]].head())
