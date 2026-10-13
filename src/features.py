import numpy as np
import pandas as pd
import sys

# Thiết lập UTF-8 cho console Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass


def compute_team_rolling_features(df: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    """Trích xuất đặc trưng phong độ ngắn hạn (rolling window matches) và các chỉ số chuyên môn

    (sút, sút trúng đích, phạt góc, phạm lỗi, thẻ phạt, bàn thắng, bàn thua, điểm số)
    hoàn toàn dựa trên các trận đấu TRƯỚC trận hiện tại của từng đội.
    """
    df = df.copy()

    # Chuẩn bị bảng sự kiện theo góc nhìn của từng đội (Match perspective)
    home_matches = df[
        [
            "Date",
            "HomeTeam",
            "FTHG",
            "FTAG",
            "FTR",
            "HS",
            "HST",
            "HC",
            "HF",
            "HY",
            "HR",
            "AS",
            "AST",
            "AC",
            "AF",
            "AY",
            "AR",
        ]
    ].copy()
    home_matches.columns = [
        "Date",
        "Team",
        "GF",
        "GA",
        "Result",
        "Shots",
        "ShotsTarget",
        "Corners",
        "Fouls",
        "YellowCards",
        "RedCards",
        "OppShots",
        "OppShotsTarget",
        "OppCorners",
        "OppFouls",
        "OppYellowCards",
        "OppRedCards",
    ]
    home_matches["Points"] = home_matches["Result"].map(
        {"H": 3, "D": 1, "A": 0}
    )
    home_matches["IsHome"] = 1
    home_matches["MatchIndex"] = df.index

    away_matches = df[
        [
            "Date",
            "AwayTeam",
            "FTAG",
            "FTHG",
            "FTR",
            "AS",
            "AST",
            "AC",
            "AF",
            "AY",
            "AR",
            "HS",
            "HST",
            "HC",
            "HF",
            "HY",
            "HR",
        ]
    ].copy()
    away_matches.columns = [
        "Date",
        "Team",
        "GF",
        "GA",
        "Result",
        "Shots",
        "ShotsTarget",
        "Corners",
        "Fouls",
        "YellowCards",
        "RedCards",
        "OppShots",
        "OppShotsTarget",
        "OppCorners",
        "OppFouls",
        "OppYellowCards",
        "OppRedCards",
    ]
    away_matches["Points"] = away_matches["Result"].map(
        {"A": 3, "D": 1, "H": 0}
    )
    away_matches["IsHome"] = 0
    away_matches["MatchIndex"] = df.index

    # Gộp tất cả các trận
    all_team_matches = (
        pd.concat([home_matches, away_matches], ignore_index=True)
        .sort_values(by=["Date", "MatchIndex"])
        .reset_index(drop=True)
    )

    stat_cols = [
        "Points",
        "GF",
        "GA",
        "Shots",
        "ShotsTarget",
        "Corners",
        "Fouls",
        "YellowCards",
        "RedCards",
    ]

    # Tính rolling mean 5 trận gần nhất cho từng đội (shift(1) để chỉ dùng quá khứ)
    rolled_records = []
    for team, group in all_team_matches.groupby("Team"):
        group = group.copy()
        # Rolling overall
        for col in stat_cols:
            group[f"{col}_roll_{window}"] = (
                group[col].shift(1).rolling(window=window, min_periods=1).mean()
            )

        # Phong độ riêng khi đá sân nhà (cho đội chủ nhà) và sân khách (cho đội khách)
        home_sub = group[group["IsHome"] == 1].copy()
        away_sub = group[group["IsHome"] == 0].copy()

        home_sub[f"Points_venue_roll_{window}"] = (
            home_sub["Points"]
            .shift(1)
            .rolling(window=window, min_periods=1)
            .mean()
        )
        home_sub[f"GF_venue_roll_{window}"] = (
            home_sub["GF"].shift(1).rolling(window=window, min_periods=1).mean()
        )
        home_sub[f"GA_venue_roll_{window}"] = (
            home_sub["GA"].shift(1).rolling(window=window, min_periods=1).mean()
        )

        away_sub[f"Points_venue_roll_{window}"] = (
            away_sub["Points"]
            .shift(1)
            .rolling(window=window, min_periods=1)
            .mean()
        )
        away_sub[f"GF_venue_roll_{window}"] = (
            away_sub["GF"].shift(1).rolling(window=window, min_periods=1).mean()
        )
        away_sub[f"GA_venue_roll_{window}"] = (
            away_sub["GA"].shift(1).rolling(window=window, min_periods=1).mean()
        )

        group = group.merge(
            home_sub[
                [
                    "MatchIndex",
                    f"Points_venue_roll_{window}",
                    f"GF_venue_roll_{window}",
                    f"GA_venue_roll_{window}",
                ]
            ],
            on="MatchIndex",
            how="left",
        )
        group = group.merge(
            away_sub[
                [
                    "MatchIndex",
                    f"Points_venue_roll_{window}",
                    f"GF_venue_roll_{window}",
                    f"GA_venue_roll_{window}",
                ]
            ],
            on="MatchIndex",
            how="left",
            suffixes=("", "_away"),
        )

        # Hợp nhất venue roll (nếu trận đó là sân nhà thì lấy home, sân khách lấy away)
        group[f"Points_venue_roll_{window}"] = group[
            f"Points_venue_roll_{window}"
        ].combine_first(group[f"Points_venue_roll_{window}_away"])
        group[f"GF_venue_roll_{window}"] = group[
            f"GF_venue_roll_{window}"
        ].combine_first(group[f"GF_venue_roll_{window}_away"])
        group[f"GA_venue_roll_{window}"] = group[
            f"GA_venue_roll_{window}"
        ].combine_first(group[f"GA_venue_roll_{window}_away"])

        group = group.drop(
            columns=[
                f"Points_venue_roll_{window}_away",
                f"GF_venue_roll_{window}_away",
                f"GA_venue_roll_{window}_away",
            ]
        )
        rolled_records.append(group)

    rolled_df = pd.concat(rolled_records, ignore_index=True)

    # Tách ngược lại cho HomeTeam và AwayTeam
    home_feats = (
        rolled_df[rolled_df["IsHome"] == 1]
        .copy()
        .drop(
            columns=[
                "Date",
                "Team",
                "GF",
                "GA",
                "Result",
                "Shots",
                "ShotsTarget",
                "Corners",
                "Fouls",
                "YellowCards",
                "RedCards",
                "OppShots",
                "OppShotsTarget",
                "OppCorners",
                "OppFouls",
                "OppYellowCards",
                "OppRedCards",
                "Points",
                "IsHome",
            ]
        )
    )
    away_feats = (
        rolled_df[rolled_df["IsHome"] == 0]
        .copy()
        .drop(
            columns=[
                "Date",
                "Team",
                "GF",
                "GA",
                "Result",
                "Shots",
                "ShotsTarget",
                "Corners",
                "Fouls",
                "YellowCards",
                "RedCards",
                "OppShots",
                "OppShotsTarget",
                "OppCorners",
                "OppFouls",
                "OppYellowCards",
                "OppRedCards",
                "Points",
                "IsHome",
            ]
        )
    )

    home_feats.columns = [
        f"Home_{c}" if c != "MatchIndex" else c for c in home_feats.columns
    ]
    away_feats.columns = [
        f"Away_{c}" if c != "MatchIndex" else c for c in away_feats.columns
    ]

    df = df.merge(home_feats, left_index=True, right_on="MatchIndex", how="left")
    df = df.merge(away_feats, on="MatchIndex", how="left")
    df = df.drop(columns=["MatchIndex"])

    return df


def compute_h2h_features(df: pd.DataFrame, h2h_window: int = 5) -> pd.DataFrame:
    """Tính toán lịch sử đối đầu (Head-to-Head - H2H) giữa 2 đội trong các lần gặp gỡ trước đó:

    - Điểm số đối đầu trung bình của HomeTeam trước AwayTeam
    - Hiệu số bàn thắng đối đầu trung bình
    """
    df = df.copy()
    h2h_home_pts = []
    h2h_home_gd = []

    # Lưu lịch sử các trận đối đầu (cặp bất kể ai chủ nhà ai khách)
    pair_history = {}

    for idx, row in df.iterrows():
        h = row["HomeTeam"]
        a = row["AwayTeam"]
        pair_key = tuple(sorted([h, a]))

        if pair_key not in pair_history or len(pair_history[pair_key]) == 0:
            h2h_home_pts.append(1.3)  # giá trị trung bình giả định khi chưa gặp
            h2h_home_gd.append(0.0)
        else:
            # Lấy tối đa h2h_window trận trước đó
            past_matches = pair_history[pair_key][-h2h_window:]
            pts = []
            gds = []
            for prev in past_matches:
                # prev: (prev_home, prev_away, prev_fthg, prev_ftag)
                prev_h, prev_a, prev_hg, prev_ag = prev
                if prev_h == h and prev_a == a:
                    gd = prev_hg - prev_ag
                    pt = 3 if gd > 0 else (1 if gd == 0 else 0)
                else:  # prev_h == a and prev_a == h
                    gd = prev_ag - prev_hg
                    pt = 3 if gd > 0 else (1 if gd == 0 else 0)
                pts.append(pt)
                gds.append(gd)
            h2h_home_pts.append(np.mean(pts))
            h2h_home_gd.append(np.mean(gds))

        # Lưu lại trận hiện tại vào pair_history
        if pair_key not in pair_history:
            pair_history[pair_key] = []
        pair_history[pair_key].append(
            (row["HomeTeam"], row["AwayTeam"], row["FTHG"], row["FTAG"])
        )

    df["H2H_Home_Points_Avg"] = h2h_home_pts
    df["H2H_Home_GoalDiff_Avg"] = h2h_home_gd
    return df


def build_feature_dataset(
    df: pd.DataFrame, window: int = 5
) -> tuple[pd.DataFrame, list[str]]:
    """Tổng hợp pipeline trích xuất đặc trưng đầy đủ theo đúng yêu cầu:

    1. Rolling stats 5 trận gần nhất
    2. Phong độ sân nhà/sân khách
    3. Head-to-Head
    4. Các biến chênh lệch (Differentials)
    5. Đổi nhãn FTR -> (H: 0, D: 1, A: 2)
    """
    print(f"[THÔNG BÁO] Đang tính toán phong độ ngắn hạn (Rolling {window} trận gần nhất)...")
    df_feat = compute_team_rolling_features(df, window=window)

    print("[THÔNG BÁO] Đang tính toán lịch sử đối đầu (Head-to-Head)...")
    df_feat = compute_h2h_features(df_feat, h2h_window=window)

    # Tạo các đặc trưng chênh lệch (Home - Away)
    df_feat["Diff_Points_Roll"] = (
        df_feat[f"Home_Points_roll_{window}"]
        - df_feat[f"Away_Points_roll_{window}"]
    )
    df_feat["Diff_GF_Roll"] = (
        df_feat[f"Home_GF_roll_{window}"] - df_feat[f"Away_GF_roll_{window}"]
    )
    df_feat["Diff_GA_Roll"] = (
        df_feat[f"Home_GA_roll_{window}"] - df_feat[f"Away_GA_roll_{window}"]
    )
    df_feat["Diff_GoalDiff_Roll"] = (
        df_feat[f"Home_GF_roll_{window}"] - df_feat[f"Home_GA_roll_{window}"]
    ) - (df_feat[f"Away_GF_roll_{window}"] - df_feat[f"Away_GA_roll_{window}"])
    df_feat["Diff_Shots_Roll"] = (
        df_feat[f"Home_Shots_roll_{window}"]
        - df_feat[f"Away_Shots_roll_{window}"]
    )
    df_feat["Diff_ShotsTarget_Roll"] = (
        df_feat[f"Home_ShotsTarget_roll_{window}"]
        - df_feat[f"Away_ShotsTarget_roll_{window}"]
    )
    df_feat["Diff_Corners_Roll"] = (
        df_feat[f"Home_Corners_roll_{window}"]
        - df_feat[f"Away_Corners_roll_{window}"]
    )
    df_feat["Diff_Venue_Points_Roll"] = (
        df_feat[f"Home_Points_venue_roll_{window}"]
        - df_feat[f"Away_Points_venue_roll_{window}"]
    )

    # Tỷ lệ sút trúng đích
    df_feat["Home_ShotAccuracy_Roll"] = df_feat[
        f"Home_ShotsTarget_roll_{window}"
    ] / (df_feat[f"Home_Shots_roll_{window}"] + 1e-5)
    df_feat["Away_ShotAccuracy_Roll"] = df_feat[
        f"Away_ShotsTarget_roll_{window}"
    ] / (df_feat[f"Away_Shots_roll_{window}"] + 1e-5)
    df_feat["Diff_ShotAccuracy_Roll"] = (
        df_feat["Home_ShotAccuracy_Roll"] - df_feat["Away_ShotAccuracy_Roll"]
    )

    # Đưa tỷ lệ cược thị trường (Odds) vào nếu có làm đặc trưng tham chiếu (B365H, B365D, B365A)
    # Tỷ lệ cược là thông tin có sẵn TRƯỚC trận đấu phản ánh đánh giá của chuyên gia thị trường
    odds_cols = ["B365H", "B365D", "B365A"]
    has_odds = all(c in df_feat.columns for c in odds_cols)
    if has_odds:
        # Chuyển đổi thành xác suất ngụ ý (Implied Probability)
        prob_h = 1.0 / df_feat["B365H"]
        prob_d = 1.0 / df_feat["B365D"]
        prob_a = 1.0 / df_feat["B365A"]
        total_prob = prob_h + prob_d + prob_a
        df_feat["Odds_Prob_H"] = prob_h / total_prob
        df_feat["Odds_Prob_D"] = prob_d / total_prob
        df_feat["Odds_Prob_A"] = prob_a / total_prob

    # Xác định danh sách feature columns
    feature_cols = [
        # Rolling phong độ chung
        f"Home_Points_roll_{window}",
        f"Away_Points_roll_{window}",
        f"Home_GF_roll_{window}",
        f"Away_GF_roll_{window}",
        f"Home_GA_roll_{window}",
        f"Away_GA_roll_{window}",
        # Chỉ số chuyên môn trượt
        f"Home_Shots_roll_{window}",
        f"Away_Shots_roll_{window}",
        f"Home_ShotsTarget_roll_{window}",
        f"Away_ShotsTarget_roll_{window}",
        f"Home_Corners_roll_{window}",
        f"Away_Corners_roll_{window}",
        f"Home_Fouls_roll_{window}",
        f"Away_Fouls_roll_{window}",
        f"Home_YellowCards_roll_{window}",
        f"Away_YellowCards_roll_{window}",
        f"Home_RedCards_roll_{window}",
        f"Away_RedCards_roll_{window}",
        # Sân nhà / sân khách
        f"Home_Points_venue_roll_{window}",
        f"Away_Points_venue_roll_{window}",
        f"Home_GF_venue_roll_{window}",
        f"Away_GF_venue_roll_{window}",
        f"Home_GA_venue_roll_{window}",
        f"Away_GA_venue_roll_{window}",
        # Lịch sử đối đầu H2H
        "H2H_Home_Points_Avg",
        "H2H_Home_GoalDiff_Avg",
        # Chênh lệch (Differentials)
        "Diff_Points_Roll",
        "Diff_GF_Roll",
        "Diff_GA_Roll",
        "Diff_GoalDiff_Roll",
        "Diff_Shots_Roll",
        "Diff_ShotsTarget_Roll",
        "Diff_Corners_Roll",
        "Diff_Venue_Points_Roll",
        "Home_ShotAccuracy_Roll",
        "Away_ShotAccuracy_Roll",
        "Diff_ShotAccuracy_Roll",
    ]

    if has_odds:
        feature_cols.extend(["Odds_Prob_H", "Odds_Prob_D", "Odds_Prob_A"])

    # Mã hóa nhãn Target: H -> 0, D -> 1, A -> 2
    label_mapping = {"H": 0, "D": 1, "A": 2}
    df_feat["Target"] = df_feat["FTR"].map(label_mapping)

    # Điền giá trị thiếu (nếu có ở giai đoạn mở đầu mùa giải) bằng median
    df_feat[feature_cols] = df_feat[feature_cols].fillna(
        df_feat[feature_cols].median()
    )

    print(
        f"[THÔNG BÁO] Trích xuất thành công {len(feature_cols)} đặc trưng cho {len(df_feat)} trận đấu."
    )
    return df_feat, feature_cols
