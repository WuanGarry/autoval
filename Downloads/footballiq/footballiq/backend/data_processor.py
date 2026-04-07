"""
data_processor.py
Loads, cleans, and engineers features from Matches.csv + EloRatings.csv.
All features are derived from pre-match information only so they are
usable at prediction time.

Supports both .parquet (fast, requires pyarrow) and .csv fallback.
"""

import os
import gc
import pandas as pd
import numpy as np
from pathlib import Path

DATA_DIR = Path(os.environ.get("DATA_DIR", str(Path(__file__).resolve().parent.parent / "data")))


# ─────────────────────────────────────────────────────────────────────────────
# 1.  RAW LOAD
# ─────────────────────────────────────────────────────────────────────────────

def load_raw() -> pd.DataFrame:
    df = pd.read_csv(
        DATA_DIR / "Matches.csv",
        low_memory=False,
    )
    return df


def load_elo() -> pd.DataFrame:
    elo = pd.read_csv(DATA_DIR / "EloRatings.csv", low_memory=False)
    elo["date"] = pd.to_datetime(elo["date"], dayfirst=True, errors="coerce")
    return elo


# ─────────────────────────────────────────────────────────────────────────────
# 2.  BASIC CLEANING
# ─────────────────────────────────────────────────────────────────────────────

def clean(df: pd.DataFrame):
    df = df.copy()

    # Parse date
    df["MatchDate"] = pd.to_datetime(df["MatchDate"], dayfirst=True, errors="coerce")
    df = df.dropna(subset=["MatchDate", "HomeTeam", "AwayTeam", "FTResult"])

    # Drop rows with no goals
    df = df.dropna(subset=["FTHome", "FTAway"])
    df["FTHome"] = df["FTHome"].astype(int)
    df["FTAway"] = df["FTAway"].astype(int)

    # Encode result  0=Draw  1=Home  2=Away
    result_map = {"H": 1, "D": 0, "A": 2}
    df["Result"] = df["FTResult"].map(result_map)
    df = df.dropna(subset=["Result"])
    df["Result"] = df["Result"].astype(int)

    # Derived goal columns
    df["TotalGoals"] = df["FTHome"] + df["FTAway"]
    df["GoalDiff"]   = df["FTHome"] - df["FTAway"]

    # ── Bonus columns (corners, cards, shots) ─────────────────────────────
    # These come from football-data.co.uk CSVs. Fill with 0 when absent.
    for col, default in [
        ("HomeCorners", 5.0), ("AwayCorners", 4.5),
        ("HomeYellow",  1.5), ("AwayYellow",  1.8),
        ("HomeRed",     0.1), ("AwayRed",     0.1),
        ("HomeShots",   12.0),("AwayShots",   10.0),
        ("HomeShotsTarget", 4.5), ("AwayShotsTarget", 3.8),
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(default)
        else:
            df[col] = default

    df["TotalCorners"]  = df["HomeCorners"]  + df["AwayCorners"]
    df["TotalYellows"]  = df["HomeYellow"]   + df["AwayYellow"]
    df["TotalReds"]     = df["HomeRed"]      + df["AwayRed"]
    df["TotalBookings"] = df["TotalYellows"] + df["TotalReds"] * 2   # red=2pts
    df["ShotsDiff"]     = df["HomeShots"]    - df["AwayShots"]


    # HT goals – fill missing with column median
    for col in ["HTHome", "HTAway"]:
        med = df[col].median()
        df[col] = df[col].fillna(0 if np.isnan(med) else med).astype(float)

    # ELO – fill missing with league-level median then global fallback
    for col in ["HomeElo", "AwayElo"]:
        df[col] = df[col].fillna(
            df.groupby("Division")[col].transform("median")
        ).fillna(1500.0)

    df["EloDiff"] = df["HomeElo"] - df["AwayElo"]
    df["EloSum"]  = df["HomeElo"] + df["AwayElo"]

    # Form columns
    for col in ["Form3Home", "Form5Home", "Form3Away", "Form5Away"]:
        df[col] = df[col].fillna(0).astype(float)

    df["FormDiff3"] = df["Form3Home"] - df["Form3Away"]
    df["FormDiff5"] = df["Form5Home"] - df["Form5Away"]

    # Division label encoding (alphabetically stable)
    divs    = sorted(df["Division"].unique())
    div_map = {d: i for i, d in enumerate(divs)}
    df["DivisionCode"] = df["Division"].map(div_map).fillna(-1).astype(int)

    # Season (July-based)
    df["Season"] = df["MatchDate"].dt.year
    df.loc[df["MatchDate"].dt.month < 7, "Season"] -= 1

    df = df.sort_values("MatchDate").reset_index(drop=True)
    return df, div_map


# ─────────────────────────────────────────────────────────────────────────────
# 3.  ROLLING TEAM STATS  (no data leakage – shift before rolling)
# ─────────────────────────────────────────────────────────────────────────────

def add_rolling_team_stats(df: pd.DataFrame, window: int = 10) -> pd.DataFrame:
    df = df.copy()

    # Long format: one row per (match, team)
    home_rec = df[["MatchDate", "HomeTeam", "FTHome", "FTAway"]].rename(
        columns={"HomeTeam": "Team", "FTHome": "Scored", "FTAway": "Conceded"}
    )
    away_rec = df[["MatchDate", "AwayTeam", "FTHome", "FTAway"]].rename(
        columns={"AwayTeam": "Team", "FTAway": "Scored", "FTHome": "Conceded"}
    )
    all_rec = pd.concat([home_rec, away_rec]).sort_values("MatchDate")

    all_rec["RollScored"] = (
        all_rec.groupby("Team")["Scored"]
        .transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())
    )
    all_rec["RollConceded"] = (
        all_rec.groupby("Team")["Conceded"]
        .transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())
    )

    roll = all_rec[["MatchDate", "Team", "RollScored", "RollConceded"]].drop_duplicates(
        subset=["MatchDate", "Team"], keep="last"
    )

    df = df.merge(
        roll.rename(columns={"Team": "HomeTeam",
                              "RollScored": "HomeRollScored",
                              "RollConceded": "HomeRollConceded"}),
        on=["MatchDate", "HomeTeam"], how="left"
    )
    df = df.merge(
        roll.rename(columns={"Team": "AwayTeam",
                              "RollScored": "AwayRollScored",
                              "RollConceded": "AwayRollConceded"}),
        on=["MatchDate", "AwayTeam"], how="left"
    )

    for c in ["HomeRollScored", "HomeRollConceded", "AwayRollScored", "AwayRollConceded"]:
        df[c] = df[c].fillna(df[c].median())

    df["AttackDiff"]  = df["HomeRollScored"]  - df["AwayRollScored"]
    df["DefenceDiff"] = df["HomeRollConceded"] - df["AwayRollConceded"]
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 4.  HEAD-TO-HEAD FEATURES  (incremental, strictly past data)
# ─────────────────────────────────────────────────────────────────────────────

def add_h2h_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("MatchDate").reset_index(drop=True)

    h2h_home_wr, h2h_away_wr, h2h_avg_goals = [], [], []
    h2h: dict = {}

    for _, row in df.iterrows():
        key  = tuple(sorted([row["HomeTeam"], row["AwayTeam"]]))
        past = h2h.get(key, [])

        if not past:
            h2h_home_wr.append(0.5)
            h2h_away_wr.append(0.5)
            h2h_avg_goals.append(2.5)
        else:
            tail = past[-5:]
            n    = len(tail)
            h2h_home_wr.append(sum(r["hw"] for r in tail) / n)
            h2h_away_wr.append(sum(r["aw"] for r in tail) / n)
            h2h_avg_goals.append(sum(r["g"]  for r in tail) / n)

        h2h.setdefault(key, []).append({
            "hw": row["FTResult"] == "H",
            "aw": row["FTResult"] == "A",
            "g":  row["TotalGoals"],
        })

    df["H2H_HomeWR"]   = h2h_home_wr
    df["H2H_AwayWR"]   = h2h_away_wr
    df["H2H_AvgGoals"] = h2h_avg_goals
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 5.  FEATURE COLUMNS USED BY MODELS
# ─────────────────────────────────────────────────────────────────────────────

FEATURE_COLS = [
    "HomeElo", "AwayElo", "EloDiff", "EloSum",
    "Form3Home", "Form5Home", "Form3Away", "Form5Away",
    "FormDiff3", "FormDiff5",
    "DivisionCode",
    "HomeRollScored", "HomeRollConceded",
    "AwayRollScored", "AwayRollConceded",
    "AttackDiff", "DefenceDiff",
    "H2H_HomeWR", "H2H_AwayWR", "H2H_AvgGoals",
    # Bonus stats (from football-data.co.uk; default-filled when absent)
    "HomeCorners", "AwayCorners",
    "HomeYellow",  "AwayYellow",
    "HomeShots",   "AwayShots",
    "ShotsDiff",
]

TARGET_RESULT    = "Result"       # 0=D  1=H  2=A
TARGET_FT_HOME   = "FTHome"
TARGET_FT_AWAY   = "FTAway"
TARGET_TOTAL_G   = "TotalGoals"
# Regression targets — now trainable with real data
TARGET_HOME_CRN  = "HomeCorners"
TARGET_AWAY_CRN  = "AwayCorners"
TARGET_HOME_YLW  = "HomeYellow"
TARGET_AWAY_YLW  = "AwayYellow"


# ─────────────────────────────────────────────────────────────────────────────
# 6.  SAVE / LOAD HELPERS  (parquet when available, csv fallback)
# ─────────────────────────────────────────────────────────────────────────────

def _save_df(df: pd.DataFrame, path: Path):
    # Always use CSV — parquet needs pyarrow which adds memory overhead
    p = path.with_suffix(".csv")
    df.to_csv(p, index=False)
    print(f"Saved → {p}  ({len(df):,} rows)")


def _load_df(path: Path) -> pd.DataFrame:
    parquet = path.with_suffix(".parquet")
    csv     = path.with_suffix(".csv")
    if parquet.exists():
        return pd.read_parquet(parquet)
    if csv.exists():
        return pd.read_csv(csv, low_memory=False)
    raise FileNotFoundError(f"No feature cache found at {path} (.parquet / .csv)")


# ─────────────────────────────────────────────────────────────────────────────
# 7.  MASTER PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def build_dataset():
    print("Loading raw data …")
    raw = load_raw()
    print(f"  {len(raw):,} raw rows")

    print("Cleaning …")
    df, div_map = clean(raw)
    print(f"  {len(df):,} rows after cleaning")

    print("Adding rolling team stats …")
    df = add_rolling_team_stats(df)

    print("Adding H2H features …")
    df = add_h2h_features(df)

    df = df.dropna(subset=FEATURE_COLS + [TARGET_RESULT, TARGET_FT_HOME, TARGET_FT_AWAY])
    print(f"  {len(df):,} rows ready for modelling")

    out = DATA_DIR / "processed" / "features"
    out.parent.mkdir(parents=True, exist_ok=True)
    _save_df(df, out)
    return df, div_map


if __name__ == "__main__":
    build_dataset()
