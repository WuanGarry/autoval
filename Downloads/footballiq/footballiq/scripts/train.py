"""
train.py  -  Memory-optimised training for Render free tier (512MB RAM).

Key optimisations:
  - Uses only last 3 seasons of data (not all history)
  - Light models only: LogisticRegression + lightweight GradientBoosting
  - float32 throughout
  - Clears memory aggressively between steps
"""

import sys, os, json, pickle, warnings, gc
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model    import LogisticRegression, Ridge
from sklearn.ensemble        import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.pipeline        import Pipeline
from sklearn.preprocessing   import StandardScaler
from sklearn.metrics         import accuracy_score, mean_absolute_error
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

MODELS_DIR = Path(os.environ.get("MODELS_DIR",
             str(Path(__file__).resolve().parent.parent / "models")))
DATA_DIR   = Path(os.environ.get("DATA_DIR",
             str(Path(__file__).resolve().parent.parent / "data")))
MODELS_DIR.mkdir(parents=True, exist_ok=True)

from data_processor import (build_dataset, _load_df, FEATURE_COLS,
                              TARGET_RESULT, TARGET_FT_HOME, TARGET_FT_AWAY,
                              TARGET_HOME_CRN, TARGET_AWAY_CRN,
                              TARGET_HOME_YLW, TARGET_AWAY_YLW)


def save_pkl(obj, name):
    p = MODELS_DIR / f"{name}.pkl"
    with open(p, "wb") as f:
        pickle.dump(obj, f)
    print(f"  saved {name}.pkl")
    del obj
    gc.collect()


def train_classifier(X_tr, X_te, y_tr, y_te):
    """Lightweight classifier — LogisticRegression wins on memory."""
    print("\n── Outcome Classifier ──")

    # Logistic Regression — very low memory
    lr = Pipeline([
        ("sc",  StandardScaler()),
        ("clf", LogisticRegression(max_iter=500, C=1.0,
                                   solver="lbfgs", n_jobs=1)),
    ])
    lr.fit(X_tr, y_tr)
    acc_lr = accuracy_score(y_te, lr.predict(X_te))
    print(f"  LogisticRegression   acc={acc_lr:.4f}")

    # Lightweight GradientBoosting (fewer trees, shallow)
    gb = GradientBoostingClassifier(
        n_estimators=100, max_depth=3,
        learning_rate=0.1, subsample=0.8,
        random_state=42
    )
    gb.fit(X_tr, y_tr)
    acc_gb = accuracy_score(y_te, gb.predict(X_te))
    print(f"  GradientBoosting     acc={acc_gb:.4f}")

    if acc_gb > acc_lr:
        print(f"  ✓ Best: GradientBoosting (acc={acc_gb:.4f})")
        del lr; gc.collect()
        return gb, "GradientBoosting"
    else:
        print(f"  ✓ Best: LogisticRegression (acc={acc_lr:.4f})")
        del gb; gc.collect()
        return lr, "LogisticRegression"


def train_regressor(X_tr, X_te, y_tr, y_te, label, clip_max=10):
    """Ridge regression — extremely low memory, fast, decent results."""
    model = Pipeline([
        ("sc",  StandardScaler()),
        ("reg", Ridge(alpha=1.0)),
    ])
    model.fit(X_tr, y_tr)
    preds = np.clip(model.predict(X_te), 0, clip_max)
    mae   = mean_absolute_error(y_te, preds)
    print(f"  {label:<22}  MAE={mae:.3f}")
    gc.collect()
    return model


def main():
    # ── Load features ────────────────────────────────────────────────────────
    feat_stem = DATA_DIR / "processed" / "features"
    try:
        print("Loading cached features …")
        df = _load_df(feat_stem)
    except FileNotFoundError:
        df, _ = build_dataset()

    print(f"Full dataset: {len(df):,} rows")

    # ── Use only last 3 seasons to save memory ───────────────────────────────
    # Sort by date, keep most recent 40,000 rows max
    MAX_ROWS = 40_000
    if len(df) > MAX_ROWS:
        df = df.tail(MAX_ROWS).reset_index(drop=True)
        print(f"Trimmed to last {MAX_ROWS:,} rows to fit in 512MB RAM")

    # ── Check for real stats data ─────────────────────────────────────────────
    has_corners = df["HomeCorners"].std() > 0.1
    has_cards   = df["HomeYellow"].std()  > 0.1
    print(f"Real corners: {'YES' if has_corners else 'NO'}")
    print(f"Real cards:   {'YES' if has_cards   else 'NO'}")

    # ── Prepare arrays (float32 saves ~50% memory vs float64) ────────────────
    X      = df[FEATURE_COLS].values.astype(np.float32)
    y_res  = df[TARGET_RESULT].values.astype(np.int8)
    y_hg   = df[TARGET_FT_HOME].values.astype(np.float32)
    y_ag   = df[TARGET_FT_AWAY].values.astype(np.float32)
    y_hc   = df[TARGET_HOME_CRN].values.astype(np.float32)
    y_ac   = df[TARGET_AWAY_CRN].values.astype(np.float32)
    y_hy   = df[TARGET_HOME_YLW].values.astype(np.float32)
    y_ay   = df[TARGET_AWAY_YLW].values.astype(np.float32)

    # Free the dataframe immediately
    del df; gc.collect()

    # Chronological split
    split   = int(len(X) * 0.85)
    X_tr,   X_te   = X[:split],    X[split:]
    res_tr, res_te = y_res[:split], y_res[split:]

    print(f"\nTrain: {len(X_tr):,}   Test: {len(X_te):,}")

    # ── Train outcome classifier ──────────────────────────────────────────────
    clf, clf_name = train_classifier(X_tr, X_te, res_tr, res_te)
    save_pkl(clf, "outcome_model")
    gc.collect()

    # ── Train goal regressors ─────────────────────────────────────────────────
    print("\n── Goal Regressors ──")
    save_pkl(train_regressor(X_tr, X_te, y_hg[:split], y_hg[split:], "Home Goals"), "home_goals_model")
    save_pkl(train_regressor(X_tr, X_te, y_ag[:split], y_ag[split:], "Away Goals"), "away_goals_model")

    # ── Train corners regressors ──────────────────────────────────────────────
    print("\n── Corners Regressors ──")
    save_pkl(train_regressor(X_tr, X_te, y_hc[:split], y_hc[split:], "Home Corners", clip_max=20), "home_corners_model")
    save_pkl(train_regressor(X_tr, X_te, y_ac[:split], y_ac[split:], "Away Corners", clip_max=20), "away_corners_model")

    # ── Train card regressors ─────────────────────────────────────────────────
    print("\n── Card Regressors ──")
    save_pkl(train_regressor(X_tr, X_te, y_hy[:split], y_hy[split:], "Home Yellows", clip_max=8), "home_yellows_model")
    save_pkl(train_regressor(X_tr, X_te, y_ay[:split], y_ay[split:], "Away Yellows", clip_max=8), "away_yellows_model")

    del X_tr, X_te, res_tr, res_te
    gc.collect()

    # ── Build team stats from full array ─────────────────────────────────────
    print("\nBuilding team stats …")

    # Reload just what we need
    feat_stem2 = DATA_DIR / "processed" / "features"
    df2 = _load_df(feat_stem2)
    if len(df2) > MAX_ROWS:
        df2 = df2.tail(MAX_ROWS).reset_index(drop=True)

    teams = sorted(set(df2["HomeTeam"].tolist() + df2["AwayTeam"].tolist()))
    team_stats = {}
    for team in teams:
        hr = df2[df2["HomeTeam"] == team]
        ar = df2[df2["AwayTeam"] == team]
        elos = pd.concat([hr["HomeElo"].rename("e"), ar["AwayElo"].rename("e")])
        rs   = pd.concat([hr["HomeRollScored"].rename("v"),   ar["AwayRollScored"].rename("v")])
        rc   = pd.concat([hr["HomeRollConceded"].rename("v"), ar["AwayRollConceded"].rename("v")])
        team_stats[team] = {
            "elo":            round(float(elos.iloc[-1]) if len(elos) else 1500, 2),
            "rollScored":     round(float(rs.iloc[-1])   if len(rs)   else 1.3,  3),
            "rollConceded":   round(float(rc.iloc[-1])   if len(rc)   else 1.3,  3),
            "form3Home":      round(float(hr["Form3Home"].mean()) if len(hr) else 0.0, 3),
            "form5Home":      round(float(hr["Form5Home"].mean()) if len(hr) else 0.0, 3),
            "form3Away":      round(float(ar["Form3Away"].mean()) if len(ar) else 0.0, 3),
            "form5Away":      round(float(ar["Form5Away"].mean()) if len(ar) else 0.0, 3),
            "avgHomeCorners": round(float(hr["HomeCorners"].mean()) if len(hr) else 5.0, 2),
            "avgAwayCorners": round(float(ar["AwayCorners"].mean()) if len(ar) else 4.5, 2),
            "avgHomeYellows": round(float(hr["HomeYellow"].mean())  if len(hr) else 1.5, 2),
            "avgAwayYellows": round(float(ar["AwayYellow"].mean())  if len(ar) else 1.8, 2),
        }

    from data_processor import clean, load_raw
    meta_path = MODELS_DIR / "metadata.json"
    div_map   = json.loads(meta_path.read_text()).get("div_map", {}) \
                if meta_path.exists() else {}
    if not div_map:
        _, div_map = clean(load_raw())

    metadata = {
        "feature_cols":       FEATURE_COLS,
        "teams":              teams,
        "divisions":          sorted(df2["Division"].unique().tolist()),
        "div_map":            div_map,
        "outcome_model":      clf_name,
        "team_stats":         team_stats,
        "has_corners_data":   bool(has_corners),
        "has_cards_data":     bool(has_cards),
        "global_avg_total":   round(float(df2["TotalGoals"].mean()), 3),
        "global_avg_corners": round(float(df2["HomeCorners"].mean() +
                                          df2["AwayCorners"].mean()), 3),
        "global_avg_yellows": round(float(df2["HomeYellow"].mean() +
                                          df2["AwayYellow"].mean()), 3),
    }
    meta_path.write_text(json.dumps(metadata, indent=2))
    del df2; gc.collect()

    print(f"\n  {len(teams)} teams  |  {len(metadata['divisions'])} divisions")
    print("\n✅  Training complete.")


if __name__ == "__main__":
    main()
