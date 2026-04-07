"""
train.py  –  Train all models and save to /models/.

Models trained
  • Outcome classifier         (H/D/A)
  • Home goals regressor
  • Away goals regressor
  • Home corners regressor     ← real model, trained on actual corner data
  • Away corners regressor
  • Home bookings regressor    ← real model, trained on yellow card data
  • Away bookings regressor

Run:  python scripts/train.py
"""

import sys, os, json, pickle, warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble     import (RandomForestClassifier,
                                   GradientBoostingClassifier,
                                   GradientBoostingRegressor)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics      import (accuracy_score, classification_report,
                                   mean_absolute_error, mean_squared_error)
from sklearn.pipeline     import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

try:
    from xgboost import XGBClassifier, XGBRegressor
    HAS_XGB = True
    print("XGBoost ✓")
except ImportError:
    HAS_XGB = False
    print("XGBoost not found – using GradientBoosting (install xgboost for better results)")

import os as _os
MODELS_DIR = Path(_os.environ.get("MODELS_DIR",
             str(Path(__file__).resolve().parent.parent / "models")))
DATA_DIR   = Path(_os.environ.get("DATA_DIR",
             str(Path(__file__).resolve().parent.parent / "data")))
MODELS_DIR.mkdir(parents=True, exist_ok=True)

from data_processor import (build_dataset, _load_df, FEATURE_COLS,
                              TARGET_RESULT, TARGET_FT_HOME, TARGET_FT_AWAY,
                              TARGET_HOME_CRN, TARGET_AWAY_CRN,
                              TARGET_HOME_YLW, TARGET_AWAY_YLW)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _save(obj, name):
    p = MODELS_DIR / f"{name}.pkl"
    with open(p, "wb") as f:
        pickle.dump(obj, f)
    print(f"  saved → {p.name}")
    return p


def _regressor(label=""):
    if HAS_XGB:
        m = XGBRegressor(n_estimators=400, max_depth=5, learning_rate=0.05,
                         subsample=0.8, colsample_bytree=0.8,
                         objective="count:poisson", n_jobs=-1, random_state=42)
    else:
        m = GradientBoostingRegressor(n_estimators=300, max_depth=4,
                                       learning_rate=0.05, subsample=0.8,
                                       random_state=42)
    return m


def train_regressor(X_tr, X_te, y_tr, y_te, label, clip_max=20):
    model = _regressor(label)
    model.fit(X_tr, y_tr)
    preds = np.clip(model.predict(X_te), 0, clip_max)
    mae   = mean_absolute_error(y_te, preds)
    rmse  = float(np.sqrt(mean_squared_error(y_te, preds)))
    print(f"\n── {label:<25}  MAE={mae:.3f}  RMSE={rmse:.3f}")
    return model


def train_classifier(X_tr, X_te, y_tr, y_te):
    candidates = {
        "LogisticRegression": Pipeline([
            ("sc",  StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs")),
        ]),
        "RandomForest": RandomForestClassifier(
            n_estimators=400, max_depth=12, min_samples_leaf=5,
            n_jobs=-1, random_state=42),
        "GradientBoosting": GradientBoostingClassifier(
            n_estimators=300, max_depth=5, learning_rate=0.05,
            subsample=0.8, random_state=42),
    }
    if HAS_XGB:
        candidates["XGBoost"] = XGBClassifier(
            n_estimators=400, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            eval_metric="mlogloss", n_jobs=-1, random_state=42)

    best_name, best_model, best_acc = None, None, -1
    print("\n── Outcome Classifier ──")
    for name, model in candidates.items():
        model.fit(X_tr, y_tr)
        acc = accuracy_score(y_te, model.predict(X_te))
        print(f"  {name:<25}  accuracy={acc:.4f}")
        if acc > best_acc:
            best_acc, best_name, best_model = acc, name, model

    print(f"\n  ✓ Best: {best_name}  (acc={best_acc:.4f})")
    print(classification_report(y_te, best_model.predict(X_te),
          target_names=["Draw","Home","Away"], zero_division=0))
    return best_model, best_name


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    feat_stem = DATA_DIR / "processed" / "features"
    try:
        print("Loading cached features …")
        df = _load_df(feat_stem)
    except FileNotFoundError:
        df, _ = build_dataset()

    meta_path = MODELS_DIR / "metadata.json"
    div_map   = json.loads(meta_path.read_text()).get("div_map", {}) \
                if meta_path.exists() else {}

    # ── Check for real corner / card data ────────────────────────────────
    has_corners  = df["HomeCorners"].std() > 0.1
    has_cards    = df["HomeYellow"].std()  > 0.1
    print(f"\nReal corners data: {'YES ✓' if has_corners else 'NO (using defaults)'}")
    print(f"Real cards data:   {'YES ✓' if has_cards   else 'NO (using defaults)'}")

    X      = df[FEATURE_COLS].values.astype(np.float32)
    y_res  = df[TARGET_RESULT].values.astype(int)
    y_hg   = df[TARGET_FT_HOME].values.astype(float)
    y_ag   = df[TARGET_FT_AWAY].values.astype(float)
    y_hc   = df[TARGET_HOME_CRN].values.astype(float)
    y_ac   = df[TARGET_AWAY_CRN].values.astype(float)
    y_hy   = df[TARGET_HOME_YLW].values.astype(float)
    y_ay   = df[TARGET_AWAY_YLW].values.astype(float)

    # Chronological split
    split  = int(len(X) * 0.85)
    X_tr,   X_te   = X[:split],     X[split:]
    res_tr, res_te = y_res[:split],  y_res[split:]
    hg_tr,  hg_te  = y_hg[:split],  y_hg[split:]
    ag_tr,  ag_te  = y_ag[:split],  y_ag[split:]
    hc_tr,  hc_te  = y_hc[:split],  y_hc[split:]
    ac_tr,  ac_te  = y_ac[:split],  y_ac[split:]
    hy_tr,  hy_te  = y_hy[:split],  y_hy[split:]
    ay_tr,  ay_te  = y_ay[:split],  y_ay[split:]

    print(f"\nTrain: {len(X_tr):,}   Test: {len(X_te):,}")

    clf, clf_name = train_classifier(X_tr, X_te, res_tr, res_te)
    hg_model = train_regressor(X_tr, X_te, hg_tr, hg_te, "Home Goals",    clip_max=10)
    ag_model = train_regressor(X_tr, X_te, ag_tr, ag_te, "Away Goals",    clip_max=10)
    hc_model = train_regressor(X_tr, X_te, hc_tr, hc_te, "Home Corners",  clip_max=20)
    ac_model = train_regressor(X_tr, X_te, ac_tr, ac_te, "Away Corners",  clip_max=20)
    hy_model = train_regressor(X_tr, X_te, hy_tr, hy_te, "Home Yellows",  clip_max=10)
    ay_model = train_regressor(X_tr, X_te, ay_tr, ay_te, "Away Yellows",  clip_max=10)

    print("\nSaving …")
    _save(clf,      "outcome_model")
    _save(hg_model, "home_goals_model")
    _save(ag_model, "away_goals_model")
    _save(hc_model, "home_corners_model")
    _save(ac_model, "away_corners_model")
    _save(hy_model, "home_yellows_model")
    _save(ay_model, "away_yellows_model")

    # ── Team stats ──────────────────────────────────────────────────────
    print("\nBuilding team stats …")
    teams = sorted(set(df["HomeTeam"].tolist() + df["AwayTeam"].tolist()))
    team_stats = {}
    for team in teams:
        hr = df[df["HomeTeam"] == team]
        ar = df[df["AwayTeam"] == team]
        elos = pd.concat([hr["HomeElo"].rename("e"), ar["AwayElo"].rename("e")])
        rs   = pd.concat([hr["HomeRollScored"].rename("v"),   ar["AwayRollScored"].rename("v")])
        rc   = pd.concat([hr["HomeRollConceded"].rename("v"), ar["AwayRollConceded"].rename("v")])
        team_stats[team] = {
            "elo":          round(float(elos.iloc[-1]) if len(elos) else 1500, 2),
            "rollScored":   round(float(rs.iloc[-1])   if len(rs)   else 1.3,  3),
            "rollConceded": round(float(rc.iloc[-1])   if len(rc)   else 1.3,  3),
            "form3Home":    round(float(hr["Form3Home"].mean()) if len(hr) else 0.0, 3),
            "form5Home":    round(float(hr["Form5Home"].mean()) if len(hr) else 0.0, 3),
            "form3Away":    round(float(ar["Form3Away"].mean()) if len(ar) else 0.0, 3),
            "form5Away":    round(float(ar["Form5Away"].mean()) if len(ar) else 0.0, 3),
            "avgHomeCorners": round(float(hr["HomeCorners"].mean()) if len(hr) else 5.0, 2),
            "avgAwayCorners": round(float(ar["AwayCorners"].mean()) if len(ar) else 4.5, 2),
            "avgHomeYellows": round(float(hr["HomeYellow"].mean())  if len(hr) else 1.5, 2),
            "avgAwayYellows": round(float(ar["AwayYellow"].mean())  if len(ar) else 1.8, 2),
        }

    from data_processor import clean, load_raw
    if not div_map:
        _, div_map = clean(load_raw())

    metadata = {
        "feature_cols":     FEATURE_COLS,
        "teams":            teams,
        "divisions":        sorted(df["Division"].unique().tolist()),
        "div_map":          div_map,
        "outcome_model":    clf_name,
        "team_stats":       team_stats,
        "has_corners_data": bool(has_corners),
        "has_cards_data":   bool(has_cards),
        "global_avg_total": round(float(df["TotalGoals"].mean()), 3),
        "global_avg_corners": round(float(df["HomeCorners"].mean() + df["AwayCorners"].mean()), 3),
        "global_avg_yellows": round(float(df["HomeYellow"].mean()  + df["AwayYellow"].mean()),  3),
    }
    meta_path.write_text(json.dumps(metadata, indent=2))
    print(f"\n  Metadata: {len(teams)} teams  |  {len(metadata['divisions'])} divisions")
    print("\n✅  Training complete.")


if __name__ == "__main__":
    main()
