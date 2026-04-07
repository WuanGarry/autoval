"""
predictor.py
Loads trained models and generates a rich prediction for any home/away pair.
Also generates scoreline probability distribution using a Poisson model
seeded by the regressor's lambda estimates.
"""

import json
import math
import pickle
import warnings
from pathlib import Path
from functools import lru_cache

import numpy as np

warnings.filterwarnings("ignore")

import os; MODELS_DIR = Path(os.environ.get("MODELS_DIR", str(Path(__file__).resolve().parent.parent / "models")))
LABEL_MAP  = {0: "Draw", 1: "Home Win", 2: "Away Win"}
RESULT_MAP = {"Draw": "D", "Home Win": "H", "Away Win": "A"}


# ─────────────────────────────────────────────────────────────────────────────
# Load artefacts (cached so they are only read once per process)
# ─────────────────────────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def _load_artefacts():
    with open(MODELS_DIR / "metadata.json") as f:
        meta = json.load(f)
    def _pkl(name):
        p = MODELS_DIR / f"{name}.pkl"
        with open(p, "rb") as f:
            return pickle.load(f)

    outcome_model  = _pkl("outcome_model")
    home_model     = _pkl("home_goals_model")
    away_model     = _pkl("away_goals_model")
    home_crn_model = _pkl("home_corners_model")   if (MODELS_DIR/"home_corners_model.pkl").exists() else None
    away_crn_model = _pkl("away_corners_model")   if (MODELS_DIR/"away_corners_model.pkl").exists() else None
    home_ylw_model = _pkl("home_yellows_model")   if (MODELS_DIR/"home_yellows_model.pkl").exists() else None
    away_ylw_model = _pkl("away_yellows_model")   if (MODELS_DIR/"away_yellows_model.pkl").exists() else None
    return (meta, outcome_model, home_model, away_model,
            home_crn_model, away_crn_model,
            home_ylw_model, away_ylw_model)


# ─────────────────────────────────────────────────────────────────────────────
# Poisson scoreline distribution
# ─────────────────────────────────────────────────────────────────────────────

def _poisson_prob(lam: float, k: int) -> float:
    """P(X = k) for Poisson(lam)."""
    if lam <= 0:
        return 1.0 if k == 0 else 0.0
    return math.exp(-lam) * (lam ** k) / math.factorial(k)


def _scoreline_matrix(lam_home: float, lam_away: float, max_goals: int = 6):
    """
    Returns a dict  {(h, a): probability}  for all h, a in [0, max_goals].
    Probabilities are independent Poisson.
    """
    matrix = {}
    for h in range(max_goals + 1):
        for a in range(max_goals + 1):
            matrix[(h, a)] = _poisson_prob(lam_home, h) * _poisson_prob(lam_away, a)
    return matrix


def _top_scorelines(matrix: dict, n: int = 8):
    """Return the n most likely scorelines as a sorted list of dicts."""
    items = sorted(matrix.items(), key=lambda x: -x[1])
    return [
        {"scoreline": f"{h}-{a}", "probability": round(p * 100, 2)}
        for (h, a), p in items[:n]
    ]


def _result_probs_from_matrix(matrix: dict):
    """Derive H / D / A probabilities from a joint scoreline matrix."""
    ph = sum(p for (h, a), p in matrix.items() if h > a)
    pd_ = sum(p for (h, a), p in matrix.items() if h == a)
    pa = sum(p for (h, a), p in matrix.items() if h < a)
    total = ph + pd_ + pa
    return {
        "Home Win": round(ph  / total * 100, 1),
        "Draw":     round(pd_ / total * 100, 1),
        "Away Win": round(pa  / total * 100, 1),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Build feature vector for a given matchup
# ─────────────────────────────────────────────────────────────────────────────

def _build_feature_vector(home_team: str, away_team: str,
                           division: str, meta: dict) -> np.ndarray:
    ts = meta["team_stats"]
    div_map = meta["div_map"]

    hs = ts.get(home_team, {})
    as_ = ts.get(away_team, {})

    home_elo  = hs.get("elo",  1500.0)
    away_elo  = as_.get("elo", 1500.0)
    elo_diff  = home_elo - away_elo
    elo_sum   = home_elo + away_elo

    f3h = hs.get("form3Home", 0.0)
    f5h = hs.get("form5Home", 0.0)
    f3a = as_.get("form3Away", 0.0)
    f5a = as_.get("form5Away", 0.0)

    div_code = div_map.get(division, -1)

    home_rs = hs.get("rollScored",    1.3)
    home_rc = hs.get("rollConceded",  1.3)
    away_rs = as_.get("rollScored",   1.3)
    away_rc = as_.get("rollConceded", 1.3)

    attack_diff  = home_rs - away_rs
    defence_diff = home_rc - away_rc

    # H2H defaults (neutral)
    h2h_hwr  = 0.5
    h2h_awr  = 0.5
    h2h_avg  = meta.get("global_avg_total", 2.6)

    # Bonus stats — use team-level averages from metadata when available
    ts_h = ts.get(home_team, {})
    ts_a = ts.get(away_team, {})
    home_crn = ts_h.get("avgHomeCorners", 5.0)
    away_crn = ts_a.get("avgAwayCorners", 4.5)
    home_ylw = ts_h.get("avgHomeYellows", 1.5)
    away_ylw = ts_a.get("avgAwayYellows", 1.8)
    # Approximate shots from goals/rolling stats
    home_shots = max(6.0, home_rs * 7.0)
    away_shots = max(5.0, away_rs * 6.5)
    shots_diff = home_shots - away_shots

    feat = [
        home_elo, away_elo, elo_diff, elo_sum,
        f3h, f5h, f3a, f5a,
        f3h - f3a, f5h - f5a,
        div_code,
        home_rs, home_rc, away_rs, away_rc,
        attack_diff, defence_diff,
        h2h_hwr, h2h_awr, h2h_avg,
        # Bonus stats (positions 20-26)
        home_crn, away_crn,
        home_ylw, away_ylw,
        home_shots, away_shots,
        shots_diff,
    ]
    return np.array(feat, dtype=np.float32).reshape(1, -1)


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC API
# ─────────────────────────────────────────────────────────────────────────────

def get_teams(division: str = None) -> list[str]:
    meta, *_ = _load_artefacts()
    if division:
        # Filter teams that have played in that division
        # (metadata doesn't store per-division teams; return all for now)
        pass
    return sorted(meta["teams"])


def get_divisions() -> list[str]:
    meta, *_ = _load_artefacts()
    return meta["divisions"]


def predict(home_team: str, away_team: str, division: str = None) -> dict:
    """
    Generate a full match prediction.

    Returns
    -------
    dict with keys:
      home_team, away_team, division,
      predicted_result, result_probabilities,
      expected_goals_home, expected_goals_away,
      expected_total_goals,
      top_scorelines,
      expected_bookings_home, expected_bookings_away,
      expected_corners_home, expected_corners_away,
      team_stats_home, team_stats_away
    """
    (meta, outcome_model, home_model, away_model,
     home_crn_model, away_crn_model,
     home_ylw_model, away_ylw_model) = _load_artefacts()

    if division is None:
        division = meta["divisions"][0]

    # Build feature vector
    X = _build_feature_vector(home_team, away_team, division, meta)

    # ── Outcome probabilities ──────────────────────────────────────────────
    outcome_proba = outcome_model.predict_proba(X)[0]   # [Draw, Home, Away]
    outcome_classes = outcome_model.classes_             # e.g. [0, 1, 2]
    prob_dict_model = {
        LABEL_MAP[c]: round(float(p) * 100, 1)
        for c, p in zip(outcome_classes, outcome_proba)
    }

    # ── Goal expectations ──────────────────────────────────────────────────
    lam_home = float(np.clip(home_model.predict(X)[0], 0.05, 8.0))
    lam_away = float(np.clip(away_model.predict(X)[0], 0.05, 8.0))

    # ── Scoreline distribution (Poisson) ──────────────────────────────────
    matrix      = _scoreline_matrix(lam_home, lam_away)
    top_scores  = _top_scorelines(matrix)
    poisson_probs = _result_probs_from_matrix(matrix)

    # Blend model outcome proba (60 %) with Poisson-derived (40 %)
    blended = {}
    for label in ["Home Win", "Draw", "Away Win"]:
        blended[label] = round(
            0.60 * prob_dict_model.get(label, 0) +
            0.40 * poisson_probs.get(label, 0),
            1,
        )

    predicted_result = max(blended, key=blended.get)

    # ── Derived / estimated stats ──────────────────────────────────────────
    ts = meta["team_stats"]

    # Corners — use trained model if available, else scale from goals
    if home_crn_model is not None:
        est_corners_home = round(float(np.clip(home_crn_model.predict(X)[0], 1, 15)), 1)
        est_corners_away = round(float(np.clip(away_crn_model.predict(X)[0], 1, 15)), 1)
    else:
        est_corners_home = round(max(3.0, lam_home * 4.5), 1)
        est_corners_away = round(max(3.0, lam_away * 4.2), 1)

    # Bookings (yellow cards) — use trained model if available
    if home_ylw_model is not None:
        est_bookings_home = round(float(np.clip(home_ylw_model.predict(X)[0], 0, 6)), 1)
        est_bookings_away = round(float(np.clip(away_ylw_model.predict(X)[0], 0, 6)), 1)
    else:
        home_elo = ts.get(home_team, {}).get("elo", 1500)
        away_elo = ts.get(away_team, {}).get("elo", 1500)
        competitiveness = max(0.5, 1.5 - abs(home_elo - away_elo) / 600)
        est_bookings_home = round(3.5 * 0.5 * competitiveness + 0.5, 1)
        est_bookings_away = round(3.5 * 0.5 * competitiveness + 0.8, 1)

    # BTTS, Over 2.5
    btts_prob   = round((1 - _poisson_prob(lam_home, 0)) *
                        (1 - _poisson_prob(lam_away, 0)) * 100, 1)
    over25_prob = round(sum(p for (h, a), p in matrix.items() if h + a > 2) * 100, 1)
    over35_prob = round(sum(p for (h, a), p in matrix.items() if h + a > 3) * 100, 1)

    return {
        "home_team": home_team,
        "away_team": away_team,
        "division":  division,
        "predicted_result":        predicted_result,
        "result_probabilities":    blended,
        "expected_goals_home":     round(lam_home, 2),
        "expected_goals_away":     round(lam_away, 2),
        "expected_total_goals":    round(lam_home + lam_away, 2),
        "top_scorelines":          top_scores,
        "expected_bookings_home":  est_bookings_home,
        "expected_bookings_away":  est_bookings_away,
        "expected_corners_home":   est_corners_home,
        "expected_corners_away":   est_corners_away,
        "expected_total_corners":  round(est_corners_home + est_corners_away, 1),
        "corners_model":           "trained" if home_crn_model else "estimated",
        "bookings_model":          "trained" if home_ylw_model else "estimated",
        "betting_insights": {
            "btts_probability":   btts_prob,
            "over_2_5_goals":     over25_prob,
            "over_3_5_goals":     over35_prob,
        },
        "team_stats_home": ts.get(home_team, {}),
        "team_stats_away": ts.get(away_team, {}),
    }
