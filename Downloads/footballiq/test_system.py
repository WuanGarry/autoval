#!/usr/bin/env python3
"""
test_system.py  –  Smoke-tests the entire pipeline without starting a server.

Run:   python test_system.py
Expected: all tests pass with ✓
"""

import sys, os, json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "backend"))

PASS, FAIL = "✓", "✗"
errors = []

def check(label, condition, detail=""):
    if condition:
        print(f"  {PASS}  {label}")
    else:
        print(f"  {FAIL}  {label}" + (f"  →  {detail}" if detail else ""))
        errors.append(label)


print("\n══════════════════════════════════════════")
print("  FootballIQ  –  System Smoke Test")
print("══════════════════════════════════════════\n")

# ── 1. File structure ─────────────────────────────────────────────────────
print("[ 1 ]  File structure")
check("data/Matches.csv exists",           (ROOT/"data"/"Matches.csv").exists())
check("backend/data_processor.py exists",  (ROOT/"backend"/"data_processor.py").exists())
check("backend/predictor.py exists",       (ROOT/"backend"/"predictor.py").exists())
check("backend/app.py exists",             (ROOT/"backend"/"app.py").exists())
check("scripts/train.py exists",           (ROOT/"scripts"/"train.py").exists())
check("scripts/update_data.py exists",     (ROOT/"scripts"/"update_data.py").exists())
check("frontend/index.html exists",        (ROOT/"frontend"/"index.html").exists())
check("requirements.txt exists",           (ROOT/"requirements.txt").exists())
check("Procfile exists",                   (ROOT/"Procfile").exists())

# ── 2. Trained models ─────────────────────────────────────────────────────
print("\n[ 2 ]  Trained model files")
check("models/outcome_model.pkl",    (ROOT/"models"/"outcome_model.pkl").exists())
check("models/home_goals_model.pkl", (ROOT/"models"/"home_goals_model.pkl").exists())
check("models/away_goals_model.pkl", (ROOT/"models"/"away_goals_model.pkl").exists())
check("models/metadata.json",        (ROOT/"models"/"metadata.json").exists())

# ── 3. Metadata content ───────────────────────────────────────────────────
print("\n[ 3 ]  Metadata integrity")
try:
    meta = json.loads((ROOT/"models"/"metadata.json").read_text())
    check("teams list non-empty",      len(meta.get("teams", [])) > 0,
          f"got {len(meta.get('teams',[]))}")
    check("divisions list non-empty",  len(meta.get("divisions", [])) > 0)
    check("feature_cols present",      len(meta.get("feature_cols", [])) == 27,
          f"got {len(meta.get('feature_cols',[]))}")
    check("team_stats populated",      len(meta.get("team_stats", {})) > 0)
    check("outcome_model named",       bool(meta.get("outcome_model")))
except Exception as e:
    errors.append(f"metadata read: {e}")
    print(f"  {FAIL}  metadata read failed: {e}")

# ── 4. Predictor engine ───────────────────────────────────────────────────
print("\n[ 4 ]  Predictor engine")
try:
    import predictor
    teams = predictor.get_teams()
    check("get_teams() returns list",  isinstance(teams, list) and len(teams) > 0,
          f"got {len(teams)}")

    divs = predictor.get_divisions()
    check("get_divisions() returns list", isinstance(divs, list) and len(divs) > 0)

    home, away = teams[0], teams[1]
    result = predictor.predict(home, away)

    check("predict() returns dict",    isinstance(result, dict))
    check("predicted_result present",  result.get("predicted_result") in
          ["Home Win", "Draw", "Away Win"],
          result.get("predicted_result"))
    probs = result.get("result_probabilities", {})
    total = sum(probs.values())
    check("probabilities sum ≈ 100",   95 <= total <= 105,  f"sum={total:.1f}")
    check("expected_goals_home > 0",   result.get("expected_goals_home", 0) > 0)
    check("expected_goals_away > 0",   result.get("expected_goals_away", 0) > 0)
    check("top_scorelines has 8 rows", len(result.get("top_scorelines", [])) == 8)
    check("betting_insights present",  "btts_probability" in result.get("betting_insights", {}))
    check("corners > 0",               result.get("expected_corners_home", 0) > 0)
    check("bookings > 0",              result.get("expected_bookings_home", 0) > 0)
    print(f"\n     Sample: {home} vs {away}")
    print(f"     Result : {result['predicted_result']}")
    print(f"     Probs  : {probs}")
    print(f"     Goals  : {result['expected_goals_home']} – {result['expected_goals_away']}")
    print(f"     Score  : {result['top_scorelines'][0]['scoreline']} "
          f"({result['top_scorelines'][0]['probability']}%)")
except Exception as e:
    errors.append(f"predictor: {e}")
    print(f"  {FAIL}  predictor failed: {e}")

# ── 5. Flask API routes ───────────────────────────────────────────────────
print("\n[ 5 ]  Flask API routes")
try:
    from app import app
    client = app.test_client()

    r = client.get("/api/teams")
    d = json.loads(r.data)
    check("GET /api/teams → ok",       d.get("status") == "ok")
    check("teams count > 0",           len(d.get("data", {}).get("teams", [])) > 0)

    r = client.get("/api/divisions")
    d = json.loads(r.data)
    check("GET /api/divisions → ok",   d.get("status") == "ok")

    r = client.get("/api/model-status")
    d = json.loads(r.data)
    check("GET /api/model-status → ok", d.get("status") == "ok")

    payload = json.dumps({"home_team": teams[0], "away_team": teams[1]})
    r = client.post("/api/predict",
                    data=payload, content_type="application/json")
    d = json.loads(r.data)
    check("POST /api/predict → ok",    d.get("status") == "ok",
          d.get("message", ""))
    check("predict result field",      "predicted_result" in d.get("data", {}))

    # Error handling
    r = client.post("/api/predict",
                    data=json.dumps({"home_team": teams[0], "away_team": teams[0]}),
                    content_type="application/json")
    d = json.loads(r.data)
    check("same-team → error response", d.get("status") == "error")

except Exception as e:
    errors.append(f"Flask API: {e}")
    print(f"  {FAIL}  Flask API failed: {e}")

# ── Summary ───────────────────────────────────────────────────────────────
print("\n══════════════════════════════════════════")
if errors:
    print(f"  FAILED  ({len(errors)} issue(s)):")
    for e in errors:
        print(f"    • {e}")
    sys.exit(1)
else:
    print("  ALL TESTS PASSED  🎉")
print("══════════════════════════════════════════\n")
