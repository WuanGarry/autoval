# ⚽ FootballIQ – AI Football Match Predictor

> 230 000+ historical matches · 20 European leagues · XGBoost + RandomForest · Flask API · Single-page UI

---

## Project Structure

```
football-predictor/
├── data/
│   ├── Matches.csv          ← your raw data (place here)
│   ├── EloRatings.csv       ← your ELO data (place here)
│   └── processed/
│       └── features.parquet ← auto-generated after training
│
├── models/
│   ├── outcome_model.pkl    ← best classifier (H/D/A)
│   ├── home_goals_model.pkl ← XGBoost Poisson regressor
│   ├── away_goals_model.pkl ← XGBoost Poisson regressor
│   └── metadata.json        ← team list, division map, stats
│
├── backend/
│   ├── app.py               ← Flask REST API
│   ├── data_processor.py    ← cleaning + feature engineering
│   └── predictor.py         ← prediction engine
│
├── frontend/
│   └── index.html           ← single-file UI (HTML + CSS + JS)
│
├── scripts/
│   ├── train.py             ← model training script
│   ├── update_data.py       ← live data fetcher
│   └── scheduler.py         ← daily cron entry point
│
├── requirements.txt
├── Procfile                 ← for Render / Heroku
├── render.yaml              ← one-click Render config
└── README.md
```

---

## Quick Start (Local)

### 1. Install dependencies
```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Place your data files
```
data/Matches.csv
data/EloRatings.csv
```

### 3. Train models  (~5-10 minutes first time)
```bash
python scripts/train.py
```

### 4. Start the server
```bash
python backend/app.py
# or with gunicorn:
gunicorn backend.app:app --workers 2 --bind 0.0.0.0:5000
```

### 5. Open in browser
```
http://localhost:5000
```

---

## API Endpoints

| Method | Endpoint            | Description                          |
|--------|---------------------|--------------------------------------|
| GET    | `/api/teams`        | All teams (optional `?division=E0`)  |
| GET    | `/api/divisions`    | All competition codes + labels       |
| POST   | `/api/predict`      | Predict a match                      |
| POST   | `/api/update-data`  | Trigger live data fetch + retrain    |
| GET    | `/api/model-status` | Last train time, team count, etc.    |

### POST /api/predict – Example
```bash
curl -X POST http://localhost:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{"home_team": "Arsenal", "away_team": "Chelsea", "division": "E0"}'
```

Response:
```json
{
  "status": "ok",
  "data": {
    "home_team": "Arsenal",
    "away_team": "Chelsea",
    "predicted_result": "Home Win",
    "result_probabilities": {"Home Win": 47.3, "Draw": 26.1, "Away Win": 26.6},
    "expected_goals_home": 1.83,
    "expected_goals_away": 1.21,
    "expected_total_goals": 3.04,
    "top_scorelines": [{"scoreline": "2-1", "probability": 11.4}, ...],
    "betting_insights": {"btts_probability": 58.2, "over_2_5_goals": 54.7, ...},
    ...
  }
}
```

---

## Automatic Data Updates

1. Register free at https://www.football-data.org/client/register
2. Set your API key:
   ```bash
   export FOOTBALL_DATA_API_KEY=your_key_here
   ```
3. Run manually:
   ```bash
   python scripts/update_data.py
   ```
4. Schedule daily (crontab):
   ```
   0 3 * * * cd /path/to/football-predictor && python scripts/scheduler.py
   ```

---

## Deploy to Render

1. Push this repo to GitHub
2. Go to https://render.com → New Web Service → connect your repo
3. Render auto-detects `render.yaml`
4. Set `FOOTBALL_DATA_API_KEY` in Render environment variables
5. Deploy – build step runs `scripts/train.py` automatically

---

## Models

| Target           | Model                  | Metric               |
|------------------|------------------------|----------------------|
| Match outcome    | XGBoost / RandomForest | ~58-62% accuracy     |
| Home goals       | XGBoost (Poisson)      | MAE ~0.9             |
| Away goals       | XGBoost (Poisson)      | MAE ~0.8             |
| Scoreline dist.  | Bivariate Poisson      | derived from above   |

Features used (pre-match only):
- ELO ratings (home & away)
- Recent form (last 3 & 5 matches)
- Rolling goals scored / conceded (last 10 games)
- Head-to-head win rate & avg goals
- Division encoding

---

## Notes

- **No data leakage**: all features use only past information (shift + rolling)
- **Corners & bookings**: estimated from goal expectation + ELO competitiveness
  (dataset does not include raw corners/cards — extend `Matches.csv` columns to add these)
- **BTTS / Over 2.5**: computed from bivariate Poisson distribution
