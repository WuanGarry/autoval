"""
fetch_data.py
Downloads ALL match data directly from football-data.co.uk — no API key,
no login, completely free. Updated twice weekly by the source.

Run standalone:   python scripts/fetch_data.py
Called by:        build.sh  (during Render deploy)
                  scripts/scheduler.py  (daily cron)
"""

import os, sys, time, logging
from io import StringIO
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))
logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger("fetch_data")

DATA_DIR = Path(os.environ.get("DATA_DIR",
           str(Path(__file__).resolve().parent.parent / "data")))
DATA_DIR.mkdir(parents=True, exist_ok=True)

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/120.0.0.0",
    "Referer":    "https://www.football-data.co.uk/",
}

# ── All seasons to download (most recent 5 seasons) ──────────────────────────
def get_seasons(n=5):
    from datetime import date
    yr = date.today().year if date.today().month >= 7 else date.today().year - 1
    return [f"{str(y)[2:]}{str(y+1)[2:]}" for y in range(yr, yr - n, -1)]

# ── Main leagues — football-data.co.uk/mmz4281/SSYY/DIV.csv ─────────────────
MAIN_LEAGUES = {
    "E0":  "English Premier League",
    "E1":  "English Championship",
    "E2":  "English League One",
    "E3":  "English League Two",
    "SP1": "Spanish La Liga",
    "SP2": "Spanish Segunda",
    "D1":  "German Bundesliga",
    "D2":  "German 2. Bundesliga",
    "I1":  "Italian Serie A",
    "I2":  "Italian Serie B",
    "F1":  "French Ligue 1",
    "F2":  "French Ligue 2",
    "N1":  "Dutch Eredivisie",
    "B1":  "Belgian First Division",
    "P1":  "Portuguese Primeira Liga",
    "T1":  "Turkish Super Lig",
    "SC0": "Scottish Premiership",
    "SC1": "Scottish Championship",
    "SC2": "Scottish League One",
    "SC3": "Scottish League Two",
}

# ── Extra leagues — football-data.co.uk/new/NAME.csv (all seasons in one) ───
EXTRA_LEAGUES = {
    "ARG": "ARG",   # Argentina
    "AUT": "AT1",   # Austria
    "BRA": "BSA",   # Brazil Serie A
    "DNK": "DK1",   # Denmark
    "FIN": "FIN",   # Finland
    "GRC": "GR1",   # Greece
    "JPN": "JP1",   # Japan J1
    "MEX": "MX1",   # Mexico Liga MX
    "NOR": "NO1",   # Norway
    "POL": "PL1",   # Poland
    "ROM": "RO1",   # Romania
    "RUS": "RU1",   # Russia
    "SWE": "SE1",   # Sweden
    "SWZ": "CH1",   # Switzerland
    "USA": "MLS",   # MLS
}

BASE = "https://www.football-data.co.uk"

# Column mapping: football-data.co.uk → our Matches.csv format
COL_MAP = {
    "Div":   "Division",  "Date":  "MatchDate", "Time":  "MatchTime",
    "HomeTeam": "HomeTeam", "AwayTeam": "AwayTeam",
    "FTHG":  "FTHome",    "FTAG":  "FTAway",    "FTR":   "FTResult",
    "HTHG":  "HTHome",    "HTAG":  "HTAway",    "HTR":   "HTResult",
    "HC":    "HomeCorners","AC":   "AwayCorners",
    "HY":    "HomeYellow","AY":    "AwayYellow",
    "HR":    "HomeRed",   "AR":    "AwayRed",
    "HS":    "HomeShots", "AS":    "AwayShots",
    "HST":   "HomeShotsTarget","AST": "AwayShotsTarget",
}


def fetch_csv(url):
    try:
        r = requests.get(url, headers=HEADERS, timeout=30)
        if r.status_code == 404:
            return None
        r.raise_for_status()
        text = r.content.decode("windows-1252", errors="replace")
        df = pd.read_csv(StringIO(text), low_memory=False)
        if df.empty or "HomeTeam" not in df.columns:
            return None
        return df
    except Exception as e:
        log.debug(f"fetch failed {url}: {e}")
        return None


def clean_df(df, div_override=None):
    df = df.rename(columns={k: v for k, v in COL_MAP.items() if k in df.columns})
    if div_override:
        df["Division"] = div_override
    needed = ["HomeTeam", "AwayTeam", "FTHome", "FTAway", "FTResult", "MatchDate"]
    for c in needed:
        if c not in df.columns:
            return pd.DataFrame()
    df = df.dropna(subset=["FTResult", "FTHome", "FTAway"])
    df = df[df["FTResult"].isin(["H", "D", "A"])]
    df["MatchDate"] = pd.to_datetime(
        df["MatchDate"], dayfirst=True, errors="coerce"
    ).dt.strftime("%d-%m-%Y")
    df = df.dropna(subset=["MatchDate"])
    # Add missing columns
    for c in ["HTHome","HTAway","HTResult","HomeElo","AwayElo",
              "Form3Home","Form5Home","Form3Away","Form5Away",
              "HomeCorners","AwayCorners","HomeYellow","AwayYellow",
              "HomeRed","AwayRed","HomeShots","AwayShots"]:
        if c not in df.columns:
            df[c] = ""
    keep = ["Division","MatchDate","MatchTime","HomeTeam","AwayTeam",
            "HomeElo","AwayElo","Form3Home","Form5Home","Form3Away","Form5Away",
            "FTHome","FTAway","FTResult","HTHome","HTAway","HTResult",
            "HomeCorners","AwayCorners","HomeYellow","AwayYellow",
            "HomeRed","AwayRed","HomeShots","AwayShots"]
    return df[[c for c in keep if c in df.columns]]


def fetch_all():
    seasons = get_seasons(n=5)
    log.info(f"Seasons to fetch: {seasons}")
    all_frames = []
    total_files = 0

    # ── Main leagues ─────────────────────────────────────────────────────────
    log.info(f"\nFetching {len(MAIN_LEAGUES)} main leagues × {len(seasons)} seasons...")
    for div, name in MAIN_LEAGUES.items():
        div_rows = 0
        for season in seasons:
            url = f"{BASE}/mmz4281/{season}/{div}.csv"
            df  = fetch_csv(url)
            if df is None:
                continue
            df = clean_df(df, div_override=div)
            if not df.empty:
                all_frames.append(df)
                div_rows += len(df)
                total_files += 1
            time.sleep(0.4)
        if div_rows:
            log.info(f"  {div:4s}  {name:<30}  {div_rows:6,} rows")

    # ── Extra leagues ─────────────────────────────────────────────────────────
    log.info(f"\nFetching {len(EXTRA_LEAGUES)} extra leagues...")
    for name, div in EXTRA_LEAGUES.items():
        url = f"{BASE}/new/{name}.csv"
        df  = fetch_csv(url)
        if df is None:
            log.debug(f"  {name} not found")
            continue
        df = clean_df(df, div_override=div)
        if not df.empty:
            all_frames.append(df)
            log.info(f"  {name:4s}  → {div:<8}  {len(df):6,} rows")
            total_files += 1
        time.sleep(0.4)

    if not all_frames:
        log.error("No data fetched! Check internet connection.")
        return False

    # ── Combine & deduplicate ─────────────────────────────────────────────────
    combined = pd.concat(all_frames, ignore_index=True)
    combined = combined.drop_duplicates(
        subset=["MatchDate","HomeTeam","AwayTeam"], keep="last"
    )
    combined = combined.sort_values("MatchDate").reset_index(drop=True)

    # Save
    matches_path = DATA_DIR / "Matches.csv"
    combined.to_csv(matches_path, index=False)

    log.info(f"\n{'='*50}")
    log.info(f"Total files fetched : {total_files}")
    log.info(f"Total rows          : {len(combined):,}")
    log.info(f"Unique teams        : {len(set(combined['HomeTeam'].tolist()))}")
    log.info(f"Divisions           : {sorted(combined['Division'].unique().tolist())}")
    log.info(f"Date range          : {combined['MatchDate'].min()} → {combined['MatchDate'].max()}")
    log.info(f"Saved to            : {matches_path}")
    log.info(f"{'='*50}\n")
    return True


if __name__ == "__main__":
    success = fetch_all()
    sys.exit(0 if success else 1)
