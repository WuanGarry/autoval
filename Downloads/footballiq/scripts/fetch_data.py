"""
fetch_data.py  -  Downloads match data from football-data.co.uk
Memory-optimised: streams CSVs and processes in chunks.
Only fetches last 4 seasons to keep memory under 512MB.
"""

import os, sys, time, logging, gc
from io import StringIO
from pathlib import Path
from datetime import date

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

def get_seasons(n=4):
    yr = date.today().year if date.today().month >= 7 else date.today().year - 1
    return [f"{str(y)[2:]}{str(y+1)[2:]}" for y in range(yr, yr - n, -1)]

# Main leagues
MAIN_LEAGUES = {
    "E0": "E0", "E1": "E1", "E2": "E2", "E3": "E3",
    "SP1":"SP1","SP2":"SP2","D1": "D1", "D2": "D2",
    "I1": "I1", "I2": "I2", "F1": "F1", "F2": "F2",
    "N1": "N1", "B1": "B1", "P1": "P1", "T1": "T1",
    "SC0":"SC0","SC1":"SC1","SC2":"SC2","SC3":"SC3",
}

# Extra leagues
EXTRA_LEAGUES = {
    "ARG":"ARG","AUT":"AT1","BRA":"BSA","DNK":"DK1",
    "GRC":"GR1","JPN":"JP1","MEX":"MX1","NOR":"NO1",
    "POL":"PL1","ROM":"RO1","SWE":"SE1","SWZ":"CH1","USA":"MLS",
}

BASE    = "https://www.football-data.co.uk"
COL_MAP = {
    "Div":"Division","Date":"MatchDate","Time":"MatchTime",
    "HomeTeam":"HomeTeam","AwayTeam":"AwayTeam",
    "FTHG":"FTHome","FTAG":"FTAway","FTR":"FTResult",
    "HTHG":"HTHome","HTAG":"HTAway","HTR":"HTResult",
    "HC":"HomeCorners","AC":"AwayCorners",
    "HY":"HomeYellow","AY":"AwayYellow",
    "HR":"HomeRed","AR":"AwayRed",
    "HS":"HomeShots","AS":"AwayShots",
    "HST":"HomeShotsTarget","AST":"AwayShotsTarget",
}
KEEP_COLS = [
    "Division","MatchDate","MatchTime","HomeTeam","AwayTeam",
    "HomeElo","AwayElo","Form3Home","Form5Home","Form3Away","Form5Away",
    "FTHome","FTAway","FTResult","HTHome","HTAway","HTResult",
    "HomeCorners","AwayCorners","HomeYellow","AwayYellow",
    "HomeRed","AwayRed","HomeShots","AwayShots",
]


def fetch_csv(url):
    try:
        r = requests.get(url, headers=HEADERS, timeout=30)
        if r.status_code == 404:
            return None
        r.raise_for_status()
        text = r.content.decode("windows-1252", errors="replace")
        df = pd.read_csv(StringIO(text), low_memory=False)
        return df if not df.empty and "HomeTeam" in df.columns else None
    except Exception as e:
        log.debug(f"fetch failed {url}: {e}")
        return None


def clean_df(df, div_override=None):
    df = df.rename(columns={k:v for k,v in COL_MAP.items() if k in df.columns})
    if div_override:
        df["Division"] = div_override
    needed = ["HomeTeam","AwayTeam","FTHome","FTAway","FTResult","MatchDate"]
    for c in needed:
        if c not in df.columns:
            return pd.DataFrame()
    df = df.dropna(subset=["FTResult","FTHome","FTAway"])
    df = df[df["FTResult"].isin(["H","D","A"])]
    df["MatchDate"] = pd.to_datetime(
        df["MatchDate"], dayfirst=True, errors="coerce"
    ).dt.strftime("%d-%m-%Y")
    df = df.dropna(subset=["MatchDate"])
    for c in KEEP_COLS:
        if c not in df.columns:
            df[c] = ""
    # Use only float32 for numeric columns to save memory
    for c in ["FTHome","FTAway","HTHome","HTAway",
              "HomeCorners","AwayCorners","HomeYellow","AwayYellow"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype("float32")
    return df[[c for c in KEEP_COLS if c in df.columns]]


def fetch_all():
    seasons = get_seasons(n=4)
    log.info(f"Fetching seasons: {seasons}")

    out_path = DATA_DIR / "Matches.csv"
    # Write header first
    header_written = False
    total_rows = 0

    def append_df(df):
        nonlocal header_written, total_rows
        if df.empty:
            return
        df.to_csv(out_path, mode="a", index=False, header=not header_written)
        header_written = True
        total_rows += len(df)
        del df
        gc.collect()

    # Clear existing file
    if out_path.exists():
        out_path.unlink()

    # Main leagues
    log.info(f"\nFetching {len(MAIN_LEAGUES)} main leagues...")
    for div, code in MAIN_LEAGUES.items():
        div_rows = 0
        for season in seasons:
            url = f"{BASE}/mmz4281/{season}/{div}.csv"
            df  = fetch_csv(url)
            if df is None:
                continue
            df = clean_df(df, div_override=code)
            div_rows += len(df)
            append_df(df)
            time.sleep(0.3)
        if div_rows:
            log.info(f"  {div:4s}  {div_rows:6,} rows")

    # Extra leagues
    log.info(f"\nFetching {len(EXTRA_LEAGUES)} extra leagues...")
    for name, code in EXTRA_LEAGUES.items():
        url = f"{BASE}/new/{name}.csv"
        df  = fetch_csv(url)
        if df is None:
            continue
        df = clean_df(df, div_override=code)
        if not df.empty:
            log.info(f"  {name:4s}  {len(df):6,} rows")
            append_df(df)
        time.sleep(0.3)

    if total_rows == 0:
        log.error("No data fetched!")
        return False

    # Deduplicate
    log.info(f"\nDeduplicating {total_rows:,} rows...")
    df_all = pd.read_csv(out_path, low_memory=False)
    df_all = df_all.drop_duplicates(
        subset=["MatchDate","HomeTeam","AwayTeam"], keep="last"
    ).sort_values("MatchDate").reset_index(drop=True)
    df_all.to_csv(out_path, index=False)

    log.info(f"\n{'='*45}")
    log.info(f"Total rows      : {len(df_all):,}")
    log.info(f"Unique teams    : {df_all['HomeTeam'].nunique()}")
    log.info(f"Divisions       : {sorted(df_all['Division'].unique().tolist())}")
    log.info(f"{'='*45}\n")
    del df_all; gc.collect()
    return True


if __name__ == "__main__":
    import sys
    sys.exit(0 if fetch_all() else 1)
