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
    # England
    "E0": "E0",   # Premier League
    "E1": "E1",   # Championship
    "E2": "E2",   # League One
    "E3": "E3",   # League Two
    # Spain
    "SP1":"SP1",  # La Liga
    "SP2":"SP2",  # Segunda
    # Germany
    "D1": "D1",   # Bundesliga
    "D2": "D2",   # 2. Bundesliga
    # Italy
    "I1": "I1",   # Serie A   (Serie B removed — replaced by Norway)
    # France
    "F1": "F1",   # Ligue 1
    "F2": "F2",   # Ligue 2
    # Other domestic
    "N1": "N1",   # Dutch Eredivisie
    "B1": "B1",   # Belgian First Division
    "P1": "P1",   # Portuguese Primeira Liga
    "T1": "T1",   # Turkish Süper Lig
    "SC0":"SC0",  # Scottish Premiership  (SC1/SC2/SC3 removed — replaced by UEFA cups)
}

# Extra leagues
EXTRA_LEAGUES = {
    "ARG":"ARG",  # Argentina
    "AUT":"AT1",  # Austria
    "BRA":"BSA",  # Brazil Série A
    "DNK":"DK1",  # Denmark
    "GRC":"GR1",  # Greece
    "JPN":"JP1",  # Japan J1
    "MEX":"MX1",  # Mexico Liga MX
    "NOR":"NO1",  # Norway Eliteserien  ← replaces Italian Serie B
    "POL":"PL1",  # Poland
    "ROM":"RO1",  # Romania
    "SWE":"SE1",  # Sweden
    "SWZ":"CH1",  # Switzerland
    "USA":"MLS",  # MLS
}

# ── UEFA competitions ─────────────────────────────────────────────────────────
# Source: football-data.org free API  (register free at football-data.org)
# CL  = Champions League  (FREE tier — just needs a free registered key)
# EL  = Europa League     (free tier includes this too)
# UCL = Conference League (code is UCL on their API — confusingly named)
#
# Set env var: FOOTBALL_DATA_ORG_KEY=your_free_key
# Get key free at: https://www.football-data.org/client/register
#
UEFA_COMPS = {
    "CL":  "UCL",   # Champions League
    "EL":  "UEL",   # Europa League
    "UCL": "UECL",  # Conference League (football-data.org calls it UCL)
}
FD_ORG_BASE = "https://api.football-data.org/v4"
FD_ORG_KEY  = os.environ.get("FOOTBALL_DATA_ORG_KEY", "")

RESULT_MAP = {"HOME_TEAM": "H", "AWAY_TEAM": "A", "DRAW": "D"}



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



def fetch_uefa(seasons_back=2):
    """
    Fetch UCL / UEL / UECL results from football-data.org.
    Requires a FREE registered key from football-data.org/client/register
    Set env var: FOOTBALL_DATA_ORG_KEY=your_key
    If key is missing, this step is silently skipped.
    """
    if not FD_ORG_KEY:
        log.info("\n  UEFA comps: set FOOTBALL_DATA_ORG_KEY env var for UCL/UEL/UECL")
        log.info("  (free key at football-data.org/client/register)")
        return pd.DataFrame()

    from datetime import date as _date
    yr   = _date.today().year if _date.today().month >= 7 else _date.today().year - 1
    rows = []

    log.info(f"\nFetching UEFA competitions (football-data.org)...")
    for comp_code, our_div in UEFA_COMPS.items():
        for season_yr in range(yr, yr - seasons_back, -1):
            url    = f"{FD_ORG_BASE}/competitions/{comp_code}/matches"
            params = {"season": season_yr, "status": "FINISHED"}
            hdrs   = {"X-Auth-Token": FD_ORG_KEY}
            try:
                r = requests.get(url, headers=hdrs, params=params, timeout=30)
                if r.status_code == 403:
                    log.warning(f"  {comp_code} {season_yr}: 403 — may need higher tier key")
                    break
                if r.status_code == 429:
                    log.warning("  Rate limited — waiting 65s...")
                    time.sleep(65)
                    continue
                r.raise_for_status()
                matches = r.json().get("matches", [])
                season_rows = 0
                for m in matches:
                    try:
                        score = m["score"]
                        ft    = score["fullTime"]
                        ht    = score.get("halfTime") or {}
                        if ft.get("home") is None:
                            continue
                        winner = (score.get("winner") or "")
                        ftr    = RESULT_MAP.get(winner, "D")
                        fth, fta = int(ft["home"]), int(ft["away"])
                        hth  = ht.get("home")
                        hta  = ht.get("away")
                        htr  = ("H" if hth > hta else ("A" if hta > hth else "D")) \
                               if hth is not None and hta is not None else ""
                        dstr = datetime.strptime(
                            m["utcDate"][:10], "%Y-%m-%d"
                        ).strftime("%d-%m-%Y")
                        rows.append({
                            "Division":   our_div,
                            "MatchDate":  dstr,
                            "MatchTime":  "",
                            "HomeTeam":   m["homeTeam"]["name"],
                            "AwayTeam":   m["awayTeam"]["name"],
                            "HomeElo": "", "AwayElo": "",
                            "Form3Home":"","Form5Home":"",
                            "Form3Away":"","Form5Away":"",
                            "FTHome": fth, "FTAway": fta, "FTResult": ftr,
                            "HTHome": int(hth) if hth is not None else "",
                            "HTAway": int(hta) if hta is not None else "",
                            "HTResult": htr,
                            "HomeCorners":"","AwayCorners":"",
                            "HomeYellow":"","AwayYellow":"",
                            "HomeRed":"","AwayRed":"",
                            "HomeShots":"","AwayShots":"",
                        })
                        season_rows += 1
                    except Exception:
                        continue
                log.info(f"  {comp_code} {season_yr}: {season_rows} matches")
                time.sleep(1.0)
            except Exception as e:
                log.warning(f"  {comp_code} {season_yr}: {e}")

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    for c in KEEP_COLS:
        if c not in df.columns:
            df[c] = ""
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

    # UEFA competitions (free key from football-data.org)
    df_uefa = fetch_uefa()
    if not df_uefa.empty:
        df_uefa.to_csv(out_path, mode="a", index=False, header=not header_written)
        header_written = True
        total_rows += len(df_uefa)
        log.info(f"  UEFA comps: {len(df_uefa)} rows added")
        del df_uefa; gc.collect()

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
