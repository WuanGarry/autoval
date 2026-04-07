"""
update_data.py  –  100% FREE data pipeline, zero API keys required.

SOURCE 1: football-data.co.uk  (primary)
  Free CSV downloads for all domestic leagues.
  Updated twice weekly. Includes: goals, corners, cards, shots, odds.
  URL pattern: https://www.football-data.co.uk/mmz4281/{SSYY}/{DIV}.csv
  e.g. https://www.football-data.co.uk/mmz4281/2425/E0.csv  (PL 2024/25)

SOURCE 2: flashscore_scraper   (secondary – for UCL/UEL/UECL only)
  Free open-source PyPI package using a headless Playwright browser.
  Install: pip install flashscore_scraper playwright
           playwright install chromium
  Only used for European cups not available on football-data.co.uk.

Run:  python scripts/update_data.py
"""

import os, sys, json, time, logging, re
from datetime import datetime, date, timedelta
from pathlib import Path
from io import StringIO

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger("update_data")

DATA_DIR   = Path(os.environ.get("DATA_DIR",
             str(Path(__file__).resolve().parent.parent / "data")))
CACHE_FILE = DATA_DIR / "last_update.json"

# ─────────────────────────────────────────────────────────────────────────────
# SOURCE 1: football-data.co.uk
# URL: https://www.football-data.co.uk/mmz4281/{SSYY}/{DIV}.csv
# Free, no key required, updated twice weekly.
# ─────────────────────────────────────────────────────────────────────────────

# Main leagues (season-by-season files)
FDUK_MAIN = {
    # England
    "E0": "E0", "E1": "E1", "E2": "E2", "E3": "E3",
    # Spain
    "SP1": "SP1", "SP2": "SP2",
    # Germany
    "D1": "D1",  "D2": "D2",
    # Italy
    "I1": "I1",  "I2": "I2",
    # France
    "F1": "F1",  "F2": "F2",
    # Other European
    "N1": "N1",   # Dutch Eredivisie
    "B1": "B1",   # Belgian Jupiler
    "P1": "P1",   # Portuguese Primeira
    "T1": "T1",   # Turkish Süper Lig
    "G1": "GR1",  # Greek Super League (FDUK uses G1)
    # Scotland
    "SC0": "SC0", "SC1": "SC1", "SC2": "SC2", "SC3": "SC3",
}

# Extra leagues — single all-seasons file at /new/{name}.csv
# These include Argentina, Austria, Brazil, MLS, etc.
FDUK_EXTRA = {
    "ARG":  "ARG",   # Argentine Primera División
    "AUT":  "AT1",   # Austrian Bundesliga
    "BRA":  "BSA",   # Brazilian Série A
    "CHN":  "CN1",   # Chinese Super League
    "DNK":  "DK1",   # Danish Superliga
    "FIN":  "FIN",   # Finnish Veikkausliiga
    "GRC":  "GR1",   # Greek Super League (alt path)
    "IRL":  "IR1",   # Irish Premier Division
    "JPN":  "JP1",   # Japanese J1 League
    "MEX":  "MX1",   # Liga MX
    "NOR":  "NO1",   # Norwegian Eliteserien
    "POL":  "PL1",   # Polish Ekstraklasa
    "ROM":  "RO1",   # Romanian Liga I
    "RUS":  "RU1",   # Russian Premier League
    "SCO":  "SCO",   # Scotland (alt)
    "SVK":  "SK1",   # Slovak Super Liga
    "SWE":  "SE1",   # Swedish Allsvenskan
    "SWZ":  "CH1",   # Swiss Super League
    "USA":  "MLS",   # MLS
}

FDUK_BASE      = "https://www.football-data.co.uk"
FDUK_MAIN_URL  = FDUK_BASE + "/mmz4281/{season}/{div}.csv"
FDUK_EXTRA_URL = FDUK_BASE + "/new/{name}.csv"

HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                   "AppleWebKit/537.36 (KHTML, like Gecko) "
                   "Chrome/120.0.0.0 Safari/537.36"),
    "Accept":     "text/html,application/xhtml+xml,text/csv,*/*",
    "Referer":    "https://www.football-data.co.uk/",
}

# ── Column mapping: FDUK CSV → our Matches.csv ───────────────────────────────
COL_MAP = {
    "Div":      "Division",
    "Date":     "MatchDate",
    "Time":     "MatchTime",
    "HomeTeam": "HomeTeam",
    "AwayTeam": "AwayTeam",
    "FTHG":     "FTHome",
    "FTAG":     "FTAway",
    "FTR":      "FTResult",
    "HTHG":     "HTHome",
    "HTAG":     "HTAway",
    "HTR":      "HTResult",
    "HC":       "HomeCorners",   # bonus columns — used if present
    "AC":       "AwayCorners",
    "HY":       "HomeYellow",
    "AY":       "AwayYellow",
    "HR":       "HomeRed",
    "AR":       "AwayRed",
    "HS":       "HomeShots",
    "AS":       "AwayShots",
    "HST":      "HomeShotsTarget",
    "AST":      "AwayShotsTarget",
}


def _season_codes(n_back: int = 3) -> list[str]:
    """Return the last n season codes like ['2425','2324','2223']."""
    codes = []
    today = date.today()
    yr = today.year if today.month >= 7 else today.year - 1
    for i in range(n_back):
        y = yr - i
        codes.append(f"{str(y)[2:]}{str(y+1)[2:]}")
    return codes


def _fetch_csv(url: str) -> pd.DataFrame | None:
    """Download a CSV URL and return as DataFrame, or None on failure."""
    try:
        r = requests.get(url, headers=HEADERS, timeout=30)
        if r.status_code == 404:
            return None
        r.raise_for_status()
        # FDUK files use Windows-1252 encoding
        text = r.content.decode("windows-1252", errors="replace")
        df = pd.read_csv(StringIO(text), low_memory=False)
        if df.empty or "HomeTeam" not in df.columns:
            return None
        return df
    except Exception as e:
        log.debug(f"    fetch failed ({url}): {e}")
        return None


def _normalise_fduk(df: pd.DataFrame, division_override: str = None) -> pd.DataFrame:
    """
    Rename FDUK columns to our standard names.
    Keeps bonus columns (corners, cards, shots) if present.
    """
    df = df.rename(columns={k: v for k, v in COL_MAP.items() if k in df.columns})

    # Override division code if we know the correct one (extra leagues)
    if division_override:
        df["Division"] = division_override

    # Ensure mandatory columns exist
    for col in ["FTHome", "FTAway", "FTResult", "HomeTeam", "AwayTeam", "MatchDate"]:
        if col not in df.columns:
            return pd.DataFrame()

    # Drop rows with no result
    df = df.dropna(subset=["FTResult", "FTHome", "FTAway"])
    df = df[df["FTResult"].isin(["H", "D", "A"])]

    # Standardise date format → DD-MM-YYYY
    df["MatchDate"] = pd.to_datetime(
        df["MatchDate"], dayfirst=True, errors="coerce"
    ).dt.strftime("%d-%m-%Y")
    df = df.dropna(subset=["MatchDate"])

    # Keep only our standard columns (+ bonus ones that exist)
    our_cols = list(COL_MAP.values())
    keep = [c for c in our_cols if c in df.columns]
    # Pad missing mandatory columns
    for c in ["HTHome", "HTAway", "HTResult", "HomeElo", "AwayElo",
              "Form3Home", "Form5Home", "Form3Away", "Form5Away"]:
        if c not in df.columns:
            df[c] = ""
    keep += [c for c in ["HomeElo", "AwayElo", "Form3Home", "Form5Home",
                          "Form3Away", "Form5Away"] if c not in keep]
    keep = list(dict.fromkeys(keep))
    return df[[c for c in keep if c in df.columns]]


def fetch_fduk_main(date_from: str) -> pd.DataFrame:
    """
    Download season CSV files from football-data.co.uk for all main leagues.
    Filters rows to only those on or after date_from.
    """
    cutoff   = pd.to_datetime(date_from, dayfirst=False)
    seasons  = _season_codes(n_back=2)    # current + last season
    all_rows = []

    log.info(f"\n── Source 1: football-data.co.uk  (main leagues, {len(FDUK_MAIN)} divs) ──")
    for fduk_div, our_div in FDUK_MAIN.items():
        collected = 0
        for season in seasons:
            url = FDUK_MAIN_URL.format(season=season, div=fduk_div)
            df  = _fetch_csv(url)
            if df is None:
                continue
            df  = _normalise_fduk(df, division_override=our_div)
            if df.empty:
                continue
            df["_dt"] = pd.to_datetime(df["MatchDate"], dayfirst=True, errors="coerce")
            df = df[df["_dt"] >= cutoff].drop(columns=["_dt"])
            collected += len(df)
            all_rows.append(df)
            time.sleep(0.3)   # polite delay
        if collected:
            log.info(f"  {fduk_div:4s} → {our_div:5s}  {collected:4d} new rows")

    return pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame()


def fetch_fduk_extra(date_from: str) -> pd.DataFrame:
    """
    Download all-seasons CSV files from football-data.co.uk /new/ directory.
    """
    cutoff   = pd.to_datetime(date_from, dayfirst=False)
    all_rows = []

    log.info(f"\n── Source 1 extra: football-data.co.uk  (/new/ leagues, {len(FDUK_EXTRA)} files) ──")
    for name, our_div in FDUK_EXTRA.items():
        url = FDUK_EXTRA_URL.format(name=name)
        df  = _fetch_csv(url)
        if df is None:
            log.debug(f"  {name} not found")
            continue
        df = _normalise_fduk(df, division_override=our_div)
        if df.empty:
            continue
        df["_dt"] = pd.to_datetime(df["MatchDate"], dayfirst=True, errors="coerce")
        df = df[df["_dt"] >= cutoff].drop(columns=["_dt"])
        if len(df):
            log.info(f"  {name:5s} → {our_div:5s}  {len(df):4d} new rows")
            all_rows.append(df)
        time.sleep(0.3)

    return pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame()


# ─────────────────────────────────────────────────────────────────────────────
# SOURCE 2: flashscore_scraper  (for UCL / UEL / UECL)
# Requires:  pip install flashscore_scraper playwright
#            playwright install chromium
# ─────────────────────────────────────────────────────────────────────────────

# Flashscore league configurations for UEFA competitions
# Keys = (flashscore country slug, flashscore league slug, our internal code)
FLASHSCORE_UEFA = [
    ("europe", "champions-league",           "UCL"),
    ("europe", "europa-league",              "UEL"),
    ("europe", "conference-league",          "UECL"),
]

# Additional leagues only on flashscore (not on football-data.co.uk)
FLASHSCORE_EXTRA = [
    ("europe", "champions-league",           "UCL"),
    ("europe", "europa-league",              "UEL"),
    ("europe", "conference-league",          "UECL"),
    ("south-america", "copa-libertadores",   "COPLIB"),
    ("south-america", "copa-sudamericana",   "COPASUD"),
    ("world",  "club-world-cup",             "CWC"),
    ("croatia",     "hnl",                   "HR1"),
    ("ukraine",     "premier-league",        "UA1"),
    ("romania",     "liga-1",                "RO1"),
    ("greece",      "super-league",          "GR1"),
    ("austria",     "bundesliga",            "AT1"),
    ("switzerland", "super-league",          "CH1"),
    ("denmark",     "superliga",             "DK1"),
    ("sweden",      "allsvenskan",           "SE1"),
    ("norway",      "eliteserien",           "NO1"),
    ("russia",      "premier-league",        "RU1"),
    ("scotland",    "premiership",           "SC0"),
]


def _parse_flashscore_df(matches: list, division: str) -> pd.DataFrame:
    """Convert flashscore_scraper match objects to our CSV row format."""
    rows = []
    for m in matches:
        try:
            # flashscore_scraper exposes these attributes
            home  = getattr(m, "home_team",  None) or getattr(m, "homeTeam", None)
            away  = getattr(m, "away_team",  None) or getattr(m, "awayTeam", None)
            score = getattr(m, "score",      None) or getattr(m, "result",   None)
            dt    = getattr(m, "date",       None) or getattr(m, "utcDate",  None)

            if not home or not away or not score:
                continue

            # Score can be "2:1" or "2-1" or a dict
            if isinstance(score, str):
                parts = re.split(r"[:\-]", score.strip())
                if len(parts) < 2:
                    continue
                fth, fta = int(parts[0]), int(parts[1])
            elif hasattr(score, "home"):
                fth, fta = int(score.home), int(score.away)
            elif isinstance(score, dict):
                fth = int(score.get("home", score.get("homeTeam", 0)))
                fta = int(score.get("away", score.get("awayTeam", 0)))
            else:
                continue

            ftr = "H" if fth > fta else ("A" if fta > fth else "D")

            # Date normalisation
            if isinstance(dt, (datetime, date)):
                date_str = dt.strftime("%d-%m-%Y")
            elif isinstance(dt, str):
                for fmt in ("%d.%m.%Y", "%Y-%m-%d", "%d/%m/%Y", "%d-%m-%Y"):
                    try:
                        date_str = datetime.strptime(dt[:10], fmt).strftime("%d-%m-%Y")
                        break
                    except ValueError:
                        continue
                else:
                    continue
            else:
                continue

            # Team name: use .name or .shortName attribute
            home_name = getattr(home, "name",      None) or \
                        getattr(home, "shortName",  None) or str(home)
            away_name = getattr(away, "name",      None) or \
                        getattr(away, "shortName",  None) or str(away)

            rows.append({
                "Division":   division,
                "MatchDate":  date_str,
                "MatchTime":  "",
                "HomeTeam":   home_name,
                "AwayTeam":   away_name,
                "HomeElo":    "",   "AwayElo":   "",
                "Form3Home":  "",   "Form5Home": "",
                "Form3Away":  "",   "Form5Away": "",
                "FTHome":     fth,  "FTAway":    fta,
                "FTResult":   ftr,
                "HTHome":     "",   "HTAway":    "",   "HTResult": "",
            })
        except Exception as exc:
            log.debug(f"  Skipping FS match: {exc}")
    return pd.DataFrame(rows)


def fetch_flashscore(date_from: str, leagues: list = None) -> pd.DataFrame:
    """
    Use flashscore_scraper (Playwright-based) to scrape match results.
    Requires: pip install flashscore_scraper playwright
              playwright install chromium
    """
    try:
        from flashscore_scraper.data_loaders import Football
    except ImportError:
        log.warning(
            "\n  flashscore_scraper not installed.\n"
            "  To enable UCL/UEL/UECL scraping, run:\n"
            "    pip install flashscore_scraper playwright\n"
            "    playwright install chromium\n"
        )
        return pd.DataFrame()

    if leagues is None:
        leagues = FLASHSCORE_EXTRA

    cutoff   = pd.to_datetime(date_from, dayfirst=False)
    all_rows = []
    loader   = Football()

    log.info(f"\n── Source 2: Flashscore  ({len(leagues)} competitions) ──")
    for country, league_slug, our_div in leagues:
        try:
            # flashscore_scraper Football loader — league name as displayed on flashscore
            matches = loader.load_matches(
                league=league_slug.replace("-", " ").title(),
                seasons=[f"{date.today().year - 1}/{date.today().year}",
                         f"{date.today().year}/{date.today().year + 1}"],
            )
            df = _parse_flashscore_df(matches, our_div)
            if df.empty:
                continue
            df["_dt"] = pd.to_datetime(df["MatchDate"], dayfirst=True, errors="coerce")
            df = df[df["_dt"] >= cutoff].drop(columns=["_dt"])
            if len(df):
                log.info(f"  {league_slug:30s} → {our_div:8s}  {len(df):4d} rows")
                all_rows.append(df)
        except Exception as e:
            log.warning(f"  {league_slug}: {e}")
        time.sleep(2)

    return pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame()


# ─────────────────────────────────────────────────────────────────────────────
# MERGE & DEDUPLICATE
# ─────────────────────────────────────────────────────────────────────────────

def _align_columns(df: pd.DataFrame, reference: pd.DataFrame) -> pd.DataFrame:
    """Add any missing columns (as empty strings) so concat works cleanly."""
    for col in reference.columns:
        if col not in df.columns:
            df[col] = ""
    return df[reference.columns]


def run_update(flashscore: bool = True):
    matches_path = DATA_DIR / "Matches.csv"
    if not matches_path.exists():
        log.error(f"Matches.csv not found at {matches_path}")
        return

    existing  = pd.read_csv(matches_path, low_memory=False)
    date_from = load_last_update()
    date_to   = datetime.utcnow().strftime("%Y-%m-%d")

    if date_from >= date_to:
        log.info("Data is already up to date.")
        return

    log.info(f"Updating  {date_from}  →  {date_to}")
    log.info(f"Existing rows: {len(existing):,}\n")

    frames = []

    # Source 1a: main leagues (football-data.co.uk)
    df_main = fetch_fduk_main(date_from)
    if not df_main.empty:
        frames.append(df_main)

    # Source 1b: extra leagues (football-data.co.uk /new/)
    df_extra = fetch_fduk_extra(date_from)
    if not df_extra.empty:
        frames.append(df_extra)

    # Source 2: flashscore (UCL/UEL/UECL + more)
    if flashscore:
        df_fs = fetch_flashscore(date_from)
        if not df_fs.empty:
            frames.append(df_fs)

    if not frames:
        log.info("No new data fetched.")
        save_last_update(date_to)
        return

    new_df = pd.concat(frames, ignore_index=True)
    new_df = _align_columns(new_df, existing)

    combined = pd.concat([existing, new_df], ignore_index=True)
    combined = combined.drop_duplicates(
        subset=["MatchDate", "HomeTeam", "AwayTeam"], keep="first"
    )
    net = len(combined) - len(existing)
    log.info(f"\nNet new rows: +{net:,}  (total: {len(combined):,})")
    combined.to_csv(matches_path, index=False)
    save_last_update(date_to)

    log.info("Rebuilding feature cache …")
    from data_processor import build_dataset
    build_dataset()
    log.info("✅  Update complete.")


def load_last_update() -> str:
    if CACHE_FILE.exists():
        return json.loads(CACHE_FILE.read_text()).get("last_date", "")
    return (datetime.utcnow() - timedelta(days=30)).strftime("%Y-%m-%d")


def save_last_update(date_str: str):
    CACHE_FILE.write_text(json.dumps({
        "last_date":  date_str,
        "updated_at": datetime.utcnow().isoformat(),
    }))


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--no-flashscore", action="store_true",
                   help="Skip flashscore scraping (domestic leagues only)")
    args = p.parse_args()
    run_update(flashscore=not args.no_flashscore)
