#!/usr/bin/env python3
"""
scheduler.py  -  Daily auto-update cron job.
Fetches latest results then retrains models.

Render Cron Job command:  python scripts/scheduler.py
Schedule:                 0 4 * * *  (4 AM UTC daily)
"""
import sys, subprocess, logging
from pathlib import Path
from datetime import datetime

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("scheduler")

ROOT = Path(__file__).resolve().parent.parent

def run(script):
    log.info(f"Running {script.name}...")
    r = subprocess.run([sys.executable, str(script)],
                       capture_output=True, text=True)
    if r.stdout: log.info(r.stdout[-500:])
    if r.returncode != 0:
        log.error(f"FAILED:\n{r.stderr[-300:]}")
        return False
    return True

if __name__ == "__main__":
    log.info(f"=== Scheduler started {datetime.utcnow()} UTC ===")
    if run(ROOT / "scripts" / "fetch_data.py"):
        run(ROOT / "scripts" / "train.py")
    log.info("=== Scheduler finished ===")
