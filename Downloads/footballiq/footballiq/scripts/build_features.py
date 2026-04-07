"""
build_features.py  -  Memory-optimised feature engineering.
Processes data in chunks to stay under 512MB RAM.
Called by build.sh after fetch_data.py.
"""

import os, sys, gc, logging
from pathlib import Path

import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))
logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger("build_features")

DATA_DIR = Path(os.environ.get("DATA_DIR",
           str(Path(__file__).resolve().parent.parent / "data")))


def build():
    from data_processor import build_dataset
    log.info("Building features (memory-optimised)...")
    df, div_map = build_dataset()
    log.info(f"Features ready: {len(df):,} rows")
    del df; gc.collect()
    return True


if __name__ == "__main__":
    sys.exit(0 if build() else 1)
