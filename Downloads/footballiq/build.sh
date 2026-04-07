#!/usr/bin/env bash
set -e

echo "============================================"
echo "  FootballIQ  -  Build Script"
echo "============================================"

echo ""
echo "[1/3] Installing dependencies..."
pip install -r requirements.txt

echo ""
echo "[2/3] Downloading latest match data..."
python scripts/fetch_data.py

echo ""
echo "[3/3] Training models..."
python scripts/train.py

echo ""
echo "============================================"
echo "  Build complete ✓"
echo "============================================"
