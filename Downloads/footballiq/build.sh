#!/usr/bin/env bash
set -e

echo "============================================"
echo "  FootballIQ  -  Build Script"
echo "  RAM limit: 512MB (Render free tier)"
echo "============================================"

echo ""
echo "[1/4] Installing dependencies..."
pip install -r requirements.txt

echo ""
echo "[2/4] Downloading latest match data..."
python scripts/fetch_data.py

echo ""
echo "[3/4] Building features..."
python scripts/build_features.py

echo ""
echo "[4/4] Training models (memory-optimised)..."
python scripts/train.py

echo ""
echo "============================================"
echo "  Build complete ✓"
echo "============================================"
