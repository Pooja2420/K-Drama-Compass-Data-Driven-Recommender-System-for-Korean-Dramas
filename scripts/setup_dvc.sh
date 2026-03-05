#!/bin/bash
# DVC setup script for K-Drama Compass
# Run once after cloning the repo:  bash scripts/setup_dvc.sh

set -e

echo "Installing DVC..."
pip install dvc

echo "Initializing DVC..."
dvc init

echo "Adding data/raw to DVC tracking..."
dvc add data/raw/korean_drama.csv
dvc add data/raw/reviews.csv
dvc add data/raw/wiki_actors.csv

echo "Adding data/processed to DVC tracking..."
dvc add data/processed

echo ""
echo "DVC setup complete."
echo "Next steps:"
echo "  1. Configure a DVC remote:  dvc remote add -d myremote s3://your-bucket/kdrama"
echo "  2. Push data:               dvc push"
echo "  3. Commit .dvc files to git (do NOT commit the CSVs themselves)"
