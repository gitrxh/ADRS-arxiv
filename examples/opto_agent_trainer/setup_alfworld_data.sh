#!/bin/bash
# Helper script to download ALFWorld data
# Run this when network is available

set -e

export ALFWORLD_DATA=$HOME/data/alfworld
mkdir -p $ALFWORLD_DATA

echo "Downloading ALFWorld game files..."
echo "This requires network access to github.com"

# Method 1: Use alfworld-download (preferred)
alfworld-download -f

# If alfworld-download fails, try manual download:
# wget "https://github.com/alfworld/alfworld/releases/download/0.2.2/json_2.1.1_json.zip" -O /tmp/alfworld_json.zip
# unzip /tmp/alfworld_json.zip -d $ALFWORLD_DATA/
# wget "https://github.com/alfworld/alfworld/releases/download/0.2.2/alfred.pddl" -O $ALFWORLD_DATA/logic/alfred.pddl
# wget "https://github.com/alfworld/alfworld/releases/download/0.2.2/alfred.twl2" -O $ALFWORLD_DATA/logic/alfred.twl2

echo ""
echo "Verifying data..."
ls $ALFWORLD_DATA/json_2.1.1/train/ | head -5
echo ""
echo "ALFWorld data setup complete!"
echo "Expected structure:"
echo "  $ALFWORLD_DATA/json_2.1.1/train/"
echo "  $ALFWORLD_DATA/json_2.1.1/valid_seen/"
echo "  $ALFWORLD_DATA/json_2.1.1/valid_unseen/"
echo "  $ALFWORLD_DATA/logic/alfred.pddl"
echo "  $ALFWORLD_DATA/logic/alfred.twl2"
