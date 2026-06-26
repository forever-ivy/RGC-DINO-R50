#!/bin/bash
# Quick setup script for competition automation tools

set -e

echo "==================================="
echo "Competition Automation Setup"
echo "==================================="
echo ""

# Activate environment
echo "1. Activating Python environment..."
source /data1/liuxuan/activate-py310.sh

# Install dependencies
echo ""
echo "2. Installing Python dependencies..."
pip install -q requests selenium webdriver-manager

# Create output directories
echo ""
echo "3. Creating output directories..."
mkdir -p outputs/monitor
mkdir -p outputs/submissions

# Check Chrome installation
echo ""
echo "4. Checking Chrome/Chromium installation..."
if command -v google-chrome &> /dev/null; then
    echo "   ✓ Found google-chrome: $(which google-chrome)"
elif command -v chromium-browser &> /dev/null; then
    echo "   ✓ Found chromium-browser: $(which chromium-browser)"
else
    echo "   ⚠ Chrome/Chromium not found!"
    echo "   Ask the user to provide a browser/runtime under /data1/liuxuan/; do not use sudo or apt on this server."
fi

# Make scripts executable
echo ""
echo "5. Making scripts executable..."
chmod +x scripts/check_leaderboard.py
chmod +x scripts/submit_prediction.py
chmod +x scripts/monitor_competition.py

# Test leaderboard check (without cookies)
echo ""
echo "6. Testing leaderboard check (may fail without login)..."
python scripts/check_leaderboard.py --output outputs/test_rank.json || true

echo ""
echo "==================================="
echo "Setup Complete!"
echo "==================================="
echo ""
echo "Next steps:"
echo ""
echo "1. Export browser cookies:"
echo "   - Install 'Get cookies.txt' browser extension"
echo "   - Login to https://reg.aicomp.cn"
echo "   - Export cookies to outputs/cookies.json"
echo ""
echo "2. Test leaderboard check:"
echo "   python scripts/check_leaderboard.py --cookies outputs/cookies.json"
echo ""
echo "3. Test submission (with a dummy ZIP):"
echo "   python scripts/submit_prediction.py outputs/test.zip --cookies outputs/cookies.json --no-headless"
echo ""
echo "4. Start automated monitor:"
echo "   python scripts/monitor_competition.py --cookies outputs/cookies.json --auto-submit"
echo ""
echo "See CLAUDE.md and docs/README.md for current competition automation guidance"
echo ""
