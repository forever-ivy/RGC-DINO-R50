# Competition Automation Tools

This directory contains automated tools for monitoring and submitting to the AIC2026 competition.

## Setup

### 1. Install Dependencies

```bash
source /data1/liuxuan/activate-py310.sh
pip install requests selenium webdriver-manager
```

### 2. Install Chrome/Chromium (for submission automation)

```bash
# On Ubuntu/Debian
sudo apt install chromium-browser chromium-chromedriver

# Or use portable Chrome
wget https://dl.google.com/linux/direct/google-chrome-stable_current_amd64.deb
sudo dpkg -i google-chrome-stable_current_amd64.deb
```

### 3. Export Browser Cookies

For authenticated access, you need to export cookies from your logged-in browser session.

**Method 1: Browser Extension**
1. Install "Get cookies.txt" extension for Chrome/Firefox
2. Navigate to https://reg.aicomp.cn (make sure you're logged in)
3. Click the extension icon and export cookies
4. Save as `outputs/cookies.json`

**Method 2: Use Chrome User Data**
```bash
# Run Chrome with a custom profile directory
google-chrome --user-data-dir=/data1/liuxuan/chrome-profile

# Login to the competition platform
# Then use --user-data-dir flag with submission script
```

**Method 3: Manual Cookie Export** (if needed)
```python
# In browser console (F12), run:
copy(JSON.stringify(Object.fromEntries(document.cookie.split('; ').map(c => c.split('=', 2)))))

# Paste into outputs/cookies.json
```

## Tools

### 1. Check Leaderboard (`check_leaderboard.py`)

Check your current rank and score on the leaderboard.

```bash
# Basic check (may work without login)
python scripts/check_leaderboard.py

# With cookies for authenticated access
python scripts/check_leaderboard.py --cookies outputs/cookies.json

# Save rank history
python scripts/check_leaderboard.py \
  --cookies outputs/cookies.json \
  --output outputs/current_rank.json \
  --history outputs/leaderboard_history.jsonl

# Quiet mode (only output rank number, for scripting)
python scripts/check_leaderboard.py --quiet
```

### 2. Submit Predictions (`submit_prediction.py`)

Upload prediction ZIP file to competition platform.

```bash
# Submit with cookies
python scripts/submit_prediction.py outputs/my_predictions.zip \
  --cookies outputs/cookies.json

# Submit with Chrome user data directory (recommended)
python scripts/submit_prediction.py outputs/my_predictions.zip \
  --user-data-dir /data1/liuxuan/chrome-profile

# Show browser window for debugging
python scripts/submit_prediction.py outputs/my_predictions.zip \
  --cookies outputs/cookies.json \
  --no-headless
```

**Submission Log:**
- Success/failure status saved to `outputs/submission_log.jsonl`
- Screenshot saved to `outputs/submission_error.png` on errors

### 3. Automated Monitor (`monitor_competition.py`)

Continuously monitor leaderboard and optionally auto-submit new predictions.

```bash
# Monitor only (check leaderboard every hour)
python scripts/monitor_competition.py \
  --cookies outputs/cookies.json \
  --check-interval 3600

# Monitor with auto-submission
python scripts/monitor_competition.py \
  --cookies outputs/cookies.json \
  --user-data-dir /data1/liuxuan/chrome-profile \
  --auto-submit \
  --predictions-dir outputs \
  --check-interval 3600

# Single check (no loop)
python scripts/monitor_competition.py \
  --cookies outputs/cookies.json \
  --once
```

**Monitor Features:**
- Checks leaderboard every N seconds (default 3600 = 1 hour)
- Tracks best rank and score achieved
- Automatically submits new prediction ZIPs when detected
- Saves state to `outputs/monitor/monitor_state.json`
- Saves leaderboard history to `outputs/monitor/leaderboard_history.jsonl`

### 4. Run in Background with tmux/screen

For long-running monitoring on the server:

```bash
# Using tmux
tmux new -s competition-monitor
source /data1/liuxuan/activate-py310.sh
python scripts/monitor_competition.py \
  --user-data-dir /data1/liuxuan/chrome-profile \
  --auto-submit \
  --predictions-dir outputs
# Detach: Ctrl+B, then D
# Reattach: tmux attach -t competition-monitor

# Using screen
screen -S competition-monitor
source /data1/liuxuan/activate-py310.sh
python scripts/monitor_competition.py --auto-submit
# Detach: Ctrl+A, then D
# Reattach: screen -r competition-monitor

# Using nohup
nohup python scripts/monitor_competition.py --auto-submit > monitor.log 2>&1 &
tail -f monitor.log
```

## Workflow Integration

### Automated Training + Submission Pipeline

```bash
# 1. Train model
python scripts/train_baseline.py --config configs/default.yaml

# 2. Generate predictions
python scripts/infer_baseline.py \
  --model-path checkpoints/best_model.pth \
  --dataset-root source/AIC2026_PHASE_1_1000 \
  --output-dir outputs/predictions_v1

# 3. Package submission
python scripts/make_submission.py \
  --dataset-root source/AIC2026_PHASE_1_1000 \
  --submission-dir outputs/predictions_v1 \
  --zip-path outputs/submission_v1_$(date +%Y%m%d_%H%M%S).zip

# 4. Submit automatically (if monitor is running with --auto-submit)
# Or manually:
python scripts/submit_prediction.py outputs/submission_v1_*.zip \
  --user-data-dir /data1/liuxuan/chrome-profile

# 5. Wait 1+ hour for leaderboard update, then check
python scripts/check_leaderboard.py --cookies outputs/cookies.json
```

### Integration with Claude Code

Add to your training script:

```python
# At end of training script
def auto_submit_if_improved(checkpoint_path, predictions_dir):
    """Submit if validation score improved."""
    import subprocess
    
    # Generate predictions
    subprocess.run([
        'python', 'scripts/infer_baseline.py',
        '--model-path', checkpoint_path,
        '--output-dir', predictions_dir
    ])
    
    # Package submission
    zip_path = f'outputs/submission_{datetime.now().strftime("%Y%m%d_%H%M%S")}.zip'
    subprocess.run([
        'python', 'scripts/make_submission.py',
        '--submission-dir', predictions_dir,
        '--zip-path', zip_path
    ])
    
    print(f"✓ Created submission: {zip_path}")
    print("  Monitor will auto-submit if --auto-submit is enabled")
```

## Leaderboard Data

Our team: **都是同龄人队**

Leaderboard URL: https://reg.aicomp.cn/special/phb/detail?id=4832828643476639834&rwId=4829238709759119407&stbh=4829238709759119425

Submission URL: https://reg.aicomp.cn/app/JSGLPT/639980063d903c241eb85102

**Update Frequency:** Every 1 hour

## Troubleshooting

### Leaderboard Check Fails

1. Check if you can access the leaderboard URL in browser
2. Verify cookies are valid and not expired
3. Try refreshing cookies from browser
4. Check network connectivity: `curl -I https://reg.aicomp.cn`

### Submission Fails

1. Run with `--no-headless` to see browser window
2. Check screenshot saved to `outputs/submission_error.png`
3. Verify Chrome/Chromium is installed: `which google-chrome chromium-browser`
4. Test manual upload in browser first
5. Check submission log: `tail outputs/submission_log.jsonl`

### Chrome Driver Issues

```bash
# Install ChromeDriver manually
pip install webdriver-manager
```

Or specify custom Chrome binary:
```bash
python scripts/submit_prediction.py submission.zip \
  --chrome-binary /usr/bin/chromium-browser
```

### Session Expired

If cookies expire, you need to:
1. Login to competition platform in browser
2. Re-export cookies
3. Or use `--user-data-dir` with persistent Chrome profile

## Security Notes

- **Do not commit cookies.json to git** (already in .gitignore)
- Cookies expire - refresh them periodically
- Chrome user data directory is safer than cookie export
- Monitor logs may contain sensitive URLs - keep them private
