## Quick Start Guide: Competition Automation

### 1. Initial Setup (One Time)

Run the setup script:
```bash
bash scripts/setup_automation.sh
```

This will:
- Install required Python packages (requests, selenium)
- Create output directories
- Check Chrome/Chromium installation
- Test basic functionality

### 2. Export Browser Cookies

**Method A: Browser Extension (Recommended)**
1. Install "Get cookies.txt LOCALLY" extension for Chrome/Firefox
2. Login to https://reg.aicomp.cn in your browser
3. Click extension icon → Export cookies
4. Save as `outputs/cookies.json`

**Method B: Use Chrome Profile (More Stable)**
```bash
# Login to competition in this Chrome instance:
google-chrome --user-data-dir=/data1/liuxuan/chrome-competition-profile

# Then use --user-data-dir flag with scripts
```

### 3. Test Tools

**Check Leaderboard:**
```bash
python scripts/check_leaderboard.py --cookies outputs/cookies.json
```

**Test Submission (with a test file):**
```bash
python scripts/submit_prediction.py outputs/test.zip \
  --cookies outputs/cookies.json \
  --no-headless  # Show browser for first test
```

### 4. Start Automated Monitor

**Basic monitoring (check every hour):**
```bash
python scripts/monitor_competition.py \
  --cookies outputs/cookies.json \
  --check-interval 3600
```

**With auto-submission:**
```bash
python scripts/monitor_competition.py \
  --user-data-dir /data1/liuxuan/chrome-competition-profile \
  --auto-submit \
  --predictions-dir outputs \
  --check-interval 3600
```

**Run in background with tmux:**
```bash
tmux new -s competition
source /data1/liuxuan/activate-py310.sh
python scripts/monitor_competition.py --auto-submit --user-data-dir /data1/liuxuan/chrome-competition-profile
# Press Ctrl+B, then D to detach
# Reattach with: tmux attach -t competition
```

### 5. Complete Training + Submission Pipeline

```bash
# Full pipeline: train → infer → package → submit → check rank
python scripts/run_full_pipeline.py \
  --config configs/default.yaml \
  --auto-submit \
  --user-data-dir /data1/liuxuan/chrome-competition-profile

# Or skip training and use existing checkpoint:
python scripts/run_full_pipeline.py \
  --checkpoint checkpoints/best_model.pth \
  --skip-training \
  --auto-submit \
  --cookies outputs/cookies.json
```

### 6. Manual Workflow

```bash
# 1. Generate predictions from model
python scripts/infer_baseline.py \
  --model-path checkpoints/best_model.pth \
  --dataset-root source/AIC2026_PHASE_1_1000 \
  --output-dir outputs/predictions_v1

# 2. Package submission
python scripts/make_submission.py \
  --dataset-root source/AIC2026_PHASE_1_1000 \
  --submission-dir outputs/predictions_v1 \
  --zip-path outputs/submission_v1.zip

# 3. Submit
python scripts/submit_prediction.py outputs/submission_v1.zip \
  --cookies outputs/cookies.json

# 4. Wait 1+ hour, then check rank
python scripts/check_leaderboard.py --cookies outputs/cookies.json
```

### 7. Integration with Training

Add to your training script:

```python
# At end of each epoch or when validation improves:
if val_map > best_map:
    # Save checkpoint
    torch.save(model.state_dict(), 'outputs/best_model.pth')
    
    # Generate submission automatically
    subprocess.run([
        'python', 'scripts/run_full_pipeline.py',
        '--checkpoint', 'outputs/best_model.pth',
        '--skip-training',
        '--auto-submit',
        '--cookies', 'outputs/cookies.json'
    ])
```

### Troubleshooting

**Cookies expire:**
- Re-export from browser
- Or use `--user-data-dir` method (more stable)

**Chrome not found:**
```bash
sudo apt install chromium-browser chromium-chromedriver
```

**Leaderboard check fails:**
- Verify URL is accessible: `curl -I https://reg.aicomp.cn`
- Check cookies are valid
- Try accessing leaderboard in browser first

**Submission fails:**
- Run with `--no-headless` to see browser
- Check screenshot: `outputs/submission_error.png`
- Verify ZIP file is valid: `unzip -t outputs/submission.zip`

### File Locations

- Leaderboard history: `outputs/monitor/leaderboard_history.jsonl`
- Submission log: `outputs/submission_log.jsonl`
- Monitor state: `outputs/monitor/monitor_state.json`
- Current rank: `outputs/current_rank.json`

### Team Info

- Team name: **都是同龄人队**
- Leaderboard updates: Every 1 hour
- Leaderboard URL: https://reg.aicomp.cn/special/phb/detail?id=4832828643476639834&rwId=4829238709759119407&stbh=4829238709759119425
