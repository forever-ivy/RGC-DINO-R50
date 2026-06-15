# Competition Automation System - Summary

## What Has Been Created

I've built a complete automated system for your AIC2026 competition workflow. Here's what's included:

### 🛠️ Core Tools (3 Python Scripts)

1. **`scripts/check_leaderboard.py`** (278 lines)
   - Fetches your current rank and score from the leaderboard
   - Supports both cookies and HTML parsing
   - Saves history to track progress over time
   - Can run in quiet mode for scripting

2. **`scripts/submit_prediction.py`** (276 lines)
   - Automated browser-based submission using Selenium
   - Uploads ZIP files to competition platform
   - Supports headless mode for server deployment
   - Saves screenshots on errors for debugging
   - Logs all submissions with timestamps

3. **`scripts/monitor_competition.py`** (213 lines)
   - Continuous monitoring loop (checks every hour by default)
   - Tracks best rank and score achieved
   - Auto-submits new prediction ZIPs when detected
   - Maintains state across restarts
   - Designed for long-running background execution

### 🚀 Helper Scripts

4. **`scripts/setup_automation.sh`** (Bash script)
   - One-command setup for all dependencies
   - Checks system requirements
   - Creates necessary directories
   - Tests basic functionality

5. **`scripts/run_full_pipeline.py`** (212 lines)
   - Complete end-to-end pipeline
   - Trains model → generates predictions → packages → submits → checks rank
   - Can skip training and use existing checkpoints
   - Dry-run mode for testing

### 📚 Documentation

6. **`docs/COMPETITION_AUTOMATION.md`** - Comprehensive guide with:
   - Setup instructions
   - Tool usage examples
   - Workflow integration
   - Troubleshooting tips

7. **`docs/QUICK_START_AUTOMATION.md`** - Quick reference for:
   - Getting started in 5 minutes
   - Common commands
   - Integration examples

## How It Works

### Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Your Training Code                       │
│              (trains model, saves checkpoint)               │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ↓
┌─────────────────────────────────────────────────────────────┐
│              scripts/run_full_pipeline.py                   │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │  Generate    │→ │   Package    │→ │   Submit     │     │
│  │ Predictions  │  │ Submission   │  │  (Selenium)  │     │
│  └──────────────┘  └──────────────┘  └──────────────┘     │
└─────────────────────────────────────────────────────────────┘
                         │
                         ↓
┌─────────────────────────────────────────────────────────────┐
│          Competition Platform (reg.aicomp.cn)               │
│              Updates leaderboard every 1 hour               │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ↓
┌─────────────────────────────────────────────────────────────┐
│          scripts/monitor_competition.py (background)        │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │ Check Rank   │→ │ Track Best   │→ │ Auto-submit  │     │
│  │ Every Hour   │  │ Score/Rank   │  │  New ZIPs    │     │
│  └──────────────┘  └──────────────┘  └──────────────┘     │
└─────────────────────────────────────────────────────────────┘
```

### Authentication Methods

**Method 1: Cookie Export** (Easier, but expires)
- Export cookies from logged-in browser session
- Pass `--cookies outputs/cookies.json` to scripts
- Need to refresh periodically when cookies expire

**Method 2: Chrome Profile** (More stable, recommended)
- Use dedicated Chrome user data directory
- Login once, stays logged in
- Pass `--user-data-dir /data1/liuxuan/chrome-competition-profile`
- More reliable for long-running automation

## Quick Start

### 1. Setup (One Time)

```bash
source /data1/liuxuan/activate-py310.sh
bash scripts/setup_automation.sh
```

### 2. Export Cookies

Install "Get cookies.txt LOCALLY" browser extension, login to https://reg.aicomp.cn, export to `outputs/cookies.json`

### 3. Test Tools

```bash
# Check your current rank
python scripts/check_leaderboard.py --cookies outputs/cookies.json

# Test submission (when you have a real ZIP)
python scripts/submit_prediction.py outputs/submission.zip --cookies outputs/cookies.json
```

### 4. Start Background Monitor

```bash
# Run in tmux for persistent background execution
tmux new -s competition
source /data1/liuxuan/activate-py310.sh
python scripts/monitor_competition.py \
  --cookies outputs/cookies.json \
  --auto-submit \
  --predictions-dir outputs \
  --check-interval 3600

# Detach: Ctrl+B, then D
# Reattach: tmux attach -t competition
```

## Integration with Your Training

### Option A: Manual Trigger

After training completes, manually package and submit:

```bash
python scripts/run_full_pipeline.py \
  --checkpoint outputs/best_model.pth \
  --skip-training \
  --auto-submit \
  --cookies outputs/cookies.json
```

### Option B: Automatic from Training Script

Add to your training code:

```python
# When validation improves
if val_map > best_map:
    save_checkpoint('outputs/best_model.pth')
    
    # Auto-submit
    subprocess.run([
        'python', 'scripts/run_full_pipeline.py',
        '--checkpoint', 'outputs/best_model.pth',
        '--skip-training', '--auto-submit',
        '--cookies', 'outputs/cookies.json'
    ])
```

### Option C: Monitor Watches for New ZIPs

Keep monitor running in background. When you create a new submission ZIP in `outputs/`, it auto-detects and submits:

```bash
# In background (tmux)
python scripts/monitor_competition.py --auto-submit --predictions-dir outputs

# In your workflow, just create ZIPs:
python scripts/make_submission.py \
  --submission-dir outputs/predictions \
  --zip-path outputs/submission_$(date +%Y%m%d_%H%M%S).zip
```

## Features

✅ **Automated leaderboard checking** - No manual browser refreshing  
✅ **Automated submission** - Upload ZIPs via Selenium browser automation  
✅ **Progress tracking** - History of all ranks and scores  
✅ **Best score tracking** - Automatically tracks your best performance  
✅ **Auto-submit new models** - Detects new prediction ZIPs and submits them  
✅ **Background execution** - Runs continuously via tmux/screen  
✅ **Error recovery** - Screenshots and logs for debugging  
✅ **Integration ready** - Easy to call from training scripts  

## Important Notes

### Competition Info
- **Team name:** 都是同龄人队
- **Leaderboard refresh:** Every 1 hour
- **Platform:** https://reg.aicomp.cn

### Security
- `cookies.json` contains sensitive authentication data
- Already added to `.gitignore` - DO NOT commit to git
- Cookie files expire - refresh them periodically
- Chrome profile method (`--user-data-dir`) is more stable

### Timing
- Leaderboard updates every 1 hour after submission
- Default check interval: 3600 seconds (1 hour)
- Allow 1-2 hours after submission before expecting rank update

### Dependencies
- Python packages: `requests`, `selenium`, `webdriver-manager`
- System: Chrome or Chromium browser + ChromeDriver
- Already handled by `setup_automation.sh`

## File Outputs

All outputs saved to `outputs/` directory:

```
outputs/
├── monitor/
│   ├── monitor_state.json          # Current state (best rank, last check time)
│   ├── leaderboard_history.jsonl   # Historical rank/score data
│   └── current_rank.json           # Latest rank check result
├── submission_log.jsonl            # All submission attempts
├── submission_error.png            # Screenshot when submission fails
└── cookies.json                    # Browser cookies (YOU create this)
```

## Next Steps

1. **Run setup:** `bash scripts/setup_automation.sh`
2. **Export cookies** from logged-in browser
3. **Test leaderboard check:** `python scripts/check_leaderboard.py --cookies outputs/cookies.json`
4. **Start monitor in tmux** for continuous background monitoring
5. **Integrate with your training pipeline** using one of the three options above

## Getting Help

- See `docs/COMPETITION_AUTOMATION.md` for detailed documentation
- See `docs/QUICK_START_AUTOMATION.md` for quick reference
- Check submission logs: `tail outputs/submission_log.jsonl`
- Check monitor state: `cat outputs/monitor/monitor_state.json`
- View leaderboard history: `cat outputs/monitor/leaderboard_history.jsonl`

## For Your AI Agent and Coworkers

This system allows Claude Code and your team to:
- Check competition status programmatically
- Automatically submit new model versions
- Track progress over time without manual browser checks
- Integrate submissions into training workflows
- Run everything on your Linux server in the background

The monitor can run 24/7 in tmux, checking every hour and auto-submitting new predictions, so you can focus on improving the model instead of manually managing submissions.
