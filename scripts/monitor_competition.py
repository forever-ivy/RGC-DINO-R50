#!/usr/bin/env python3
"""
Automated competition workflow coordinator.
Monitors leaderboard, triggers training, and submits predictions.
"""

import argparse
import json
import subprocess
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional


class CompetitionMonitor:
    """Coordinates automated competition workflow."""

    def __init__(
        self,
        output_dir: Path,
        cookies_file: Optional[Path] = None,
        user_data_dir: Optional[Path] = None,
        local_storage_file: Optional[Path] = None,
        check_interval: int = 3600,  # 1 hour
    ):
        self.output_dir = output_dir
        self.cookies_file = cookies_file
        self.user_data_dir = user_data_dir
        self.local_storage_file = local_storage_file
        self.check_interval = check_interval
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.state_file = output_dir / 'monitor_state.json'
        self.state = self.load_state()

    def load_state(self) -> Dict:
        """Load monitor state from disk."""
        if self.state_file.exists():
            with open(self.state_file, 'r') as f:
                return json.load(f)
        return {
            'last_check': None,
            'last_submission': None,
            'best_rank': None,
            'best_score': None,
            'submission_count': 0,
        }

    def save_state(self):
        """Save monitor state to disk."""
        with open(self.state_file, 'w') as f:
            json.dump(self.state, f, indent=2)

    def check_leaderboard(self) -> Optional[Dict]:
        """Check current leaderboard position."""
        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Checking leaderboard...")

        cmd = ['python', 'scripts/check_leaderboard.py']

        if self.cookies_file:
            cmd.extend(['--cookies', str(self.cookies_file)])

        cmd.extend([
            '--output', str(self.output_dir / 'current_rank.json'),
            '--history', str(self.output_dir / 'leaderboard_history.jsonl'),
        ])

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)

            if result.returncode == 0:
                rank_file = self.output_dir / 'current_rank.json'
                if rank_file.exists():
                    with open(rank_file, 'r') as f:
                        rank_data = json.load(f)

                    self.state['last_check'] = datetime.now().isoformat()

                    # Update best rank/score
                    if self.state['best_rank'] is None or rank_data['rank'] < self.state['best_rank']:
                        self.state['best_rank'] = rank_data['rank']
                        print(f"  🎉 New best rank: {rank_data['rank']}")

                    if self.state['best_score'] is None or rank_data['score'] > self.state['best_score']:
                        self.state['best_score'] = rank_data['score']
                        print(f"  🎉 New best score: {rank_data['score']}")

                    self.save_state()
                    return rank_data

            else:
                print(f"  ⚠ Leaderboard check failed: {result.stderr}")

        except subprocess.TimeoutExpired:
            print("  ⚠ Leaderboard check timed out")
        except Exception as e:
            print(f"  ⚠ Error checking leaderboard: {e}")

        return None

    def submit_prediction(self, zip_path: Path) -> bool:
        """Submit prediction ZIP file."""
        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Submitting {zip_path.name}...")

        if not zip_path.exists():
            print(f"  ✗ ZIP file not found: {zip_path}")
            return False

        cmd = ['python', 'scripts/submit_prediction.py', str(zip_path)]

        if self.cookies_file:
            cmd.extend(['--cookies', str(self.cookies_file)])

        if self.user_data_dir:
            cmd.extend(['--user-data-dir', str(self.user_data_dir)])

        if self.local_storage_file:
            cmd.extend(['--local-storage', str(self.local_storage_file)])

        cmd.extend(['--log', str(self.output_dir / 'submission_log.jsonl')])

        try:
            result = subprocess.run(cmd, timeout=300)

            if result.returncode == 0:
                print(f"  ✓ Submission successful")
                self.state['last_submission'] = datetime.now().isoformat()
                self.state['submission_count'] += 1
                self.save_state()
                return True
            else:
                print(f"  ✗ Submission failed")
                return False

        except subprocess.TimeoutExpired:
            print("  ✗ Submission timed out")
            return False
        except Exception as e:
            print(f"  ✗ Error during submission: {e}")
            return False

    def find_latest_prediction(self, predictions_dir: Path) -> Optional[Path]:
        """Find the most recent prediction ZIP file."""
        if not predictions_dir.exists():
            return None

        zip_files = list(predictions_dir.glob('*.zip'))
        if not zip_files:
            return None

        # Sort by modification time
        zip_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return zip_files[0]

    def mark_existing_predictions_seen(self, predictions_dir: Path):
        """Set a submission baseline so existing ZIP files are not auto-submitted."""
        self.state['last_submission'] = datetime.now().isoformat()
        self.save_state()
        latest_zip = self.find_latest_prediction(predictions_dir)
        if latest_zip:
            print(f"  Existing ZIPs ignored through baseline: {latest_zip.name}")
        else:
            print(f"  No existing ZIPs found; baseline set for future predictions")

    def print_status(self):
        """Print current monitor status."""
        print(f"\n{'='*70}")
        print(f"Competition Monitor Status - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*70}")

        if self.state['last_check']:
            print(f"Last leaderboard check: {self.state['last_check']}")

        if self.state['best_rank']:
            print(f"Best rank achieved: {self.state['best_rank']}")

        if self.state['best_score']:
            print(f"Best score achieved: {self.state['best_score']}")

        print(f"Total submissions: {self.state['submission_count']}")

        if self.state['last_submission']:
            print(f"Last submission: {self.state['last_submission']}")

        print(f"Next check in: {self.check_interval // 60} minutes")
        print(f"{'='*70}\n")

    def run_monitor_loop(self, auto_submit: bool = False, predictions_dir: Optional[Path] = None):
        """Run continuous monitoring loop."""
        print(f"\n🤖 Starting Competition Monitor")
        print(f"   Check interval: {self.check_interval // 60} minutes")
        print(f"   Auto-submit: {auto_submit}")
        print(f"   Output directory: {self.output_dir}")
        print()

        iteration = 0

        while True:
            iteration += 1
            print(f"\n{'#'*70}")
            print(f"Iteration {iteration} - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"{'#'*70}")

            # Check leaderboard
            rank_data = self.check_leaderboard()

            if rank_data:
                print(f"  Current rank: {rank_data.get('rank')} / {rank_data.get('total_teams')}")
                print(f"  Current score: {rank_data.get('score')}")

            # Auto-submit if enabled
            if auto_submit and predictions_dir:
                latest_zip = self.find_latest_prediction(predictions_dir)
                if latest_zip:
                    # Check if we've already submitted this file
                    last_submission_time = None
                    if self.state['last_submission']:
                        last_submission_time = datetime.fromisoformat(self.state['last_submission'])

                    file_mtime = datetime.fromtimestamp(latest_zip.stat().st_mtime)

                    if not last_submission_time or file_mtime > last_submission_time:
                        print(f"  📦 Found new prediction: {latest_zip.name}")
                        self.submit_prediction(latest_zip)
                    else:
                        print(f"  No new predictions since last submission")
                else:
                    print(f"  No prediction ZIPs found in {predictions_dir}")

            # Print status
            self.print_status()

            # Wait for next check
            print(f"💤 Sleeping for {self.check_interval // 60} minutes...")
            time.sleep(self.check_interval)


def main():
    parser = argparse.ArgumentParser(
        description='Automated competition workflow monitor'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('outputs/monitor'),
        help='Output directory for logs and state'
    )
    parser.add_argument(
        '--cookies',
        type=Path,
        help='Path to cookies JSON file'
    )
    parser.add_argument(
        '--user-data-dir',
        type=Path,
        help='Chrome user data directory'
    )
    parser.add_argument(
        '--local-storage',
        type=Path,
        help='JSON file with localStorage auth values for submission'
    )
    parser.add_argument(
        '--check-interval',
        type=int,
        default=3600,
        help='Leaderboard check interval in seconds (default: 3600 = 1 hour)'
    )
    parser.add_argument(
        '--auto-submit',
        action='store_true',
        help='Automatically submit new predictions'
    )
    parser.add_argument(
        '--predictions-dir',
        type=Path,
        default=Path('outputs'),
        help='Directory to watch for new prediction ZIPs'
    )
    parser.add_argument(
        '--ignore-existing',
        action='store_true',
        help='Do not submit ZIP files that already exist when the monitor starts'
    )
    parser.add_argument(
        '--once',
        action='store_true',
        help='Run once and exit (no loop)'
    )

    args = parser.parse_args()

    monitor = CompetitionMonitor(
        output_dir=args.output_dir,
        cookies_file=args.cookies,
        user_data_dir=args.user_data_dir,
        local_storage_file=args.local_storage,
        check_interval=args.check_interval,
    )

    if args.once:
        # Single check
        monitor.check_leaderboard()
        monitor.print_status()
    else:
        # Continuous monitoring
        try:
            if args.auto_submit and args.ignore_existing:
                monitor.mark_existing_predictions_seen(args.predictions_dir)
            monitor.run_monitor_loop(
                auto_submit=args.auto_submit,
                predictions_dir=args.predictions_dir,
            )
        except KeyboardInterrupt:
            print("\n\n⏹ Monitor stopped by user")
            monitor.print_status()


if __name__ == '__main__':
    main()
