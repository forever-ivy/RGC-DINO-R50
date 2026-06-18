#!/usr/bin/env python3
"""
Automated competition workflow coordinator.
Monitors leaderboard, gates promoted candidates, and submits predictions.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rgc_dino.submission_manifest import file_sha256  # noqa: E402


@dataclass(frozen=True)
class CandidateDecision:
    zip_path: Path
    metadata_path: Optional[Path]
    sha256: Optional[str]
    eligible: bool
    reason: str
    metadata: dict


class CompetitionMonitor:
    """Coordinates automated competition workflow."""

    def __init__(
        self,
        output_dir: Path,
        cookies_file: Optional[Path] = None,
        user_data_dir: Optional[Path] = None,
        local_storage_file: Optional[Path] = None,
        check_interval: int = 3600,
        min_submit_interval_seconds: int = 3900,
        require_promotion: bool = True,
        accept_unconfirmed_submit: bool = False,
    ):
        self.output_dir = output_dir
        self.cookies_file = cookies_file
        self.user_data_dir = user_data_dir
        self.local_storage_file = local_storage_file
        self.check_interval = check_interval
        self.min_submit_interval_seconds = min_submit_interval_seconds
        self.require_promotion = require_promotion
        self.accept_unconfirmed_submit = accept_unconfirmed_submit
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.state_file = output_dir / 'monitor_state.json'
        self.state = self.load_state()

    def load_state(self) -> Dict:
        """Load monitor state from disk."""
        if self.state_file.exists():
            with open(self.state_file, 'r', encoding='utf-8') as f:
                state = json.load(f)
        else:
            state = {}
        defaults = {
            'last_check': None,
            'last_submission': None,  # backward-compatible timestamp field
            'last_submission_time': None,
            'last_submitted_zip': None,
            'last_submitted_zip_sha256': None,
            'best_rank': None,
            'best_score': None,
            'submission_count': 0,
            'submitted_sha256s': [],
            'attempted_sha256s': [],
            'ignored_sha256s': [],
            'cooldown_until': None,
            'last_submission_result': None,
        }
        for key, value in defaults.items():
            state.setdefault(key, value)
        if state.get('last_submission') and not state.get('last_submission_time'):
            state['last_submission_time'] = state['last_submission']
        return state

    def save_state(self):
        """Save monitor state to disk."""
        with open(self.state_file, 'w', encoding='utf-8') as f:
            json.dump(self.state, f, indent=2, ensure_ascii=False, sort_keys=True)

    def check_leaderboard(self) -> Optional[Dict]:
        """Check current leaderboard position."""
        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Checking leaderboard...")

        cmd = [sys.executable, str(ROOT / 'scripts' / 'check_leaderboard.py')]

        if self.cookies_file:
            cmd.extend(['--cookies', str(self.cookies_file)])

        cmd.extend([
            '--output', str(self.output_dir / 'current_rank.json'),
            '--history', str(self.output_dir / 'leaderboard_history.jsonl'),
        ])

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60, cwd=ROOT)

            if result.returncode == 0:
                rank_file = self.output_dir / 'current_rank.json'
                if rank_file.exists():
                    with open(rank_file, 'r', encoding='utf-8') as f:
                        rank_data = json.load(f)

                    self.state['last_check'] = datetime.now().isoformat()

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
        except Exception as e:  # noqa: BLE001 - monitor should keep running
            print(f"  ⚠ Error checking leaderboard: {e}")

        return None

    def submit_prediction(self, zip_path: Path) -> bool:
        """Submit prediction ZIP file."""
        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Submitting {zip_path.name}...")

        if not zip_path.exists():
            print(f"  ✗ ZIP file not found: {zip_path}")
            return False

        cmd = [sys.executable, str(ROOT / 'scripts' / 'submit_prediction.py'), str(zip_path)]

        if self.cookies_file:
            cmd.extend(['--cookies', str(self.cookies_file)])

        if self.user_data_dir:
            cmd.extend(['--user-data-dir', str(self.user_data_dir)])

        if self.local_storage_file:
            cmd.extend(['--local-storage', str(self.local_storage_file)])

        cmd.extend(['--log', str(self.output_dir / 'submission_log.jsonl')])
        if self.accept_unconfirmed_submit:
            cmd.append('--accept-unconfirmed')

        try:
            result = subprocess.run(cmd, timeout=300, cwd=ROOT)
            ok = result.returncode == 0
            if ok:
                print("  ✓ Submission successful")
            else:
                print("  ✗ Submission failed or unconfirmed")
            return ok

        except subprocess.TimeoutExpired:
            print("  ✗ Submission timed out")
            return False
        except Exception as e:  # noqa: BLE001
            print(f"  ✗ Error during submission: {e}")
            return False

    def find_latest_prediction(self, predictions_dir: Path) -> Optional[Path]:
        """Find the most recent prediction ZIP file (legacy helper)."""
        if not predictions_dir.exists():
            return None

        zip_files = list(predictions_dir.glob('*.zip'))
        if not zip_files:
            return None

        zip_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return zip_files[0]

    def scan_candidate_queue(self, predictions_dir: Path, *, queue_order: str = 'oldest') -> list[CandidateDecision]:
        """Return candidate decisions for ZIPs in the watched directory."""
        if not predictions_dir.exists():
            return []
        zip_files = sorted(predictions_dir.glob('*.zip'), key=lambda p: p.stat().st_mtime)
        if queue_order == 'newest':
            zip_files.reverse()
        return [self.evaluate_candidate(path) for path in zip_files]

    def evaluate_candidate(self, zip_path: Path) -> CandidateDecision:
        metadata_path = zip_path.with_suffix('.promotion.json')
        metadata: dict = {}
        if metadata_path.exists():
            try:
                metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
            except json.JSONDecodeError as exc:
                return CandidateDecision(zip_path, metadata_path, None, False, f'invalid promotion JSON: {exc}', {})
        elif self.require_promotion:
            return CandidateDecision(zip_path, None, None, False, 'missing .promotion.json sidecar', {})

        try:
            sha = file_sha256(zip_path)
        except FileNotFoundError:
            return CandidateDecision(zip_path, metadata_path if metadata_path.exists() else None, None, False, 'zip missing', metadata)

        if metadata.get('zip_sha256') and metadata['zip_sha256'] != sha:
            return CandidateDecision(zip_path, metadata_path, sha, False, 'zip SHA does not match promotion metadata', metadata)
        if metadata and metadata.get('ready_for_submit') is not True:
            return CandidateDecision(zip_path, metadata_path, sha, False, 'promotion metadata is not ready_for_submit', metadata)
        manifest_path = metadata.get('manifest_path') if metadata else None
        if self.require_promotion and (not manifest_path or not Path(manifest_path).exists()):
            return CandidateDecision(zip_path, metadata_path, sha, False, 'missing submission manifest', metadata)
        if sha in set(self.state.get('submitted_sha256s', [])):
            return CandidateDecision(zip_path, metadata_path, sha, False, 'already submitted by SHA', metadata)
        if sha in set(self.state.get('ignored_sha256s', [])):
            return CandidateDecision(zip_path, metadata_path, sha, False, 'ignored by startup baseline', metadata)
        cooldown_until = self.state.get('cooldown_until')
        if cooldown_until:
            try:
                if datetime.now() < datetime.fromisoformat(cooldown_until):
                    return CandidateDecision(zip_path, metadata_path, sha, False, f'in cooldown until {cooldown_until}', metadata)
            except ValueError:
                pass
        return CandidateDecision(zip_path, metadata_path if metadata_path.exists() else None, sha, True, 'eligible', metadata)

    def submit_candidate(self, decision: CandidateDecision) -> bool:
        if not decision.eligible or decision.sha256 is None:
            print(f"  Skip {decision.zip_path.name}: {decision.reason}")
            return False
        attempted = set(self.state.get('attempted_sha256s', []))
        attempted.add(decision.sha256)
        self.state['attempted_sha256s'] = sorted(attempted)
        ok = self.submit_prediction(decision.zip_path)
        now = datetime.now()
        self.state['last_submission_result'] = {
            'zip_path': str(decision.zip_path),
            'zip_sha256': decision.sha256,
            'attempted_at': now.isoformat(),
            'status': 'submitted' if ok else 'failed_or_unconfirmed',
            'reason': decision.metadata.get('reason'),
        }
        if ok:
            submitted = set(self.state.get('submitted_sha256s', []))
            submitted.add(decision.sha256)
            self.state['submitted_sha256s'] = sorted(submitted)
            self.state['last_submission'] = now.isoformat()
            self.state['last_submission_time'] = now.isoformat()
            self.state['last_submitted_zip'] = str(decision.zip_path)
            self.state['last_submitted_zip_sha256'] = decision.sha256
            self.state['submission_count'] += 1
            self.state['cooldown_until'] = (now + timedelta(seconds=self.min_submit_interval_seconds)).isoformat()
        self.save_state()
        return ok

    def mark_existing_predictions_seen(self, predictions_dir: Path):
        """Set a submission baseline so existing ZIP files are not auto-submitted."""
        self.state['last_submission'] = datetime.now().isoformat()
        self.state['last_submission_time'] = self.state['last_submission']
        ignored = set(self.state.get('ignored_sha256s', []))
        existing_zips = predictions_dir.glob('*.zip') if predictions_dir.exists() else []
        for zip_path in existing_zips:
            try:
                ignored.add(file_sha256(zip_path))
            except FileNotFoundError:
                pass
        self.state['ignored_sha256s'] = sorted(ignored)
        self.save_state()
        latest_zip = self.find_latest_prediction(predictions_dir)
        if latest_zip:
            print(f"  Existing ZIPs ignored through baseline: {latest_zip.name}")
        else:
            print("  No existing ZIPs found; baseline set for future predictions")

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

        if self.state['last_submission_time']:
            print(f"Last submission: {self.state['last_submission_time']}")
        if self.state.get('cooldown_until'):
            print(f"Cooldown until: {self.state['cooldown_until']}")

        print(f"Next check in: {self.check_interval // 60} minutes")
        print(f"{'='*70}\n")

    def process_candidate_queue(self, *, auto_submit: bool, predictions_dir: Optional[Path], queue_order: str = 'oldest') -> None:
        if not predictions_dir:
            return
        decisions = self.scan_candidate_queue(predictions_dir, queue_order=queue_order)
        if not decisions:
            print(f"  No prediction ZIPs found in {predictions_dir}")
            return
        for decision in decisions:
            marker = 'READY' if decision.eligible else 'SKIP'
            print(f"  [{marker}] {decision.zip_path.name}: {decision.reason}")
        if auto_submit:
            eligible = next((decision for decision in decisions if decision.eligible), None)
            if eligible is None:
                print("  No eligible promoted candidates to submit")
            else:
                self.submit_candidate(eligible)

    def run_monitor_loop(
        self,
        auto_submit: bool = False,
        predictions_dir: Optional[Path] = None,
        queue_order: str = 'oldest',
    ):
        """Run continuous monitoring loop."""
        print("\n🤖 Starting Competition Monitor")
        print(f"   Check interval: {self.check_interval // 60} minutes")
        print(f"   Auto-submit: {auto_submit}")
        print(f"   Require promotion: {self.require_promotion}")
        print(f"   Output directory: {self.output_dir}")
        print()

        iteration = 0

        while True:
            iteration += 1
            print(f"\n{'#'*70}")
            print(f"Iteration {iteration} - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"{'#'*70}")

            rank_data = self.check_leaderboard()

            if rank_data:
                print(f"  Current rank: {rank_data.get('rank')} / {rank_data.get('total_teams')}")
                print(f"  Current score: {rank_data.get('score')}")

            self.process_candidate_queue(
                auto_submit=auto_submit,
                predictions_dir=predictions_dir,
                queue_order=queue_order,
            )

            self.print_status()

            print(f"💤 Sleeping for {self.check_interval // 60} minutes...")
            time.sleep(self.check_interval)


def main():
    parser = argparse.ArgumentParser(
        description='Automated competition workflow monitor'
    )
    parser.add_argument('--output-dir', type=Path, default=Path('outputs/monitor'), help='Output directory for logs and state')
    parser.add_argument('--cookies', type=Path, help='Path to cookies JSON file')
    parser.add_argument('--user-data-dir', type=Path, help='Chrome user data directory')
    parser.add_argument('--local-storage', type=Path, help='JSON file with localStorage auth values for submission')
    parser.add_argument('--check-interval', type=int, default=3600, help='Leaderboard check interval in seconds')
    parser.add_argument('--auto-submit', action='store_true', help='Automatically submit eligible promoted predictions')
    parser.add_argument('--predictions-dir', type=Path, default=Path('outputs/submissions'), help='Directory to watch for new prediction ZIPs')
    parser.add_argument('--ignore-existing', action='store_true', help='Do not submit ZIP files that already exist when the monitor starts')
    parser.add_argument('--once', action='store_true', help='Run one leaderboard/candidate check and exit')
    parser.add_argument('--require-promotion', dest='require_promotion', action='store_true', default=True, help='Only submit ZIPs with promotion metadata and manifest')
    parser.add_argument('--allow-unpromoted', dest='require_promotion', action='store_false', help='Legacy mode: allow raw ZIPs without promotion metadata')
    parser.add_argument('--min-submit-interval-seconds', type=int, default=3900, help='Cooldown after a confirmed submission')
    parser.add_argument('--queue-order', choices=('oldest', 'newest'), default='oldest')
    parser.add_argument('--accept-unconfirmed-submit', action='store_true', help='Pass through to submit script and count unconfirmed clicks as success')

    args = parser.parse_args()

    monitor = CompetitionMonitor(
        output_dir=args.output_dir,
        cookies_file=args.cookies,
        user_data_dir=args.user_data_dir,
        local_storage_file=args.local_storage,
        check_interval=args.check_interval,
        min_submit_interval_seconds=args.min_submit_interval_seconds,
        require_promotion=args.require_promotion,
        accept_unconfirmed_submit=args.accept_unconfirmed_submit,
    )

    if args.auto_submit and args.ignore_existing:
        monitor.mark_existing_predictions_seen(args.predictions_dir)

    if args.once:
        rank_data = monitor.check_leaderboard()
        if rank_data:
            print(f"  Current rank: {rank_data.get('rank')} / {rank_data.get('total_teams')}")
            print(f"  Current score: {rank_data.get('score')}")
        monitor.process_candidate_queue(
            auto_submit=args.auto_submit,
            predictions_dir=args.predictions_dir,
            queue_order=args.queue_order,
        )
        monitor.print_status()
    else:
        try:
            monitor.run_monitor_loop(
                auto_submit=args.auto_submit,
                predictions_dir=args.predictions_dir,
                queue_order=args.queue_order,
            )
        except KeyboardInterrupt:
            print("\n\n⏹ Monitor stopped by user")
            monitor.print_status()


if __name__ == '__main__':
    main()
