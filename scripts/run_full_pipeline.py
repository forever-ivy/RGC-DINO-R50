#!/usr/bin/env python3
"""
Example: Complete automated training and submission workflow.

This script demonstrates how to integrate training, inference, submission,
and leaderboard checking into one automated pipeline.
"""

import argparse
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


def run_command(cmd, description, timeout=None):
    """Run a command and check for success."""
    print(f"\n{'='*70}")
    print(f"{description}")
    print(f"Command: {' '.join(str(c) for c in cmd)}")
    print(f"{'='*70}\n")

    try:
        result = subprocess.run(cmd, timeout=timeout, check=True)
        print(f"\n✓ {description} - SUCCESS")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n✗ {description} - FAILED (exit code {e.returncode})", file=sys.stderr)
        return False
    except subprocess.TimeoutExpired:
        print(f"\n✗ {description} - TIMEOUT", file=sys.stderr)
        return False


def main():
    parser = argparse.ArgumentParser(
        description='Complete automated training and submission workflow'
    )
    parser.add_argument(
        '--config',
        type=Path,
        default=Path('configs/default.yaml'),
        help='Training config file'
    )
    parser.add_argument(
        '--checkpoint',
        type=Path,
        help='Use existing checkpoint instead of training'
    )
    parser.add_argument(
        '--skip-training',
        action='store_true',
        help='Skip training step (use existing checkpoint)'
    )
    parser.add_argument(
        '--cookies',
        type=Path,
        help='Cookies file for authenticated submission'
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
        '--auto-submit',
        action='store_true',
        help='Automatically submit to competition'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Print what would be done without executing'
    )

    args = parser.parse_args()

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_base = Path('outputs') / f'run_{timestamp}'
    output_base.mkdir(parents=True, exist_ok=True)

    predictions_dir = output_base / 'predictions'
    submission_zip = output_base / f'submission_{timestamp}.zip'

    print(f"\n{'#'*70}")
    print(f"# Automated Training & Submission Pipeline")
    print(f"# Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"# Output: {output_base}")
    print(f"{'#'*70}\n")

    steps = []

    # Step 1: Training (optional)
    if not args.skip_training:
        checkpoint_path = output_base / 'checkpoints' / 'best_model.pth'
        steps.append({
            'name': 'Train Model',
            'cmd': [
                'python', 'scripts/train_baseline.py',
                '--config', str(args.config),
                '--output-dir', str(output_base / 'checkpoints'),
            ],
            'timeout': None,  # Training can take a long time
        })
    else:
        checkpoint_path = args.checkpoint
        if not checkpoint_path or not checkpoint_path.exists():
            print(f"ERROR: Checkpoint not found: {checkpoint_path}", file=sys.stderr)
            sys.exit(1)

    # Step 2: Generate Predictions
    steps.append({
        'name': 'Generate Predictions',
        'cmd': [
            'python', 'scripts/infer_baseline.py',
            '--model-path', str(checkpoint_path),
            '--dataset-root', 'source/AIC2026_PHASE_1_1000',
            '--output-dir', str(predictions_dir),
        ],
        'timeout': 3600,  # 1 hour
    })

    # Step 3: Package Submission
    steps.append({
        'name': 'Package Submission',
        'cmd': [
            'python', 'scripts/make_submission.py',
            '--dataset-root', 'source/AIC2026_PHASE_1_1000',
            '--submission-dir', str(predictions_dir),
            '--zip-path', str(submission_zip),
        ],
        'timeout': 300,  # 5 minutes
    })

    # Step 4: Submit (optional)
    if args.auto_submit:
        submit_cmd = [
            'python', 'scripts/submit_prediction.py',
            str(submission_zip),
        ]

        if args.cookies:
            submit_cmd.extend(['--cookies', str(args.cookies)])

        if args.user_data_dir:
            submit_cmd.extend(['--user-data-dir', str(args.user_data_dir)])

        if args.local_storage:
            submit_cmd.extend(['--local-storage', str(args.local_storage)])

        submit_cmd.extend(['--log', str(output_base / 'submission_log.jsonl')])

        steps.append({
            'name': 'Submit to Competition',
            'cmd': submit_cmd,
            'timeout': 300,  # 5 minutes
        })

    # Step 5: Check Leaderboard
    if args.auto_submit:
        check_cmd = [
            'python', 'scripts/check_leaderboard.py',
            '--output', str(output_base / 'rank.json'),
        ]

        if args.cookies:
            check_cmd.extend(['--cookies', str(args.cookies)])

        steps.append({
            'name': 'Check Leaderboard (wait 1+ hour after submission)',
            'cmd': check_cmd,
            'timeout': 60,
        })

    # Execute pipeline
    if args.dry_run:
        print("\n[DRY RUN MODE - Commands that would be executed:]\n")
        for i, step in enumerate(steps, 1):
            print(f"{i}. {step['name']}")
            print(f"   {' '.join(str(c) for c in step['cmd'])}\n")
        sys.exit(0)

    # Run each step
    failed = False
    for i, step in enumerate(steps, 1):
        print(f"\n\n{'='*70}")
        print(f"STEP {i}/{len(steps)}: {step['name']}")
        print(f"{'='*70}")

        success = run_command(
            step['cmd'],
            step['name'],
            timeout=step.get('timeout')
        )

        if not success:
            print(f"\n✗✗✗ Pipeline FAILED at step {i}: {step['name']} ✗✗✗\n")
            failed = True
            break

        # Wait between steps
        if i < len(steps):
            time.sleep(2)

    # Final summary
    print(f"\n\n{'#'*70}")
    if failed:
        print(f"# Pipeline FAILED")
        print(f"# Check logs in: {output_base}")
    else:
        print(f"# Pipeline COMPLETED SUCCESSFULLY")
        print(f"# Outputs saved to: {output_base}")
        if submission_zip.exists():
            print(f"# Submission ZIP: {submission_zip}")
            print(f"# Size: {submission_zip.stat().st_size / 1024 / 1024:.2f} MB")
    print(f"# Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'#'*70}\n")

    sys.exit(1 if failed else 0)


if __name__ == '__main__':
    main()
