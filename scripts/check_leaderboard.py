#!/usr/bin/env python3
"""
Automated leaderboard checker for AIC2026 competition.
Fetches current rank and scores from the competition platform.
"""

import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import requests
except ImportError:
    print("ERROR: requests not installed. Run: pip install requests", file=sys.stderr)
    sys.exit(1)


LEADERBOARD_URL = "https://reg.aicomp.cn/special/phb/detail"
LEADERBOARD_API_URL = "https://jluat-smart-app-api.yuntu.cn/third/jsphb"
TEAM_NAME = "都是同龄人队"


def _parse_optional_float(value: Any) -> Optional[float]:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _parse_optional_int(value: Any) -> Optional[int]:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        try:
            return int(float(value))
        except (TypeError, ValueError):
            return None


def parse_jsphb_entries(entries: List[Dict]) -> List[Dict]:
    """Normalize /third/jsphb leaderboard rows into the script's common shape."""
    teams = []
    for entry in entries:
        team_name = entry.get("TDMC_") or entry.get("team_name")
        if not team_name:
            continue
        teams.append(
            {
                "rank": _parse_optional_int(entry.get("XH_") or entry.get("rank")),
                "team_name": str(team_name).strip(),
                "score": _parse_optional_float(entry.get("FS_") or entry.get("score")),
            }
        )
    return teams


def _first_stage_name(stage_payload: Dict) -> Optional[str]:
    data = stage_payload.get("data") if isinstance(stage_payload, dict) else None
    if not isinstance(data, dict):
        return None

    names = data.get("JDMC_")
    if isinstance(names, str):
        for name in names.split(","):
            name = name.strip()
            if name:
                return name

    return None


def fetch_leaderboard(
    competition_id: str = "4832828643476639834",
    rw_id: str = "4829238709759119407",
    stbh: str = "4829238709759119425",
    cookies: Optional[Dict[str, str]] = None,
) -> Dict:
    """
    Fetch leaderboard data from competition platform.

    Args:
        competition_id: Competition ID
        rw_id: Task ID
        stbh: Stage ID
        cookies: Optional cookies dict for authentication

    Returns:
        Dict containing leaderboard data
    """
    headers = {
        'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36',
        'Accept': 'application/json, text/plain, */*',
        'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
        'Content-Type': 'application/json',
    }

    try:
        stage_response = requests.post(
            LEADERBOARD_API_URL,
            json={
                'type': 'JSJD',
                'bdId': competition_id,
                'stbh': stbh,
            },
            headers=headers,
            cookies=cookies or {},
            timeout=30,
        )
        stage_response.raise_for_status()
        stage_payload = stage_response.json()
        stage_name = _first_stage_name(stage_payload)

        payload = {
            'pageNo': 0,
            'pageSize': 20,
            'type': 'JSDF',
            'rwId': rw_id,
            'stbh': stbh,
        }
        if stage_name:
            payload['jd'] = stage_name

        response = requests.post(
            LEADERBOARD_API_URL,
            json=payload,
            headers=headers,
            cookies=cookies or {},
            timeout=30,
        )
        response.raise_for_status()

        data = response.json()
        if isinstance(data, dict) and isinstance(data.get('data'), list):
            return parse_jsphb_entries(data['data'])
        return data

    except requests.RequestException as e:
        return {'error': str(e), 'status': 'failed'}
    except (json.JSONDecodeError, ValueError) as e:
        return {'error': f'Failed to parse leaderboard JSON: {e}', 'status': 'failed'}


def parse_leaderboard_html(html: str) -> List[Dict]:
    """
    Parse leaderboard data from HTML response.

    Args:
        html: HTML content

    Returns:
        List of team entries with rank, name, and score
    """
    teams = []

    # Try multiple patterns to extract leaderboard data
    # Pattern 1: Table rows
    table_pattern = r'<tr[^>]*>.*?<td[^>]*>(\d+)</td>.*?<td[^>]*>(.*?)</td>.*?<td[^>]*>([\d.]+)</td>'
    matches = re.findall(table_pattern, html, re.DOTALL)

    for match in matches:
        rank, name, score = match
        teams.append({
            'rank': int(rank),
            'team_name': name.strip(),
            'score': float(score),
        })

    # Pattern 2: JSON embedded in script tag
    json_pattern = r'var\s+leaderboardData\s*=\s*(\[.*?\]);'
    json_match = re.search(json_pattern, html, re.DOTALL)
    if json_match and not teams:
        try:
            data = json.loads(json_match.group(1))
            teams = data
        except json.JSONDecodeError:
            pass

    return teams


def find_team_rank(teams: List[Dict], team_name: str) -> Optional[Dict]:
    """
    Find our team's rank in the leaderboard.

    Args:
        teams: List of team entries
        team_name: Our team name

    Returns:
        Team entry dict or None
    """
    for team in teams:
        if team.get('team_name') == team_name or team_name in team.get('team_name', ''):
            return team
    return None


def load_cookies_from_file(cookie_file: Path) -> Dict[str, str]:
    """
    Load cookies from JSON file.

    Expected format:
    {
        "cookie_name": "cookie_value",
        ...
    }

    Or Netscape format (cookies.txt from browser extension).
    """
    if not cookie_file.exists():
        return {}

    try:
        with open(cookie_file, 'r', encoding='utf-8') as f:
            content = f.read().strip()

            # Try JSON format first
            if content.startswith('{'):
                return json.loads(content)

            # Try Netscape format
            cookies = {}
            for line in content.split('\n'):
                if line.startswith('#') or not line.strip():
                    continue
                parts = line.split('\t')
                if len(parts) >= 7:
                    cookies[parts[5]] = parts[6]
            return cookies

    except Exception as e:
        print(f"Warning: Failed to load cookies: {e}", file=sys.stderr)
        return {}


def save_leaderboard_history(
    rank_data: Dict,
    history_file: Path,
    max_entries: int = 1000
):
    """Save leaderboard check result to history file."""
    history = []

    if history_file.exists():
        with open(history_file, 'r', encoding='utf-8') as f:
            history = [json.loads(line) for line in f if line.strip()]

    entry = {
        'timestamp': datetime.now().isoformat(),
        'rank': rank_data.get('rank'),
        'score': rank_data.get('score'),
        'team_name': rank_data.get('team_name'),
        'total_teams': rank_data.get('total_teams'),
    }

    history.append(entry)
    history = history[-max_entries:]  # Keep last N entries

    with open(history_file, 'w', encoding='utf-8') as f:
        for entry in history:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')


def main():
    parser = argparse.ArgumentParser(
        description='Check AIC2026 competition leaderboard rank'
    )
    parser.add_argument(
        '--team-name',
        default=TEAM_NAME,
        help='Team name to search for (default: 都是同龄人队)'
    )
    parser.add_argument(
        '--cookies',
        type=Path,
        help='Path to cookies file (JSON or Netscape format)'
    )
    parser.add_argument(
        '--output',
        type=Path,
        help='Save current rank to JSON file'
    )
    parser.add_argument(
        '--history',
        type=Path,
        default=Path('outputs/leaderboard_history.jsonl'),
        help='Save history to JSONL file'
    )
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Only output rank number (for scripting)'
    )

    args = parser.parse_args()

    # Load cookies if provided
    cookies = {}
    if args.cookies:
        cookies = load_cookies_from_file(args.cookies)
        if not args.quiet:
            print(f"Loaded {len(cookies)} cookies from {args.cookies}")

    # Fetch leaderboard
    if not args.quiet:
        print(f"Fetching leaderboard...")

    data = fetch_leaderboard(cookies=cookies)

    if isinstance(data, dict) and data.get('status') == 'failed':
        print(f"ERROR: Failed to fetch leaderboard: {data.get('error')}", file=sys.stderr)
        sys.exit(1)

    # Parse response
    teams = []
    if 'html' in data:
        teams = parse_leaderboard_html(data['html'])
    elif isinstance(data, list):
        teams = data
    elif 'data' in data and isinstance(data['data'], list):
        teams = data['data']

    if not teams:
        if not args.quiet:
            print("Warning: Could not parse leaderboard data", file=sys.stderr)
            print("Raw response preview:", data.get('html', '')[:500] if 'html' in data else data)
        sys.exit(1)

    # Find our team
    our_team = find_team_rank(teams, args.team_name)

    if not our_team:
        if not args.quiet:
            print(f"Warning: Team '{args.team_name}' not found in leaderboard")
            print(f"Found {len(teams)} teams total")
        sys.exit(1)

    # Add total teams count
    our_team['total_teams'] = len(teams)

    # Output results
    if args.quiet:
        print(our_team.get('rank', 'N/A'))
    else:
        print(f"\n{'='*60}")
        print(f"Team: {our_team.get('team_name')}")
        print(f"Rank: {our_team.get('rank')} / {len(teams)}")
        print(f"Score: {our_team.get('score', 'N/A')}")
        print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*60}\n")

        # Show top 3 for context
        if len(teams) >= 3:
            print("Top 3 teams:")
            for i, team in enumerate(teams[:3], 1):
                marker = " ← YOU" if team.get('team_name') == args.team_name else ""
                print(f"  {i}. {team.get('team_name')}: {team.get('score', 'N/A')}{marker}")
            print()

    # Save to output file
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(our_team, f, ensure_ascii=False, indent=2)
        if not args.quiet:
            print(f"Saved rank data to {args.output}")

    # Save to history
    if args.history:
        args.history.parent.mkdir(parents=True, exist_ok=True)
        save_leaderboard_history(our_team, args.history)
        if not args.quiet:
            print(f"Saved to history: {args.history}")


if __name__ == '__main__':
    main()
