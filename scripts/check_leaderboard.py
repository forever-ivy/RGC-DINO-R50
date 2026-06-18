#!/usr/bin/env python3
"""
Automated leaderboard checker for AIC2026 competition.
Fetches current rank and scores from the competition platform.
"""

from __future__ import annotations

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
    page_size: int = 20,
    max_pages: int = 1,
    return_metadata: bool = False,
) -> Dict | List[Dict]:
    """
    Fetch leaderboard data from competition platform.

    By default this preserves the historical API and returns a list of normalized
    rows. Set ``return_metadata=True`` to receive pagination metadata.
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

        teams: list[dict] = []
        fetched_pages = 0
        for page_no in range(max(1, max_pages)):
            payload = {
                'pageNo': page_no,
                'pageSize': page_size,
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
            fetched_pages += 1

            data = response.json()
            if not (isinstance(data, dict) and isinstance(data.get('data'), list)):
                if return_metadata:
                    return {
                        'teams': teams,
                        'status': 'ok',
                        'fetched_pages': fetched_pages,
                        'page_size': page_size,
                        'raw_response': data,
                    }
                return data
            page_rows = parse_jsphb_entries(data['data'])
            if not page_rows:
                break
            teams.extend(page_rows)
            if len(page_rows) < page_size:
                break

        if return_metadata:
            return {
                'teams': teams,
                'status': 'ok',
                'fetched_pages': fetched_pages,
                'page_size': page_size,
                'timestamp': datetime.now().isoformat(),
            }
        return teams

    except requests.RequestException as e:
        return {'error': str(e), 'status': 'failed'}
    except (json.JSONDecodeError, ValueError) as e:
        return {'error': f'Failed to parse leaderboard JSON: {e}', 'status': 'failed'}


def parse_leaderboard_html(html: str) -> List[Dict]:
    """Parse leaderboard data from HTML response."""
    teams = []
    table_pattern = r'<tr[^>]*>.*?<td[^>]*>(\d+)</td>.*?<td[^>]*>(.*?)</td>.*?<td[^>]*>([\d.]+)</td>'
    matches = re.findall(table_pattern, html, re.DOTALL)

    for match in matches:
        rank, name, score = match
        teams.append({
            'rank': int(rank),
            'team_name': name.strip(),
            'score': float(score),
        })

    json_pattern = r'var\s+leaderboardData\s*=\s*(\[.*?\]);'
    json_match = re.search(json_pattern, html, re.DOTALL)
    if json_match and not teams:
        try:
            teams = json.loads(json_match.group(1))
        except json.JSONDecodeError:
            pass

    return teams


def find_team_rank(teams: List[Dict], team_name: str) -> Optional[Dict]:
    """Find our team's rank in the leaderboard."""
    for team in teams:
        if team.get('team_name') == team_name or team_name in team.get('team_name', ''):
            return team
    return None


def load_cookies_from_file(cookie_file: Path) -> Dict[str, str]:
    """Load cookies from JSON or Netscape cookies.txt file."""
    if not cookie_file.exists():
        return {}

    try:
        with open(cookie_file, 'r', encoding='utf-8') as f:
            content = f.read().strip()

            if content.startswith('{'):
                return json.loads(content)

            cookies = {}
            for line in content.split('\n'):
                if line.startswith('#') or not line.strip():
                    continue
                parts = line.split('\t')
                if len(parts) >= 7:
                    cookies[parts[5]] = parts[6]
            return cookies

    except Exception as e:  # noqa: BLE001 - auth file parsing should warn only
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
        'page_found': rank_data.get('page_found'),
        'fetched_pages': rank_data.get('fetched_pages'),
    }

    history.append(entry)
    history = history[-max_entries:]

    with open(history_file, 'w', encoding='utf-8') as f:
        for entry in history:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')


def _extract_teams(data: Dict | List[Dict]) -> tuple[list[dict], dict]:
    metadata: dict = {}
    if isinstance(data, dict) and 'teams' in data:
        metadata = dict(data)
        teams = data.get('teams') if isinstance(data.get('teams'), list) else []
        return teams, metadata
    if isinstance(data, dict) and 'html' in data:
        return parse_leaderboard_html(data['html']), metadata
    if isinstance(data, list):
        return data, metadata
    if isinstance(data, dict) and 'data' in data and isinstance(data['data'], list):
        return data['data'], metadata
    return [], metadata


def main():
    parser = argparse.ArgumentParser(
        description='Check AIC2026 competition leaderboard rank'
    )
    parser.add_argument('--team-name', default=TEAM_NAME, help='Team name to search for')
    parser.add_argument('--cookies', type=Path, help='Path to cookies file (JSON or Netscape format)')
    parser.add_argument('--output', type=Path, help='Save current rank to JSON file')
    parser.add_argument('--history', type=Path, default=Path('outputs/leaderboard_history.jsonl'), help='Save history to JSONL file')
    parser.add_argument('--quiet', action='store_true', help='Only output rank number (for scripting)')
    parser.add_argument('--page-size', type=int, default=20, help='Leaderboard API page size')
    parser.add_argument('--max-pages', type=int, default=10, help='Maximum leaderboard pages to fetch')
    parser.add_argument('--strict-team-found', action='store_true', default=True, help='Exit nonzero when team is not found')
    parser.add_argument('--allow-missing-team', dest='strict_team_found', action='store_false', help='Exit zero even if team is not found')

    args = parser.parse_args()

    cookies = {}
    if args.cookies:
        cookies = load_cookies_from_file(args.cookies)
        if not args.quiet:
            print(f"Loaded {len(cookies)} cookies from {args.cookies}")

    if not args.quiet:
        print("Fetching leaderboard...")

    data = fetch_leaderboard(
        cookies=cookies,
        page_size=args.page_size,
        max_pages=args.max_pages,
        return_metadata=True,
    )

    if isinstance(data, dict) and data.get('status') == 'failed':
        print(f"ERROR: Failed to fetch leaderboard: {data.get('error')}", file=sys.stderr)
        sys.exit(1)

    teams, metadata = _extract_teams(data)

    if not teams:
        if not args.quiet:
            print("Warning: Could not parse leaderboard data", file=sys.stderr)
            print("Raw response preview:", data.get('html', '')[:500] if isinstance(data, dict) and 'html' in data else data)
        sys.exit(1)

    our_team = find_team_rank(teams, args.team_name)

    if not our_team:
        if not args.quiet:
            print(f"Warning: Team '{args.team_name}' not found in leaderboard")
            print(f"Fetched {len(teams)} teams across {metadata.get('fetched_pages', 'unknown')} page(s)")
        if args.strict_team_found:
            sys.exit(1)
        return

    our_team = dict(our_team)
    our_team['total_teams'] = len(teams)
    our_team['fetched_pages'] = metadata.get('fetched_pages')
    our_team['page_size'] = metadata.get('page_size')
    our_team['timestamp'] = datetime.now().isoformat()
    if our_team.get('rank') is not None and args.page_size:
        our_team['page_found'] = int((int(our_team['rank']) - 1) // args.page_size)

    if args.quiet:
        print(our_team.get('rank', 'N/A'))
    else:
        print(f"\n{'='*60}")
        print(f"Team: {our_team.get('team_name')}")
        print(f"Rank: {our_team.get('rank')} / {len(teams)}")
        print(f"Score: {our_team.get('score', 'N/A')}")
        print(f"Fetched pages: {our_team.get('fetched_pages')}")
        print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*60}\n")

        if len(teams) >= 3:
            print("Top 3 teams:")
            for i, team in enumerate(teams[:3], 1):
                marker = " ← YOU" if team.get('team_name') == args.team_name else ""
                print(f"  {i}. {team.get('team_name')}: {team.get('score', 'N/A')}{marker}")
            print()

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(our_team, f, ensure_ascii=False, indent=2)
        if not args.quiet:
            print(f"Saved rank data to {args.output}")

    if args.history:
        args.history.parent.mkdir(parents=True, exist_ok=True)
        save_leaderboard_history(our_team, args.history)
        if not args.quiet:
            print(f"Saved to history: {args.history}")


if __name__ == '__main__':
    main()
