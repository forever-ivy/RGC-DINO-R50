import importlib.util
import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "check_leaderboard.py"


def load_check_leaderboard_module():
    spec = importlib.util.spec_from_file_location("check_leaderboard", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class CheckLeaderboardTest(unittest.TestCase):
    def test_fetch_leaderboard_uses_jsphb_post_flow(self) -> None:
        module = load_check_leaderboard_module()
        calls = []

        class FakeResponse:
            def __init__(self, payload):
                self.payload = payload

            def raise_for_status(self):
                return None

            def json(self):
                return self.payload

        class FakeRequests:
            RequestException = Exception

            def post(self, url, **kwargs):
                calls.append((url, kwargs["json"]))
                if kwargs["json"]["type"] == "JSJD":
                    return FakeResponse(
                        {
                            "success": True,
                            "data": {
                                "JDXH_": "1,2",
                                "JDMC_": "初赛,复赛",
                            },
                        }
                    )
                return FakeResponse(
                    {
                        "success": True,
                        "data": [
                            {"XH_": "1", "TDMC_": "第一队", "FS_": "0.81234"},
                            {"XH_": "2", "TDMC_": "都是同龄人队", "FS_": "0.71234"},
                        ],
                    }
                )

        module.requests = FakeRequests()

        rows = module.fetch_leaderboard(cookies={"session": "redacted"})

        self.assertEqual(
            calls,
            [
                (
                    "https://jluat-smart-app-api.yuntu.cn/third/jsphb",
                    {
                        "type": "JSJD",
                        "bdId": "4832828643476639834",
                        "stbh": "4829238709759119425",
                    },
                ),
                (
                    "https://jluat-smart-app-api.yuntu.cn/third/jsphb",
                    {
                        "pageNo": 0,
                        "pageSize": 20,
                        "type": "JSDF",
                        "rwId": "4829238709759119407",
                        "stbh": "4829238709759119425",
                        "jd": "初赛",
                    },
                ),
            ],
        )
        self.assertEqual(
            rows,
            [
                {"rank": 1, "team_name": "第一队", "score": 0.81234},
                {"rank": 2, "team_name": "都是同龄人队", "score": 0.71234},
            ],
        )

    def test_fetch_leaderboard_paginates(self) -> None:
        module = load_check_leaderboard_module()
        pages = {
            0: [{"XH_": "1", "TDMC_": "第一队", "FS_": "0.8"}],
            1: [{"XH_": "2", "TDMC_": "都是同龄人队", "FS_": "0.7"}],
            2: [],
        }

        class FakeResponse:
            def __init__(self, payload):
                self.payload = payload

            def raise_for_status(self):
                return None

            def json(self):
                return self.payload

        class FakeRequests:
            RequestException = Exception

            def post(self, url, **kwargs):
                payload = kwargs["json"]
                if payload["type"] == "JSJD":
                    return FakeResponse({"data": {"JDMC_": "初赛"}})
                return FakeResponse({"data": pages[payload["pageNo"]]})

        module.requests = FakeRequests()

        result = module.fetch_leaderboard(page_size=1, max_pages=3, return_metadata=True)

        self.assertEqual(result["fetched_pages"], 3)
        self.assertEqual([row["team_name"] for row in result["teams"]], ["第一队", "都是同龄人队"])

    def test_parse_jsphb_entries_normalizes_empty_scores(self) -> None:
        module = load_check_leaderboard_module()

        entries = module.parse_jsphb_entries(
            [
                {"XH_": "3", "TDMC_": "都是同龄人队", "FS_": ""},
                {"XH_": "4", "TDMC_": "其他队", "FS_": None},
            ]
        )

        self.assertEqual(
            entries,
            [
                {"rank": 3, "team_name": "都是同龄人队", "score": None},
                {"rank": 4, "team_name": "其他队", "score": None},
            ],
        )

    def test_main_accepts_normalized_fetch_results(self) -> None:
        module = load_check_leaderboard_module()
        module.fetch_leaderboard = lambda cookies=None, page_size=20, max_pages=10, return_metadata=True: {
            "teams": [
                {"rank": 1, "team_name": "第一队", "score": 0.8},
                {"rank": 2, "team_name": "都是同龄人队", "score": 0.7},
            ],
            "fetched_pages": 1,
            "page_size": 20,
        }

        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "rank.json"
            history = Path(tmp) / "history.jsonl"
            argv = [
                "check_leaderboard.py",
                "--output",
                str(output),
                "--history",
                str(history),
            ]
            with patch.object(sys, "argv", argv), redirect_stdout(io.StringIO()):
                module.main()

            saved = json.loads(output.read_text(encoding="utf-8"))

        self.assertEqual(saved["rank"], 2)
        self.assertEqual(saved["team_name"], "都是同龄人队")
        self.assertEqual(saved["total_teams"], 2)
        self.assertEqual(saved["fetched_pages"], 1)


if __name__ == "__main__":
    unittest.main()
