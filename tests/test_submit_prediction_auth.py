import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "submit_prediction.py"


def load_submit_prediction_module():
    spec = importlib.util.spec_from_file_location("submit_prediction", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class FakeDriver:
    def __init__(self):
        self.visited = []
        self.scripts = []

    def get(self, url):
        self.visited.append(url)

    def execute_script(self, script, *args):
        self.scripts.append((script, args))


class FakeElement:
    def __init__(self, **attrs):
        self.attrs = attrs

    def get_attribute(self, name):
        return self.attrs.get(name)


class SubmitPredictionAuthTest(unittest.TestCase):
    def test_load_local_storage_sets_string_and_structured_values(self) -> None:
        module = load_submit_prediction_module()
        driver = FakeDriver()

        with tempfile.TemporaryDirectory() as tmp:
            storage_path = Path(tmp) / "storage.json"
            storage_path.write_text(
                json.dumps(
                    {
                        "auth": "token-value",
                        "createdAuth": {"token": "created-token"},
                        "currentRoleId": 123,
                        "empty": None,
                    }
                ),
                encoding="utf-8",
            )

            loaded = module.load_local_storage_to_driver(driver, storage_path)

        self.assertEqual(loaded, 3)
        self.assertEqual(driver.visited, ["https://reg.aicomp.cn"])
        self.assertEqual(
            [args for _, args in driver.scripts],
            [
                ("auth", "token-value"),
                ("createdAuth", '{"token": "created-token"}'),
                ("currentRoleId", "123"),
            ],
        )

    def test_load_local_storage_derives_current_role_id(self) -> None:
        module = load_submit_prediction_module()
        driver = FakeDriver()

        with tempfile.TemporaryDirectory() as tmp:
            storage_path = Path(tmp) / "storage.json"
            storage_path.write_text(
                json.dumps({"localStorage": {"createdRoleId": "XueSheng"}}),
                encoding="utf-8",
            )

            loaded = module.load_local_storage_to_driver(driver, storage_path)

        self.assertEqual(loaded, 2)
        self.assertEqual(
            [args for _, args in driver.scripts],
            [
                ("createdRoleId", "XueSheng"),
                ("currentRoleId", "XueSheng"),
            ],
        )

    def test_load_local_storage_accepts_wrapped_local_storage_export(self) -> None:
        module = load_submit_prediction_module()
        driver = FakeDriver()

        with tempfile.TemporaryDirectory() as tmp:
            storage_path = Path(tmp) / "storage.json"
            storage_path.write_text(
                json.dumps({"localStorage": {"auth": "token-value"}}),
                encoding="utf-8",
            )

            loaded = module.load_local_storage_to_driver(driver, storage_path)

        self.assertEqual(loaded, 1)
        self.assertEqual(driver.scripts[0][1], ("auth", "token-value"))

    def test_choose_zip_file_input_prefers_zip_specific_input(self) -> None:
        module = load_submit_prediction_module()

        inputs = [
            FakeElement(accept=".jpeg,.jpg,.png"),
            FakeElement(accept=".zip,.rar"),
            FakeElement(accept=".pdf"),
            FakeElement(accept=".jpeg,.zip,.rar,.7z"),
        ]

        self.assertIs(module.choose_zip_file_input(inputs), inputs[1])

    def test_classify_submission_page_distinguishes_unconfirmed(self) -> None:
        module = load_submit_prediction_module()

        status, _message, success = module.classify_submission_page("页面没有明显结果")

        self.assertEqual(status, module.STATUS_SUBMIT_CLICKED_UNCONFIRMED)
        self.assertFalse(success)

    def test_classify_submission_page_detects_success_and_error(self) -> None:
        module = load_submit_prediction_module()

        status, _message, success = module.classify_submission_page("提交成功")
        self.assertEqual(status, module.STATUS_SUBMIT_CONFIRMED)
        self.assertTrue(success)

        status, _message, success = module.classify_submission_page("上传失败")
        self.assertEqual(status, module.STATUS_UPLOAD_VALIDATION_FAILED)
        self.assertFalse(success)


if __name__ == "__main__":
    unittest.main()
