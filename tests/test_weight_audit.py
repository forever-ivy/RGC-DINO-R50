import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from rgc_dino.weight_audit import audit_load_report


ROOT = Path(__file__).resolve().parents[1]


class WeightAuditTest(unittest.TestCase):
    def test_audit_fails_on_fatal_backbone_missing_keys(self) -> None:
        audit = audit_load_report(
            missing_keys=["backbone.patch_embed.proj.weight", "class_embed.0.weight"],
            fatal_missing_patterns=["patch_embed"],
            allowed_missing_patterns=["class_embed"],
        )

        self.assertFalse(audit.ok)
        self.assertEqual(audit.fatal_missing_keys, ("backbone.patch_embed.proj.weight",))

    def test_audit_allows_class_dependent_heads(self) -> None:
        audit = audit_load_report(
            missing_keys=["class_embed.0.weight", "label_enc.weight"],
            skipped_keys=["enc_out_class_embed.weight"],
            fatal_missing_patterns=["class_embed", "label_enc"],
            fatal_skipped_patterns=["enc_out_class_embed"],
            allowed_missing_patterns=["class_embed", "label_enc", "enc_out_class_embed"],
        )

        self.assertTrue(audit.ok)

    def test_cli_returns_nonzero_for_fatal_report(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            report = Path(tmp) / "report.json"
            report.write_text(
                json.dumps({"missing_keys": ["backbone.stages.0.weight"], "skipped_keys": []}),
                encoding="utf-8",
            )

            result = subprocess.run(
                [sys.executable, str(ROOT / "scripts" / "audit_weight_load_report.py"), str(report)],
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )

            self.assertEqual(result.returncode, 2)
            self.assertIn('"ok": false', result.stdout)


if __name__ == "__main__":
    unittest.main()
