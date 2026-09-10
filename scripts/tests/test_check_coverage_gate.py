#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Ambiq
# SPDX-License-Identifier: LicenseRef-Ambiq-Apollo-SDK
"""Coverage transition and continuing enforcement. Refs #484."""
import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("gate", ROOT / "scripts/ci/check_coverage_gate.py")
gate = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(gate)


class CoverageGateTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.current = Path(self.tmp.name) / "current.json"
        self.floor = Path(self.tmp.name) / "floor.json"
        self.config = {"line_floor_pct": 81.0, "baseline_reset": {"epoch": "sum", "line_rate": 83.44}}

    def run_gate(self, rate=83.44, previous=84.02, previous_epoch=None, stamp=True, token="test"):
        summary = {"overall_line_rate": rate, "sentinel": [1, 2]}
        self.current.write_text(json.dumps(summary))
        self.floor.write_text(json.dumps(self.config))
        argv = ["gate", "--summary-json", str(self.current), "--floor-file", str(self.floor), "--repo", "test/test"]
        baseline = {"overall_line_rate": previous}
        if previous_epoch is not None:
            baseline["baseline_epoch"] = previous_epoch
        output = io.StringIO()
        with patch.dict(os.environ, {"GITHUB_TOKEN": token, "GITHUB_STEP_SUMMARY": ""}), contextlib.redirect_stdout(output):
            with patch.object(gate, "fetch_baseline", return_value=(baseline, "remote baseline")) as fetch:
                if stamp:
                    with patch("sys.argv", argv + ["--stamp-epoch"]):
                        self.assertEqual(gate.main(), 0)
                    fetch.assert_not_called()
                    self.assertEqual(json.loads(self.current.read_text())["sentinel"], [1, 2])
                with patch("sys.argv", argv):
                    result = gate.main()
        return result, output.getvalue()

    def test_explicit_transition(self):
        result, output = self.run_gate()
        self.assertEqual(result, 0)
        self.assertIn("reviewed baseline reset sum: 83.44%", output)
        self.config.pop("baseline_reset")
        self.assertEqual(self.run_gate()[0], 1)

    def test_floor_and_tolerance_remain_enforced(self):
        for rate, expected in [(80.99, 1), (83.20, 1), (83.30, 0)]:
            with self.subTest(rate=rate):
                self.assertEqual(self.run_gate(rate=rate)[0], expected)

    def test_matching_epoch_resumes_main_ratchet(self):
        self.assertEqual(self.run_gate(previous=84, previous_epoch="sum")[0], 1)
        self.assertEqual(self.run_gate(rate=83.90, previous=84, previous_epoch="sum")[0], 0)
        self.assertNotIn("reviewed baseline reset", self.run_gate(previous_epoch="sum")[1])

    def test_unknown_epoch_fails(self):
        with self.assertRaisesRegex(ValueError, "epoch differs"):
            self.run_gate(previous_epoch="future")

    def test_missing_current_stamp_fails_even_without_token(self):
        with self.assertRaisesRegex(ValueError, "stamp before upload"):
            self.run_gate(stamp=False, token="")

    def test_invalid_reset_fails(self):
        for value in ["bad", {}, {"epoch": "", "line_rate": 83.44}, {"epoch": "sum", "line_rate": float("nan")}, {"epoch": "sum", "line_rate": 80}]:
            with self.subTest(value=value), self.assertRaises((ValueError, KeyError, TypeError)):
                self.config["baseline_reset"] = value
                self.run_gate(token="")

    def test_nonfinite_current_fails(self):
        with self.assertRaises(ValueError):
            self.run_gate(rate=float("nan"))

    def test_missing_remote_keeps_loud_floor_fallback(self):
        result, output = self.run_gate(token="")
        self.assertEqual(result, 0)
        self.assertIn("WARNING: no-regression check skipped", output)
        self.assertEqual(self.run_gate(rate=80.99, token="")[0], 1)

    def test_workflow_stamps_before_upload_and_gates_after(self):
        workflow = (ROOT / ".github/workflows/helia-core-tester.yml").read_text()
        self.assertLess(workflow.index("--stamp-epoch"), workflow.index("name: Upload merged coverage reports"))
        self.assertLess(workflow.index("name: Upload merged coverage reports"), workflow.index("name: Gate merged line coverage"))


if __name__ == "__main__":
    unittest.main()
