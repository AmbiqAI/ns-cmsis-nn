#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright 2026 Ambiq <opensource@ambiq.com>
# SPDX-License-Identifier: LicenseRef-Ambiq-Apollo-SDK
"""Complete Unity result and suite identity controls."""

import importlib.util
import contextlib
import io
import sys
from unittest.mock import patch
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "check_atfe_word_copy.py"
spec = importlib.util.spec_from_file_location("word_copy", SCRIPT)
word_copy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(word_copy)

RUNNER = '''UnityBegin("TestCases/test_arm_pad_s16/Unity/unity_test_arm_pad_s16.c");
run_test(test_pad_1, "test_pad_1", 38);
run_test(test_pad_2, "test_pad_2", 40);
'''
PASS = '''TestCases/test_arm_pad_s16/Unity/unity_test_arm_pad_s16.c:38:test_pad_1:PASS
TestCases/test_arm_pad_s16/Unity/unity_test_arm_pad_s16.c:40:test_pad_2:PASS

-----------------------
2 Tests 0 Failures 0 Ignored
OK
'''


class UnityResults(unittest.TestCase):
    def test_complete_stream_and_suite_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            runner = Path(directory) / "runner.c"
            runner.write_text(RUNNER)
            expected = word_copy.runner_cases(runner)
        self.assertTrue(word_copy.unity_pass(PASS, expected))
        self.assertTrue(word_copy.unity_pass(PASS.replace(" Ignored\n", " Ignored \n"), expected))
        self.assertTrue(word_copy.unity_pass("diagnostic: sweep complete\n" + PASS, expected))
        self.assertTrue(word_copy.unity_pass(PASS.replace("\n", "\r\n") + "\x04\0", expected))
        variants = {
            "empty": "",
            "summary_only": "2 Tests 0 Failures 0 Ignored\nOK\n",
            "zero_tests": "0 Tests 0 Failures 0 Ignored\nOK\n",
            "earlier_failure": "bad.c:1:test_bad:FAIL\n1 Tests 1 Failures 0 Ignored\nFAIL\n" + PASS,
            "failed_record": PASS.replace(":PASS", ":FAIL", 1),
            "earlier_failed_record": "bad.c:1:test_bad:FAIL\n" + PASS,
            "later_failure": PASS + "bad.c:1:test_bad:FAIL\nFAIL\n",
            "extra_summary": "1 Tests 0 Failures 0 Ignored\nOK\n" + PASS,
            "missing_case": PASS.split("\n", 1)[1],
            "duplicate_case": PASS.replace("40:test_pad_2", "38:test_pad_1"),
            "wrong_suite": PASS.replace("test_arm_pad_s16", "test_arm_convolve_s8"),
            "wrong_line": PASS.replace(":38:", ":39:"),
            "ignored_case": PASS.replace(":PASS", ":IGNORE", 1).replace("0 Ignored", "1 Ignored"),
            "nonzero_exit": PASS + "\x04\x01",
        }
        for name, text in variants.items():
            with self.subTest(name=name):
                self.assertFalse(word_copy.unity_pass(text, expected))
        self.assertFalse(word_copy.unity_pass(PASS, []))

    def test_main_binds_the_requested_suite(self):
        class Runtime:
            def __init__(self, command, stdout, stderr):
                stdout.write(self.output)
            def poll(self):
                return 0
            def wait(self, timeout):
                return 0

        with tempfile.TemporaryDirectory() as directory:
            build = Path(directory)
            runner = build / "generated/TestCases/test_arm_convolve_s8/Unity/TestRunner/unity_test_arm_convolve_s8_runner.c"
            runner.parent.mkdir(parents=True)
            runner.write_text(RUNNER.replace("test_arm_pad_s16", "test_arm_convolve_s8"))
            for text, exit_code in [(PASS, 1), (PASS.replace("test_arm_pad_s16", "test_arm_convolve_s8"), 0)]:
                Runtime.output = text
                with patch.object(word_copy, "SUITES", ("test_arm_convolve_s8",)), \
                     patch.object(sys, "argv", ["check", "--build", directory]), \
                     patch.object(word_copy.subprocess, "run"), \
                     patch.object(word_copy.subprocess, "Popen", Runtime), \
                     contextlib.redirect_stdout(io.StringIO()):
                    self.assertEqual(word_copy.main(), exit_code)

    def test_missing_or_duplicate_runner_cases_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            runner = Path(directory) / "runner.c"
            for text in ('UnityBegin("test.c");', RUNNER + 'run_test(test_pad_1, "test_pad_1", 38);'):
                runner.write_text(text)
                with self.assertRaises(ValueError):
                    word_copy.runner_cases(runner)


if __name__ == "__main__":
    unittest.main()
