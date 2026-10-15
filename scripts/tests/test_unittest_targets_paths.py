#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright 2026 Ambiq <opensource@ambiq.com>
# SPDX-License-Identifier: LicenseRef-Ambiq-Apollo-SDK
"""Preserve argument boundaries for binary-directory paths."""

import importlib.util
import unittest
from pathlib import Path
from unittest.mock import patch

SCRIPT = Path(__file__).resolve().parents[2] / "Tests/UnitTest/unittest_targets.py"
spec = importlib.util.spec_from_file_location("generator", SCRIPT)
generator = importlib.util.module_from_spec(spec)
spec.loader.exec_module(generator)


class CommandPaths(unittest.TestCase):
    def test_path_arguments_and_legacy_commands(self):
        commands = [
            ["ruby", "/build with space/Unity/auto/generate_test_runner.rb", "test.c", "/build with space/runner.c"],
            ["tar", "xzf", "/scratch with space/Unity.tar.gz", "-C", "/build with space/Unity", "--strip-components=1"],
            ["rm", "-f", "/scratch with space/download"],
        ]
        with patch.object(generator.subprocess, "run") as run:
            run.return_value.returncode = 0
            for command in commands:
                with self.subTest(command=command):
                    self.assertEqual(generator.run_command(command), 0)
                    run.assert_called_with(command)
            generator.run_command("mbed deploy")
            run.assert_called_with(["mbed", "deploy"])


if __name__ == "__main__":
    unittest.main()
