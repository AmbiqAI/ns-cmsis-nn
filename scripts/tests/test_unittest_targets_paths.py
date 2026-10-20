#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright 2026 Ambiq <opensource@ambiq.com>
# SPDX-License-Identifier: LicenseRef-Ambiq-Apollo-SDK
"""Check runner generation and argument boundaries for binary-directory paths."""

import importlib.util
import subprocess
import tempfile
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


class RunnerHeaders(unittest.TestCase):
    def test_exit_runner_compiles_without_wrapper_includes(self):
        with tempfile.TemporaryDirectory() as tmp:
            runner = Path(tmp) / "runner.c"
            runner.write_text("#define UNITY_END() 7\nint main(void)\n{\n    return UNITY_END();\n}\n")
            generator.parse_generated_test_runner(str(runner))
            self.assertIn("exit(UNITY_END());", runner.read_text())
            executable = Path(tmp) / "runner"
            result = subprocess.run(
                ["cc", "-std=c11", "-Wall", "-Werror", str(runner), "-o", str(executable)],
                capture_output=True, text=True, check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(subprocess.run([str(executable)], check=False).returncode, 7)

    def test_existing_include_and_repeat_processing(self):
        for include in ("", "#include <stdlib.h>\n", '# include "stdlib.h"\n'):
            with self.subTest(include=include), tempfile.TemporaryDirectory() as tmp:
                runner = Path(tmp) / "runner.c"
                runner.write_text(include + "int main(void)\n{\n    return UNITY_END();\n}\n")
                generator.parse_generated_test_runner(str(runner))
                processed = runner.read_text()
                self.assertEqual(processed.count("stdlib.h"), 1)
                generator.parse_generated_test_runner(str(runner))
                self.assertEqual(runner.read_text(), processed)

    def test_no_exit_rewrite_keeps_input_unchanged(self):
        with tempfile.TemporaryDirectory() as tmp:
            runner = Path(tmp) / "runner.c"
            original = "int main(void)\n{\n    return 0;\n}\n"
            runner.write_text(original)
            generator.parse_generated_test_runner(str(runner))
            self.assertEqual(runner.read_text(), original)


if __name__ == "__main__":
    unittest.main()
