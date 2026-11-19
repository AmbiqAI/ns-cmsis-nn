#!/usr/bin/env python3
"""Exercise legacy FVP runner outcomes without a compiler or simulator."""
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "Tests/UnitTest/build_and_run_tests.sh"


class RunnerTests(unittest.TestCase):
    def invoke(self, mock, *, image=True, limit="1"):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            binary = root / "FVP_Corstone_SSE-300_Ethos-U55"
            binary.write_text("#!/bin/bash\n" + mock + "\n")
            binary.chmod(0o755)
            build = root / "build-cortex-m55-gcc"
            build.mkdir()
            if image:
                (build / "case with spaces.elf").touch()
            source = SCRIPT.read_text()
            function = source[source.index("Run_Tests() {"):source.index("\nif [[ ${BUILD}")]
            env = dict(os.environ, PATH=str(root) + os.pathsep + os.environ["PATH"],
                       FVP_TEST_TIMEOUT_SECONDS=limit)
            result = subprocess.run(["bash", "-c", function +
                                     "\ncpu=cortex-m55; compiler=gcc; QUIET=1; Run_Tests"],
                                    cwd=root, env=env, text=True, capture_output=True, timeout=10)
            logs = list(build.glob("*.fvp.log"))
            return result, logs[0].read_text() if logs else None

    def test_pass(self):
        result, log = self.invoke("printf '1 Tests 0 Failures 0 Ignored\\nOK\\n'")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("OK", log)

    def test_failure_summary(self):
        result, _ = self.invoke("printf '10 Tests 10 Failures 0 Ignored\\nOK\\n'")
        self.assertNotEqual(result.returncode, 0)

    def test_process_failure_despite_pass_text(self):
        result, _ = self.invoke("printf '1 Tests 0 Failures 0 Ignored\\nOK\\n'; exit 7")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("simulator exit 7", result.stdout)

    def test_timeout_despite_pass_text(self):
        result, log = self.invoke("printf '1 Tests 0 Failures 0 Ignored\\nOK\\n'; sleep 30")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("simulator exit 124", result.stdout)
        self.assertIn("OK", log)

    def test_empty_discovery(self):
        result, _ = self.invoke("exit 0", image=False)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("No test images", result.stdout)

    def test_invalid_limit(self):
        for limit in ("0", "-1", "unlimited"):
            result, _ = self.invoke("exit 0", limit=limit)
            self.assertNotEqual(result.returncode, 0)


if __name__ == "__main__":
    unittest.main()
