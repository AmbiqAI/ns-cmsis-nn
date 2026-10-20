#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Ambiq
# SPDX-License-Identifier: LicenseRef-Ambiq-Apollo-SDK
"""Check DIV's CMake compatibility gate without compiling kernels. Refs #831."""

import json
from pathlib import Path
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]


class DivOptionsTests(unittest.TestCase):
    def test_clang_requirement_only_applies_to_enabled_div(self):
        for old in (False, True):
            for f32, f16 in ((0, 0), (1, 0), (0, 1), (1, 1)):
                with self.subTest(old=old, f32=f32, f16=f16), tempfile.TemporaryDirectory() as tmp:
                    directory = Path(tmp)
                    # Model the helper's compiler/version selection after normal compiler detection.
                    # Actual Clang object compilation is covered separately by the toolchain checks.
                    version = 'set(CMAKE_VERSION "3.17.5")' if old else ''
                    (directory / "CMakeLists.txt").write_text(f'''
cmake_minimum_required(VERSION 3.18)
project(div_options C)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_C_COMPILER_ID Clang)
{version}
set(ARM_NN_ENABLE_F32 {f32})
set(ARM_NN_ENABLE_F16 {f16})
add_library(cmsis-nn OBJECT)
add_subdirectory("{ROOT}/Source/BasicMathFunctions" group)
''')
                    result = subprocess.run(
                        ["cmake", "-S", str(directory), "-B", str(directory / "build")],
                        text=True, capture_output=True, check=False,
                    )
                    log = result.stdout + result.stderr
                    if old and (f32 or f16):
                        self.assertNotEqual(result.returncode, 0, log)
                        self.assertIn("Clang DIV source options require CMake 3.18", log)
                    else:
                        self.assertEqual(result.returncode, 0, log)
                        commands = json.loads((directory / "build/compile_commands.json").read_text())
                        div = [c for c in commands if "arm_elementwise_div_f" in c["file"]]
                        self.assertEqual(len(div), f32 + f16)
                        self.assertTrue(all("-fno-fast-math" in c["command"] for c in div))
                        self.assertTrue(all("-fno-fast-math" not in c["command"] for c in commands if c not in div))


if __name__ == "__main__":
    unittest.main()
