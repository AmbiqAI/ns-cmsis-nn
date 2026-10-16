#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright 2026 Ambiq <opensource@ambiq.com>
# SPDX-License-Identifier: Apache-2.0

"""Exercise runtime selection and failed extraction without an Arm licence."""

import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MODULE = ROOT / "Tests/UnitTest/cmake/armclang_runtime.cmake"


@unittest.skipUnless(all(shutil.which(x) for x in ("cmake", "gcc", "ar")), "CMake/GNU tools required")
class ArmclangTestRuntime(unittest.TestCase):
    def configure(self, cpu="cortex-m4", missing=None, host=False):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        source = Path(directory.name)
        lib = source / "arm/lib/armlib"
        lib.mkdir(parents=True)
        for archive, members in {
            "m_wm.l": ["math.o"],
            "m_wv.l": ["math.o"],
            "c_wu.l": ["_rserrno.o"],
            "fz_wm.l": ["frnd.o", "fnaninf.o", "usenofp.o"],
        }.items():
            objects = []
            for name in members:
                if name != missing:
                    obj = source / name
                    obj.write_bytes(("fixture:" + name).encode())
                    objects.append(str(obj))
            subprocess.run(["ar", "qc", str(lib / archive), *objects], check=True, capture_output=True)
        if missing == "m_wm.l":
            (lib / missing).unlink()
        (source / "CMakeLists.txt").write_text(
            'cmake_minimum_required(VERSION 3.15.6)\n'
            'project(runtime_test C)\n'
            'add_library(kernel STATIC IMPORTED)\n'
            'set_target_properties(kernel PROPERTIES IMPORTED_LOCATION "${CMAKE_SOURCE_DIR}/dummy.a")\n'
            f'include("{MODULE.as_posix()}")\n'
            'cmsis_nn_link_armclang_test_runtime(kernel "${CMAKE_SOURCE_DIR}/arm")\n'
            'get_target_property(links kernel INTERFACE_LINK_LIBRARIES)\n'
            'file(WRITE "${CMAKE_BINARY_DIR}/selected.txt" "${links}")\n'
        )
        build = source / "build"
        command = ["cmake", "-S", str(source), "-B", str(build), "-DCMAKE_C_COMPILER=gcc"]
        if not host:
            command += ["-DCMAKE_SYSTEM_NAME=Generic", f"-DCMAKE_SYSTEM_PROCESSOR={cpu}"]
        result = subprocess.run(command, text=True, capture_output=True)
        return build, result

    def test_m4_and_m55_select_only_their_native_dependencies(self):
        for cpu, math in [("cortex-m4", "m_wm.l"), ("cortex-m55", "m_wv.l")]:
            with self.subTest(cpu=cpu):
                build, result = self.configure(cpu)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                links = (build / "selected.txt").read_text()
                self.assertIn(math, links)
                self.assertIn("cmsis_nn_test_armclang_errno", links)
                self.assertEqual((build / "armclang-test-runtime/_rserrno.o").read_bytes(), b"fixture:_rserrno.o")
                self.assertEqual("frnd.o" in links, cpu == "cortex-m4")
                self.assertNotIn("fz_wm.l", links)

    def test_incomplete_runtime_and_unsupported_targets_fail(self):
        cases = [
            ({"missing": "frnd.o"}, "Cannot extract"),
            ({"missing": "_rserrno.o"}, "Cannot extract"),
            ({"missing": "m_wm.l"}, "Missing ACfE test runtime archive"),
            ({"cpu": "cortex-m0"}, "supports only cortex-m4 and cortex-m55"),
            ({"host": True}, "requires a GNU cross-compiled harness"),
        ]
        for kwargs, message in cases:
            with self.subTest(kwargs=kwargs):
                _, result = self.configure(**kwargs)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(message, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
