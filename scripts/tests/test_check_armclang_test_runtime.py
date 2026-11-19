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


@unittest.skipUnless(all(shutil.which(x) for x in ("cmake", "gcc", "ar", "nm", "objcopy")), "CMake/GNU tools required")
class ArmclangTestRuntime(unittest.TestCase):
    def configure(self, cpu="cortex-m4", missing=None, host=False, unsafe_call=False, public_export=False):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        source = Path(directory.name)
        lib = source / "arm/lib/armlib"
        lib.mkdir(parents=True)
        bodies = {
            "erfcf.o": "void erfcf(void) {} void __hardfp_erfcf(void) {}",
            "floorf.o": "void floorf(void) {} void __hardfp_floorf(void) {}",
            "roundf.o": "void roundf(void) {} void __hardfp_roundf(void) {}",
            "expf_i.o": "void __mathlib_expf(void) {}",
            "fpclassifyf.o": "void __ARM_fpclassifyf(void) {}",
            "funder.o": "void __mathlib_flt_underflow(void) {}",
            "_rserrno.o": "extern volatile int *__aeabi_errno_addr(void); "
                           "void __set_errno(int value) { *__aeabi_errno_addr() = value; }",
            "frnd.o": "void _frnd(void) {}",
            "fnaninf.o": "void __fpl_fnaninf(void) {}",
            "usenofp.o": "void __I$use$fp(void) {}",
        }
        if unsafe_call:
            bodies["expf_i.o"] = "extern void erfcf(void); void __mathlib_expf(void) { erfcf(); }"
        if public_export:
            bodies["funder.o"] += " void unexpected_public_math(void) {}"
        for name, body in bodies.items():
            src = source / (name + ".c")
            src.write_text(body)
            subprocess.run(["gcc", "-fno-builtin", "-c", str(src), "-o", str(source / name)],
                           check=True, capture_output=True)
        math_members = ["erfcf.o", "floorf.o", "roundf.o", "expf_i.o", "fpclassifyf.o", "funder.o"]
        suffix = "wv" if cpu == "cortex-m55" else "wm"
        # A wrong-CPU selection must fail rather than find an identical fixture.
        for archive, members in {
            f"m_{suffix}.l": math_members,
            "c_wu.l": ["_rserrno.o"],
            f"fz_{suffix}.l": ["frnd.o", "fnaninf.o", "usenofp.o"],
        }.items():
            objects = [str(source / name) for name in members if name != missing]
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
                self.assertNotIn(math, links)
                self.assertIn("erfcf.o", links)
                self.assertIn("cmsis_nn_test_armclang_errno", links)
                self.assertEqual((build / "armclang-test-runtime/_rserrno.o").read_bytes(),
                                 (build.parent / "_rserrno.o").read_bytes())
                symbols = subprocess.check_output(
                    ["nm", "-g", str(build / "armclang-test-runtime/erfcf.o")], text=True)
                self.assertIn("__hardfp_erfcf", symbols)
                self.assertNotRegex(symbols, r"(?m)\s[TW] erfcf$")
                self.assertIn("frnd.o", links)
                self.assertNotIn("fz_wm.l", links)
                # Native base-PCS aliases must not capture ordinary GNU calls.
                probe = build / "math.c"
                probe.write_text(
                    "#include <assert.h>\n#include <math.h>\n"
                    "int main(void) {\n"
                    " float (*volatile e)(float)=erfcf, (*volatile f)(float)=floorf;\n"
                    " float (*volatile r)(float)=roundf;\n"
                    " float x=e(1.0f); assert(x>0.1f && x<0.2f);\n"
                    " assert(f(-1.25f)==-2.0f && r(1.5f)==2.0f); return 0;\n}\n")
                program = build / "math-test"
                bridge = ROOT / "Tests/UnitTest/cmake/armclang_errno.c"
                objects = sorted((build / "armclang-test-runtime").glob("*.o"))
                subprocess.run(["gcc", "-fno-builtin", str(probe), str(bridge),
                                *map(str, objects), "-lm", "-o", str(program)], check=True)
                subprocess.run([str(program)], check=True)

    def test_incomplete_runtime_and_unsupported_targets_fail(self):
        cases = [
            ({"missing": "frnd.o"}, "Cannot extract"),
            ({"missing": "_rserrno.o"}, "Cannot extract"),
            ({"missing": "m_wm.l"}, "Missing ACfE test runtime archive"),
            ({"cpu": "cortex-m0"}, "supports only cortex-m4 and cortex-m55"),
            ({"host": True}, "requires a GNU cross-compiled harness"),
            ({"unsafe_call": True}, "Unexpected ACfE test runtime dependency: erfcf"),
            ({"public_export": True}, "Unexpected public ACfE math symbol"),
        ]
        for kwargs, message in cases:
            with self.subTest(kwargs=kwargs):
                _, result = self.configure(**kwargs)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(message, result.stdout + result.stderr)


    def test_arm_and_gnu_errno_storage_are_independent(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "errno.c"
            source.write_text(
                "#include <assert.h>\n#include <errno.h>\n"
                "extern volatile int *__aeabi_errno_addr(void);\n"
                "int main(void) {\n"
                "  assert(*__aeabi_errno_addr() == 0);\n"
                "  errno = EDOM; *__aeabi_errno_addr() = 2;\n"
                "  assert(errno == EDOM && *__aeabi_errno_addr() == 2);\n"
                "  errno = ERANGE; *__aeabi_errno_addr() = 1;\n"
                "  assert(errno == ERANGE && *__aeabi_errno_addr() == 1);\n"
                "  return 0;\n}\n")
            program = Path(directory) / "errno-test"
            bridge = ROOT / "Tests/UnitTest/cmake/armclang_errno.c"
            subprocess.run(["gcc", str(source), str(bridge), "-o", str(program)], check=True)
            subprocess.run([str(program)], check=True)


if __name__ == "__main__":
    unittest.main()
