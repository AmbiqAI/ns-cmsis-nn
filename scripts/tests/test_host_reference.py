#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Ambiq
# SPDX-License-Identifier: LicenseRef-Ambiq-Apollo-SDK
"""The host reference build (NS_CMSIS_NN_HOST_REFERENCE): what it builds and every configuration it refuses."""

import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
GENERATOR = ["-G", "Ninja"] if shutil.which("ninja") else []
REFUSAL = "ns-cmsis-nn host reference:"


def configure(build: Path, *args: str, source: Path = ROOT) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["cmake", "-S", str(source), "-B", str(build), *GENERATOR, *args],
        text=True, capture_output=True, check=False,
    )


def build(build_dir: Path, *extra: str) -> subprocess.CompletedProcess:
    jobs = str(os.cpu_count() or 4)
    return subprocess.run(
        ["cmake", "--build", str(build_dir), "--parallel", jobs, *extra],
        text=True, capture_output=True, check=False,
    )


class HostReferenceBuildTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.build_dir = Path(cls._tmp.name) / "host-ref"
        cls.configured = configure(cls.build_dir, "-DNS_CMSIS_NN_HOST_REFERENCE=ON",
                                   "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON")
        cls.built = build(cls.build_dir) if cls.configured.returncode == 0 else None

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def test_configures_and_builds(self):
        self.assertEqual(self.configured.returncode, 0, self.configured.stdout + self.configured.stderr)
        self.assertEqual(self.built.returncode, 0, self.built.stdout + self.built.stderr)

    def test_manifest(self):
        manifest = json.loads((self.build_dir / "ns_cmsis_nn_host_reference.json").read_text())
        self.assertEqual(manifest["schema"], "hct/host-reference/1")
        self.assertEqual(manifest["library"], "ns-cmsis-nn")
        self.assertEqual(Path(manifest["source_dir"]), ROOT)
        self.assertTrue(Path(manifest["archive"]).is_file())
        self.assertEqual(manifest["include_dirs"], [str(ROOT / "Include")])
        self.assertEqual(sorted(manifest["compile_definitions"]), ["ARM_NN_ENABLE_F16=1", "ARM_NN_ENABLE_F32=1"])
        self.assertEqual(manifest["library_compile_options"], ["-O2", "-ffp-contract=off", "-fno-fast-math"])
        self.assertEqual(manifest["link_libraries"], ["m"])
        self.assertIn(manifest["float16_type"], ("__fp16", "_Float16"))
        self.assertTrue(Path(manifest["c_compiler"]["path"]).is_file())
        self.assertIn(manifest["c_compiler"]["id"], ("GNU", "Clang", "AppleClang"))

    def test_every_kernel_compiles_without_fast_math(self):
        commands = json.loads((self.build_dir / "compile_commands.json").read_text())
        kernels = [c for c in commands if "/Source/" in c["file"]]
        self.assertGreater(len(kernels), 200)
        for command in kernels:
            with self.subTest(file=command["file"]):
                self.assertIn("-ffp-contract=off", command["command"])
                self.assertIn("-fno-fast-math", command["command"])
                self.assertNotIn("-Ofast", command["command"])
        self.assertTrue(any(c["file"].endswith("ns_cmsis_nn_host_reference_guard.c") for c in commands))

    def test_archive_holds_integer_float_and_marker_symbols(self):
        archive = json.loads((self.build_dir / "ns_cmsis_nn_host_reference.json").read_text())["archive"]
        symbols = subprocess.run(["nm", archive], text=True, capture_output=True, check=True).stdout
        defined = {line.split()[-1].lstrip("_") for line in symbols.splitlines() if " T " in line or " S " in line
                   or " R " in line or " D " in line}
        for name in ("arm_convolve_s8", "arm_elementwise_add_f32", "arm_elementwise_add_f16",
                     "ns_cmsis_nn_host_reference_marker"):
            self.assertIn(name, defined)


class HostReferenceRefusalTests(unittest.TestCase):
    def refused(self, *args: str) -> str:
        with tempfile.TemporaryDirectory() as tmp:
            result = configure(Path(tmp) / "b", "-DNS_CMSIS_NN_HOST_REFERENCE=ON", *args)
        # CMake wraps FATAL_ERROR text, so compare with whitespace collapsed.
        log = " ".join((result.stdout + result.stderr).split())
        self.assertNotEqual(result.returncode, 0, log)
        self.assertIn(REFUSAL, log)
        return log

    def test_float_widths_are_required(self):
        for width in ("F32", "F16"):
            with self.subTest(width=width):
                log = self.refused(f"-DARM_NN_ENABLE_{width}=OFF")
                self.assertIn("needs ARM_NN_ENABLE_F32=ON and ARM_NN_ENABLE_F16=ON", log)

    def test_fast_math_is_refused_wherever_it_is_set(self):
        cases = (
            ("-DCMSIS_OPTIMIZATION_LEVEL=-Ofast", "CMSIS_OPTIMIZATION_LEVEL contains '-Ofast'"),
            ("-DCMAKE_C_FLAGS=-ffast-math", "CMAKE_C_FLAGS contains '-ffast-math'"),
            ("-DCMAKE_C_FLAGS=-g -funsafe-math-optimizations", "contains '-funsafe-math-optimizations'"),
            ("-DCMAKE_C_FLAGS=-ffinite-math-only", "contains '-ffinite-math-only'"),
        )
        for arg, message in cases:
            with self.subTest(arg=arg):
                self.assertIn(message, self.refused(arg))
        log = self.refused("-DCMAKE_BUILD_TYPE=Release", "-DCMAKE_C_FLAGS_RELEASE=-O3 -ffp-contract=fast")
        self.assertIn("CMAKE_C_FLAGS_RELEASE contains '-ffp-contract=fast'", log)

    def test_flags_that_select_a_kernel_path_are_refused(self):
        for macro in ("ARM_MATH_DSP", "ARM_MATH_MVEI", "ARM_MATH_HELIUM", "ARM_MATH_AUTOVECTORIZE"):
            with self.subTest(macro=macro):
                log = self.refused(f"-DCMAKE_C_FLAGS=-D{macro}")
                self.assertIn("the compile flags select a kernel path", log)
                self.assertIn(macro, log)

    def test_requantize_assembly_is_refused(self):
        log = self.refused("-DCMSIS_NN_USE_REQUANTIZE_INLINE_ASSEMBLY=ON")
        self.assertIn("CMSIS_NN_USE_REQUANTIZE_INLINE_ASSEMBLY selects Arm assembly", log)

    def test_cross_compiling_is_refused(self):
        log = self.refused("-DCMAKE_SYSTEM_NAME=Generic", "-DCMAKE_TRY_COMPILE_TARGET_TYPE=STATIC_LIBRARY")
        self.assertIn("configure it without a cross toolchain", log)

    def test_compiler_without_float16_is_refused(self):
        # Fault injection: hide the compiler's float16 support from the header.
        log = self.refused("-DCMAKE_C_FLAGS=-U__FLT16_MAX__ -U__ARM_FP16_FORMAT_IEEE -Wno-builtin-macro-redefined")
        self.assertIn("has no IEEE float16 type", log)


class HostReferenceGuardTests(unittest.TestCase):
    def test_path_macro_on_the_target_fails_the_build(self):
        with tempfile.TemporaryDirectory() as tmp:
            wrapper = Path(tmp) / "wrapper"
            wrapper.mkdir()
            (wrapper / "CMakeLists.txt").write_text(f'''
cmake_minimum_required(VERSION 3.15.6)
project(host_reference_guard C)
set(NS_CMSIS_NN_HOST_REFERENCE ON CACHE BOOL "" FORCE)
add_subdirectory("{ROOT}" ns-cmsis-nn)
target_compile_definitions(cmsis-nn PRIVATE ARM_MATH_NEON)
''')
            result = configure(Path(tmp) / "b", source=wrapper)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            built = build(Path(tmp) / "b")
        log = built.stdout + built.stderr
        self.assertNotEqual(built.returncode, 0, log)
        self.assertIn("a kernel path macro is defined", log)


class DefaultBuildTests(unittest.TestCase):
    def test_option_off_keeps_the_firmware_defaults(self):
        with tempfile.TemporaryDirectory() as tmp:
            build_dir = Path(tmp) / "b"
            result = configure(build_dir)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            cache = (build_dir / "CMakeCache.txt").read_text()
            self.assertIn("CMSIS_OPTIMIZATION_LEVEL:STRING=-Ofast", cache)
            self.assertIn("ARM_NN_ENABLE_F32:BOOL=OFF", cache)
            self.assertIn("ARM_NN_ENABLE_F16:BOOL=OFF", cache)
            self.assertFalse((build_dir / "ns_cmsis_nn_host_reference.json").exists())


if __name__ == "__main__":
    unittest.main()
