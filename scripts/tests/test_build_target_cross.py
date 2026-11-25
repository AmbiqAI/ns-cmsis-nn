# SPDX-License-Identifier: LicenseRef-Ambiq-Apollo-SDK
"""The build-target check against a real cross compiler (arm-none-eabi-gcc): what Include/arm_nn_build_target.h
derives per CPU and flag set, the library stamp each build defines, the #error of arm_nn_build_target_check.h for
every mismatch, the link failure against a library of another build, and the probe -> render_guard round trip.

Set ARM_NONE_EABI_GCC to pick the compiler; with NS_CMSIS_NN_REQUIRE_CROSS=1 (CI) a missing compiler fails the run
instead of skipping it.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
try:
    import ns_cmsis_nn_contracts  # noqa: F401  (the installed package, in CI)
except ImportError:
    sys.path.insert(0, str(ROOT / "python"))
from ns_cmsis_nn_contracts import (BuildTargetError, Target, load_implementations, load_signatures, probe_compiler,
                                   render_guard)

GCC = os.environ.get("ARM_NONE_EABI_GCC") or shutil.which("arm-none-eabi-gcc")
STAMP_SOURCE = ROOT / "Source" / "NNSupportFunctions" / "arm_nn_build_target.c"
STRICT = ["-mthumb", "-O2", "-Wall", "-Wextra", "-Werror"]

M0 = ["-mcpu=cortex-m0"]
M4 = ["-mcpu=cortex-m4", "-mfpu=fpv4-sp-d16", "-mfloat-abi=hard"]
M55 = ["-mcpu=cortex-m55", "-mfloat-abi=hard"]
M55_AV = M55 + ["-DARM_MATH_AUTOVECTORIZE"]
M55_SR = M55 + ["-DCMSIS_NN_USE_SINGLE_ROUNDING"]
MVE = {"mvei", "dsp", "mvef", "mve_f16"}


@pytest.fixture(scope="module")
def gcc():
    if GCC is None:
        if os.environ.get("NS_CMSIS_NN_REQUIRE_CROSS") == "1":
            pytest.fail("NS_CMSIS_NN_REQUIRE_CROSS=1 but arm-none-eabi-gcc is not on PATH (set ARM_NONE_EABI_GCC)")
        pytest.skip("arm-none-eabi-gcc not on PATH")
    return GCC


def tool(gcc: str, name: str) -> str:
    return gcc[: -len("gcc")] + name


def compile_c(gcc, flags, source: Path, output: Path, *extra):
    return subprocess.run([gcc, *flags, *STRICT, f"-I{ROOT / 'Include'}", *extra, "-c", str(source), "-o", str(output)],
                          capture_output=True, text=True, check=False)


def errors(result) -> list[str]:
    return [line.split("#error ", 1)[1].strip().strip('"') for line in result.stderr.splitlines()
            if "error: #error " in line]


def guard_source(tmp_path, text, body="int model_init(void) { return ARM_NN_LIBRARY_STAMP() == 1 ? 0 : -1; }\n"):
    source = tmp_path / "model.c"
    source.write_text(text + body)
    return source


@pytest.mark.parametrize("flags, target, features", [
    (M0, Target("scalar"), set()),
    (M4, Target("dsp"), {"dsp"}),
    (["-mcpu=cortex-m55"], Target("dsp"), {"dsp"}),  # soft float ABI: GCC turns MVE off
    (["-mcpu=cortex-m55+nomve", "-mfloat-abi=hard"], Target("dsp"), {"dsp"}),
    (["-mcpu=cortex-m55+nomve.fp", "-mfloat-abi=hard"], Target("mve"), {"mvei", "dsp"}),
    (M55, Target("mve"), MVE),
    (M55_AV, Target("mve", autovectorize=True), {"mvei", "dsp", "autovectorize"}),
    (M55_SR, Target("mve", single_rounding=True), MVE | {"single_rounding"}),
    (M4 + ["-DCMSIS_NN_USE_SINGLE_ROUNDING"], Target("dsp", single_rounding=True), {"dsp", "single_rounding"}),
    (M4 + ["-DARM_NN_ENABLE_F32=1"], Target("dsp"), {"dsp", "f32_api"}),
    (M55 + ["-DARM_NN_ENABLE_F16"], Target("mve"), MVE | {"f16_api"}),
    (M55 + ["-DARM_NN_ENABLE_F32=1", "-DARM_NN_ENABLE_F16=1"], Target("mve"), MVE | {"f32_api", "f16_api"}),
    (M55 + ["-DARM_NN_ENABLE_F32=0", "-DARM_NN_ENABLE_F16=1"], Target("mve"), MVE | {"f16_api"}),
])
def test_derivation_and_stamp_agree(gcc, tmp_path, flags, target, features):
    build = probe_compiler(ROOT, gcc, flags)
    assert (build.target, build.toolchain, build.features) == (target, "gcc", frozenset(features))
    obj = tmp_path / "stamp.o"
    result = compile_c(gcc, flags, STAMP_SOURCE, obj)
    assert result.returncode == 0, result.stderr
    defined = subprocess.run([tool(gcc, "nm"), "--defined-only", str(obj)], capture_output=True, text=True,
                             check=True).stdout.split()
    assert [s for s in defined if s.startswith("arm_nn_build_")] == [build.stamp]


def test_probe_reads_the_requested_checkout_over_a_competing_include(gcc, tmp_path):
    competing = tmp_path / "other" / "Include"
    (competing / "Internal").mkdir(parents=True)
    for name in ("arm_nn_build_target.h", "arm_nn_math_types.h", "Internal/arm_nn_compiler.h"):
        (competing / name).write_text('#error "the competing copy was used"\n')
    build = probe_compiler(ROOT, gcc, [f"-I{competing}", f"-iquote{competing}", *M55])
    assert build.target == Target("mve") and build.stamp == "arm_nn_build_mve_av0_sr0_gcc"


def test_probe_refuses_autovectorize_without_mve(gcc):
    with pytest.raises(BuildTargetError, match="does not model"):
        probe_compiler(ROOT, gcc, M4 + ["-DARM_MATH_AUTOVECTORIZE"])


def test_probe_reports_bad_flags(gcc):
    with pytest.raises(BuildTargetError, match="exited with 1"):
        probe_compiler(ROOT, gcc, ["-mcpu=cortex-m99"])


BUILDS = {"scalar": M0, "dsp": M4, "mve": M55, "mve+av": M55_AV, "mve+sr": M55_SR}
PLANS = {"scalar": Target("scalar"), "dsp": Target("dsp"), "mve": Target("mve"),
         "mve+av": Target("mve", autovectorize=True), "mve+sr": Target("mve", single_rounding=True)}
ISA_ERRORS = {
    ("mve", "dsp"): "planned for an mve library, this compiler builds dsp (no ARM_MATH_MVEI)",
    ("mve", "scalar"): "planned for an mve library, this compiler builds scalar (no ARM_MATH_MVEI)",
    ("dsp", "mve"): "planned for a dsp library, this compiler builds mve (ARM_MATH_MVEI)",
    ("dsp", "scalar"): "planned for a dsp library, this compiler builds scalar (no ARM_MATH_DSP)",
    ("scalar", "mve"): "planned for a scalar library, this compiler builds mve (ARM_MATH_MVEI)",
    ("scalar", "dsp"): "planned for a scalar library, this compiler builds dsp (ARM_MATH_DSP)",
}


def expected_errors(plan: Target, build: Target) -> list[str]:
    found = []
    if plan.isa != build.isa:
        found.append(ISA_ERRORS[(plan.isa, build.isa)])
    for flag, macro in (("autovectorize", "ARM_MATH_AUTOVECTORIZE"), ("single_rounding", "CMSIS_NN_USE_SINGLE_ROUNDING")):
        if getattr(plan, flag) and not getattr(build, flag):
            found.append(f"planned with {macro}, this build does not define it")
        elif getattr(build, flag) and not getattr(plan, flag):
            found.append(f"planned without {macro}, this build defines it")
    return ["ns-cmsis-nn build check: " + message for message in found]


@pytest.mark.parametrize("plan_name", sorted(PLANS))
@pytest.mark.parametrize("build_name", sorted(BUILDS))
def test_guard_matrix(gcc, tmp_path, plan_name, build_name):
    source = guard_source(tmp_path, render_guard(PLANS[plan_name], "gcc"))
    result = compile_c(gcc, BUILDS[build_name], source, tmp_path / "model.o")
    want = expected_errors(PLANS[plan_name], PLANS[build_name])
    assert (result.returncode == 0) == (not want), result.stderr
    assert errors(result) == want


def test_guard_toolchain_mismatch(gcc, tmp_path):
    source = guard_source(tmp_path, render_guard(Target("mve"), "armclang"))
    result = compile_c(gcc, M55, source, tmp_path / "model.o")
    assert errors(result) == ["ns-cmsis-nn build check: planned for Arm Compiler 6 (armclang), this compiler is not "
                              "armclang"]


EXPECT = {"ISA": "ARM_NN_ISA_MVE", "AUTOVECTORIZE": "0", "SINGLE_ROUNDING": "0", "TOOLCHAIN": "ARM_NN_TOOLCHAIN_ID_GCC"}


def expectations(**changes):
    values = dict(EXPECT, **changes)
    return "".join(f"#define ARM_NN_EXPECT_{name} {value}\n" for name, value in values.items() if value is not None)


MISSING = "define ARM_NN_EXPECT_ISA, _AUTOVECTORIZE, _SINGLE_ROUNDING and _TOOLCHAIN first"
RANGE = "an ARM_NN_EXPECT_* value is out of range (ISA 0-2, the others 0 or 1)"


@pytest.mark.parametrize("prelude, message", [
    (expectations(TOOLCHAIN=None), MISSING),
    ("", MISSING),
    (expectations(ISA="3"), RANGE),
    (expectations(ISA="-1"), RANGE),
    (expectations(AUTOVECTORIZE="2"), RANGE),
    (expectations(SINGLE_ROUNDING="-1"), RANGE),
    (expectations(TOOLCHAIN="2"), RANGE),
    (expectations(ISA="ARM_NN_ISA_DSP", AUTOVECTORIZE="1"), "ARM_NN_EXPECT_AUTOVECTORIZE applies to an MVE plan only"),
])
def test_guard_refuses_bad_expectations(gcc, tmp_path, prelude, message):
    source = guard_source(tmp_path, prelude + '#include "arm_nn_build_target_check.h"\n', body="")
    result = compile_c(gcc, M55, source, tmp_path / "model.o")
    assert result.returncode != 0
    assert errors(result) == ["ns-cmsis-nn build check: " + message]


def test_guard_compiles_as_c99_and_cxx(gcc, tmp_path):
    text = render_guard(Target("mve"), "gcc")
    source = guard_source(tmp_path, text)
    assert compile_c(gcc, M55, source, tmp_path / "c99.o", "-std=c99").returncode == 0
    cxx = tmp_path / "model.cpp"
    cxx.write_text(text + 'extern "C" int model_init(void) { return ARM_NN_LIBRARY_STAMP() == 1 ? 0 : -1; }\n')
    result = subprocess.run([tool(gcc, "g++"), *M55, *STRICT, f"-I{ROOT / 'Include'}", "-c", str(cxx), "-o",
                             str(tmp_path / "cxx.o")], capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
    symbols = subprocess.run([tool(gcc, "nm"), str(tmp_path / "cxx.o")], capture_output=True, text=True, check=True)
    assert "U arm_nn_build_mve_av0_sr0_gcc" in symbols.stdout


def link(gcc, tmp_path, archive_flags, objects=None):
    """Link a model planned for M55 against an archive whose stamp is compiled with ``archive_flags``."""
    model = tmp_path / "model.o"
    assert compile_c(gcc, M55, guard_source(tmp_path, render_guard(Target("mve"), "gcc")), model).returncode == 0
    archive = tmp_path / "libns-cmsis-nn.a"
    archive.unlink(missing_ok=True)
    if objects is None:
        objects = [tmp_path / "stamp.o"]
        assert compile_c(gcc, archive_flags, STAMP_SOURCE, objects[0]).returncode == 0
    subprocess.run([tool(gcc, "ar"), "rcs", str(archive), *map(str, objects)], check=True)
    main = tmp_path / "main.c"
    main.write_text("int model_init(void);\nint main(void) { return model_init(); }\n")
    return subprocess.run([gcc, *M55, "-mthumb", "--specs=nosys.specs", "-Wl,--gc-sections", str(main), str(model),
                           str(archive), "-o", str(tmp_path / "app.elf")], capture_output=True, text=True, check=False)


def test_link_against_the_planned_library(gcc, tmp_path):
    result = link(gcc, tmp_path, M55)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("archive_flags", [M4, M0, M55_AV, M55_SR, ["-mcpu=cortex-m55"]])
def test_link_against_another_build_fails(gcc, tmp_path, archive_flags):
    result = link(gcc, tmp_path, archive_flags)
    assert result.returncode != 0
    assert "undefined reference to `arm_nn_build_mve_av0_sr0_gcc'" in result.stderr


def test_link_against_a_library_without_a_stamp_fails(gcc, tmp_path):
    other = tmp_path / "other.c"
    other.write_text("int unrelated(void) { return 0; }\n")
    assert compile_c(gcc, M55, other, tmp_path / "other.o").returncode == 0
    result = link(gcc, tmp_path, M55, objects=[tmp_path / "other.o"])
    assert "undefined reference to `arm_nn_build_mve_av0_sr0_gcc'" in result.stderr


def test_probe_guard_round_trip(gcc, tmp_path):
    impls = load_implementations(ROOT, load_signatures(ROOT))
    picked = [impls.require("depthwise_conv_s8_opt_3x3")]
    build = probe_compiler(ROOT, gcc, M55)
    source = guard_source(tmp_path, render_guard(build.target, build.toolchain, picked))
    assert compile_c(gcc, M55, source, tmp_path / "ok.o").returncode == 0
    assert errors(compile_c(gcc, M55_AV, source, tmp_path / "av.o")) == [
        "ns-cmsis-nn build check: planned without ARM_MATH_AUTOVECTORIZE, this build defines it",
        "ns-cmsis-nn build check: depthwise_conv_s8_opt_3x3 requires !autovectorize"]
    assert errors(compile_c(gcc, M4, source, tmp_path / "m4.o")) == [
        "ns-cmsis-nn build check: " + ISA_ERRORS[("mve", "dsp")],
        "ns-cmsis-nn build check: depthwise_conv_s8_opt_3x3 requires mvei"]
    with pytest.raises(BuildTargetError, match="requires mvei"):
        render_guard(probe_compiler(ROOT, gcc, M4).target, "gcc", picked)
