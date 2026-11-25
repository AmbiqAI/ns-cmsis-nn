# SPDX-License-Identifier: LicenseRef-Ambiq-Apollo-SDK
"""probe_compiler, unmet and render_guard on the host: the probe against scripted fake compilers (every way a compiler
can fail or print something unreadable), the requires evaluation over the real manifest, and the guard text. The real
cross compilers are exercised by scripts/tests/test_build_target_cross.py."""

import dataclasses
import itertools
import json
import os
import shutil
import sys
import textwrap

import pytest

from conftest import ROOT
from ns_cmsis_nn_contracts import (BuildTarget, BuildTargetError, ContractError, PlannerError, Planner, Target,
                                   build_planner, load_implementations, load_signatures, probe_compiler, render_guard,
                                   stamp_name, unmet)
from ns_cmsis_nn_contracts import vocabulary as voc

MVE_OUTPUT = {"isa": "2", "toolchain": "0", "stamp": "arm_nn_build_mve_av0_sr0_gcc", "has_mvei": "1", "has_dsp": "1",
              "has_mvef": "1", "has_mve_f16": "1", "has_f32_api": "0", "has_f16_api": "0", "has_autovectorize": "0",
              "has_single_rounding": "0"}


@pytest.fixture(scope="module")
def implementations():
    return load_implementations(ROOT, load_signatures(ROOT))


def fake_cc(tmp_path, *, lines=None, exit_code=0, stderr="", sleep=0.0, record=None):
    """A compiler command that prints ``lines`` (probe output) and exits with ``exit_code``."""
    script = tmp_path / "fake_cc.py"
    script.write_text(textwrap.dedent(f"""
        import json, sys, time
        if {record!r}:
            open({record!r}, "w").write(json.dumps(sys.argv[1:]))
        time.sleep({sleep!r})
        sys.stdout.write({json.dumps(chr(10).join(lines or []) + chr(10))})
        sys.stderr.write({stderr!r})
        sys.exit({exit_code!r})
    """))
    return [sys.executable, str(script)]


def probe_lines(values, extra=()):
    return [*extra, *(f"arm_nn_probe {name} {value}" for name, value in values.items())]


def test_probe_reads_every_field(tmp_path):
    build = probe_compiler(ROOT, fake_cc(tmp_path, lines=probe_lines(MVE_OUTPUT)))
    assert build == BuildTarget(target=Target("mve"), toolchain="gcc",
                                features=frozenset({"mvei", "dsp", "mvef", "mve_f16"}),
                                stamp="arm_nn_build_mve_av0_sr0_gcc")


def test_probe_ignores_other_preprocessed_lines(tmp_path):
    noise = ["typedef int int32_t;", "extern const uint8_t arm_nn_build_mve_av0_sr0_gcc;", "", "  "]
    values = dict(MVE_OUTPUT, has_single_rounding="1", stamp="arm_nn_build_mve_av0_sr1_gcc")
    build = probe_compiler(ROOT, fake_cc(tmp_path, lines=probe_lines(values, noise)))
    assert build.target == Target("mve", single_rounding=True)
    assert "single_rounding" in build.features


def test_probe_passes_the_flags_and_the_checkout_include(tmp_path):
    record = tmp_path / "argv.json"
    flags = ["-mcpu=cortex-m55", "-mfloat-abi=hard", "-DCMSIS_NN_USE_SINGLE_ROUNDING"]
    probe_compiler(ROOT, fake_cc(tmp_path, lines=probe_lines(MVE_OUTPUT), record=str(record)), flags)
    argv = json.loads(record.read_text())
    assert argv[:len(flags)] == flags
    assert argv[len(flags):len(flags) + 3] == ["-E", "-P", f"-I{ROOT / 'Include'}"]
    assert argv[-1].endswith("arm_nn_build_probe.c")


@pytest.mark.parametrize("change, message", [
    (lambda v: v.pop("has_mvei"), r"printed no value for \['has_mvei'\]"),
    (lambda v: v.update(isa="ARM_NN_BUILD_ISA"), "ARM_NN_BUILD_ISA is 'ARM_NN_BUILD_ISA'"),
    (lambda v: v.update(isa="3"), r"ARM_NN_BUILD_ISA is '3', not one of \[0, 1, 2\]"),
    (lambda v: v.update(isa="-1"), "ARM_NN_BUILD_ISA is '-1'"),
    (lambda v: v.update(toolchain="2"), "ARM_NN_BUILD_TOOLCHAIN is '2'"),
    (lambda v: v.update(has_dsp="2"), r"ARM_NN_BUILD_HAS_DSP is '2', not one of \[0, 1\]"),
    (lambda v: v.update(has_dsp=""), r"printed no value for \['has_dsp'\]"),
    (lambda v: v.update(stamp="arm_nn_build_dsp_av0_sr0_gcc"), "ARM_NN_BUILD_STAMP is 'arm_nn_build_dsp_av0_sr0_gcc'"),
    (lambda v: v.update(isa="1", has_mvei="0", stamp="arm_nn_build_dsp_av1_sr0_gcc", has_autovectorize="1"),
     "builds a target the planner does not model"),
])
def test_probe_refuses_unreadable_output(tmp_path, change, message):
    values = dict(MVE_OUTPUT)
    change(values)
    with pytest.raises(BuildTargetError, match=message):
        probe_compiler(ROOT, fake_cc(tmp_path, lines=probe_lines(values)))


def test_probe_refuses_a_duplicated_field(tmp_path):
    lines = probe_lines(MVE_OUTPUT) + ["arm_nn_probe isa 1"]
    with pytest.raises(BuildTargetError, match="probe field 'isa' appears twice"):
        probe_compiler(ROOT, fake_cc(tmp_path, lines=lines))


def test_probe_refuses_an_unknown_field(tmp_path):
    lines = probe_lines(MVE_OUTPUT) + ["arm_nn_probe quantum 1"]
    with pytest.raises(BuildTargetError, match="unexpected probe field 'quantum'"):
        probe_compiler(ROOT, fake_cc(tmp_path, lines=lines))


def test_probe_refuses_empty_output(tmp_path):
    with pytest.raises(BuildTargetError, match="printed no value"):
        probe_compiler(ROOT, fake_cc(tmp_path, lines=[]))


def test_probe_reports_a_failing_compiler(tmp_path):
    cc = fake_cc(tmp_path, exit_code=1, stderr="error: unrecognized -mcpu target: cortex-m99\n")
    with pytest.raises(BuildTargetError, match="exited with 1: error: unrecognized -mcpu target: cortex-m99"):
        probe_compiler(ROOT, cc, ["-mcpu=cortex-m99"])


def test_probe_reports_a_missing_compiler(tmp_path):
    with pytest.raises(BuildTargetError, match="compiler not found"):
        probe_compiler(ROOT, str(tmp_path / "no-such-cc"))


@pytest.mark.skipif(os.name == "nt", reason="POSIX permission bits")
def test_probe_reports_a_compiler_that_cannot_run(tmp_path):
    cc = tmp_path / "not-executable"
    cc.write_text("")
    cc.chmod(0o644)
    with pytest.raises(BuildTargetError, match="cannot be run"):
        probe_compiler(ROOT, str(cc))


def test_probe_times_out(tmp_path):
    with pytest.raises(BuildTargetError, match="did not finish within 0.5 s"):
        probe_compiler(ROOT, fake_cc(tmp_path, lines=probe_lines(MVE_OUTPUT), sleep=10), timeout=0.5)


def test_probe_refuses_a_checkout_without_the_header(tmp_path):
    with pytest.raises(BuildTargetError, match="predates the build-target check"):
        probe_compiler(tmp_path, fake_cc(tmp_path, lines=probe_lines(MVE_OUTPUT)))


@pytest.mark.parametrize("kwargs, message", [
    ({"cc": []}, "non-empty command"),
    ({"cc": ["gcc", ""]}, "non-empty command"),
    ({"cc": "gcc", "flags": "-mcpu=cortex-m55"}, "sequence of strings"),
    ({"cc": "gcc", "flags": ["-O2", 3]}, "sequence of strings"),
    ({"cc": "gcc", "timeout": 0}, "timeout must be positive"),
])
def test_probe_validates_its_arguments(kwargs, message):
    with pytest.raises(ValueError, match=message):
        probe_compiler(ROOT, **kwargs)


def test_stamp_name():
    assert stamp_name(Target("scalar"), "gcc") == "arm_nn_build_scalar_av0_sr0_gcc"
    assert stamp_name(Target("mve", autovectorize=True, single_rounding=True), "armclang") == \
        "arm_nn_build_mve_av1_sr1_armclang"
    with pytest.raises(ValueError, match="toolchain 'iar'"):
        stamp_name(Target("dsp"), "iar")


MVE = frozenset({"mvei", "dsp", "mvef", "mve_f16"})


@pytest.mark.parametrize("impl_id, features, expected", [
    ("depthwise_conv_s8_opt_3x3", MVE, ()),
    ("depthwise_conv_s8_opt_3x3", {"dsp"}, ("mvei",)),
    ("depthwise_conv_s8_opt_3x3", MVE | {"autovectorize"}, ("!autovectorize",)),
    ("depthwise_conv_3x3_s8", MVE, ("!mvei",)),
    ("depthwise_conv_3x3_s8", {"dsp"}, ()),
    ("fully_connected_per_channel_packed_s8", MVE | {"single_rounding"}, ("!single_rounding",)),
    ("fully_connected_per_channel_packed_s8", {"autovectorize", "single_rounding"},
     ("!autovectorize", "!single_rounding", "mvei")),
    ("depthwise_conv_s8", frozenset(), ()),
])
def test_unmet(implementations, impl_id, features, expected):
    assert unmet(implementations.require(impl_id), features) == expected


def test_unmet_is_consistent_over_every_record_and_build(implementations):
    for impl in implementations.implementations.values():
        tokens = set(impl.requires) | set(impl.record["routed_by"]["requires"] if impl.wrapper else [])
        for size in range(len(voc.BASE_FEATURES) + 1):
            for features in itertools.combinations(voc.BASE_FEATURES, size):
                result = unmet(impl, features)
                assert list(result) == sorted(result) and set(result) <= tokens
                assert all((t.lstrip("!") in features) == t.startswith("!") for t in result)
                assert all((t.lstrip("!") in features) != t.startswith("!") for t in tokens - set(result))


def test_unmet_refuses_unknown_tokens(implementations):
    impl = implementations.require("depthwise_conv_s8_opt_3x3")
    with pytest.raises(ValueError, match=r"\['quantum'\] are not manifest feature tokens"):
        unmet(impl, {"mvei", "quantum"})
    with pytest.raises(ValueError, match="are not manifest feature tokens"):
        unmet(impl, {"!mvei"})
    corrupt = dataclasses.replace(impl, requires=frozenset({"mvei", "quantum"}))
    with pytest.raises(ContractError, match="unknown feature token"):
        unmet(corrupt, MVE)


GUARD_MVE_3X3 = """\
/* ns-cmsis-nn build check (ns_cmsis_nn_contracts.render_guard) */
#define ARM_NN_EXPECT_ISA ARM_NN_ISA_MVE
#define ARM_NN_EXPECT_AUTOVECTORIZE 0
#define ARM_NN_EXPECT_SINGLE_ROUNDING 0
#define ARM_NN_EXPECT_TOOLCHAIN ARM_NN_TOOLCHAIN_ID_GCC
#include "arm_nn_build_target_check.h"
#if ARM_NN_BUILD_HAS_AUTOVECTORIZE
    #error "ns-cmsis-nn build check: depthwise_conv_s8_opt_3x3 requires !autovectorize"
#endif
#if !ARM_NN_BUILD_HAS_MVEI
    #error "ns-cmsis-nn build check: depthwise_conv_s8_opt_3x3 requires mvei"
#endif
"""


def test_render_guard_text(implementations):
    assert render_guard(Target("mve"), "gcc", [implementations.require("depthwise_conv_s8_opt_3x3")]) == GUARD_MVE_3X3


def test_render_guard_without_implementations():
    assert render_guard(Target("dsp", single_rounding=True), "armclang").splitlines()[1:] == [
        "#define ARM_NN_EXPECT_ISA ARM_NN_ISA_DSP", "#define ARM_NN_EXPECT_AUTOVECTORIZE 0",
        "#define ARM_NN_EXPECT_SINGLE_ROUNDING 1", "#define ARM_NN_EXPECT_TOOLCHAIN ARM_NN_TOOLCHAIN_ID_ARMCLANG",
        '#include "arm_nn_build_target_check.h"']


def test_render_guard_is_deterministic(implementations):
    picked = [implementations.require(i) for i in ("fully_connected_per_channel_packed_s8", "depthwise_conv_s8_opt_3x3",
                                                   "depthwise_conv_s8")]
    first = render_guard(Target("mve"), "gcc", picked)
    assert render_guard(Target("mve"), "gcc", list(reversed(picked)) + picked) == first
    assert first.count("#error") == 5


@pytest.mark.parametrize("target, impl_id, message", [
    (Target("dsp"), "depthwise_conv_s8_opt_3x3", r"requires mvei, which the planned build \(dsp, gcc\) lacks"),
    (Target("mve", autovectorize=True), "depthwise_conv_s8_opt_3x3",
     r"requires !autovectorize, which the planned build \(mve\+autovectorize, gcc\) lacks"),
    (Target("mve"), "depthwise_conv_3x3_s8", "requires !mvei"),
    (Target("mve", single_rounding=True), "fully_connected_per_channel_packed_s8", "requires !single_rounding"),
])
def test_render_guard_refuses_a_plan_its_target_cannot_run(implementations, target, impl_id, message):
    with pytest.raises(BuildTargetError, match=message):
        render_guard(target, "gcc", [implementations.require(impl_id)])


def test_render_guard_validates_its_arguments(implementations):
    impl = implementations.require("depthwise_conv_s8")
    with pytest.raises(TypeError, match="target must be a Target"):
        render_guard("mve", "gcc")
    with pytest.raises(ValueError, match="toolchain 'iar'"):
        render_guard(Target("mve"), "iar")
    with pytest.raises(TypeError, match="sequence of Implementation"):
        render_guard(Target("mve"), "gcc", impl)
    with pytest.raises(TypeError, match="sequence of Implementation"):
        render_guard(Target("mve"), "gcc", ["depthwise_conv_s8"])


def test_unknown_implementation_id_is_refused(implementations):
    with pytest.raises(ContractError, match="no such implementation"):
        implementations.require("depthwise_conv_s8_opt_9x9")


@pytest.mark.skipif(not (os.environ.get("CC") or shutil.which("cc") or shutil.which("gcc")),
                    reason="host C compiler required")
def test_planner_check_build(tmp_path, implementations):
    planner = Planner(build_planner(ROOT, "gcc", cache_dir=tmp_path), "gcc", implementations)
    gcc = BuildTarget(Target("mve"), "gcc", MVE, "arm_nn_build_mve_av0_sr0_gcc")
    assert planner.check_build(gcc) == Target("mve")
    with pytest.raises(PlannerError, match="uses toolchain armclang, this planner plans for gcc"):
        planner.check_build(dataclasses.replace(gcc, toolchain="armclang", stamp="arm_nn_build_mve_av0_sr0_armclang"))
