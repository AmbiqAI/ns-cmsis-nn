# SPDX-License-Identifier: LicenseRef-Ambiq-Apollo-SDK
"""The host planner library (Planner/): routes and sizes per target, and its refusals.

Built from Planner/sources.txt with the host compiler for both target toolchains and driven through ctypes. A plan's
sizes are checked against the library's own scratch-size queries in the same build, so the planner cannot drift
from what the firmware asks for.
"""

from __future__ import annotations

import ctypes
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
CC = shutil.which("cc") or shutil.which("gcc") or shutil.which("clang")
pytestmark = pytest.mark.skipif(CC is None, reason="host C compiler required")

SCALAR, DSP, MVE = 0, 1, 2
GCC, ARMCLANG = 0, 1
OK, E_NULL, E_TARGET, E_SHAPE, E_RANGE, E_UNSUPPORTED = range(6)
CTX_NONE, CTX_SCRATCH, CTX_KERNEL_SUMS = range(3)
INT32_MAX = 2**31 - 1


class Dims(ctypes.Structure):
    _fields_ = [("n", ctypes.c_int32), ("h", ctypes.c_int32), ("w", ctypes.c_int32), ("c", ctypes.c_int32)]


class Tile(ctypes.Structure):
    _fields_ = [("w", ctypes.c_int32), ("h", ctypes.c_int32)]


class Activation(ctypes.Structure):
    _fields_ = [("min", ctypes.c_int32), ("max", ctypes.c_int32)]


class DwConvParams(ctypes.Structure):
    _fields_ = [("input_offset", ctypes.c_int32), ("output_offset", ctypes.c_int32), ("ch_mult", ctypes.c_int32),
                ("stride", Tile), ("padding", Tile), ("dilation", Tile), ("activation", Activation)]


class PoolParams(ctypes.Structure):
    _fields_ = [("stride", Tile), ("padding", Tile), ("activation", Activation)]


class FcParams(ctypes.Structure):
    _fields_ = [("input_offset", ctypes.c_int32), ("filter_offset", ctypes.c_int32), ("output_offset", ctypes.c_int32),
                ("activation", Activation)]


class QuantParams(ctypes.Structure):
    _fields_ = [("multiplier", ctypes.POINTER(ctypes.c_int32)), ("shift", ctypes.POINTER(ctypes.c_int32)),
                ("is_per_channel", ctypes.c_bool)]


class PerChannelQuant(ctypes.Structure):
    _fields_ = [("multiplier", ctypes.POINTER(ctypes.c_int32)), ("shift", ctypes.POINTER(ctypes.c_int32))]


class Target(ctypes.Structure):
    _fields_ = [("isa", ctypes.c_int32), ("autovectorize", ctypes.c_int32), ("single_rounding", ctypes.c_int32),
                ("toolchain", ctypes.c_int32)]


class Plan(ctypes.Structure):
    _fields_ = [("kernel", ctypes.c_char_p), ("route_id", ctypes.c_char_p), ("route", ctypes.c_int32),
                ("ctx_role", ctypes.c_int32),
                ("ctx_bytes", ctypes.c_int32), ("prepared_bytes", ctypes.c_int32)]


def _check_struct_layouts(tmp: Path) -> None:
    """The ctypes mirrors above must match the C structs, field for field."""
    probe = tmp / "layout.c"
    lines = ['#include <stdio.h>', '#include <stddef.h>', '#include "arm_nn_plan.h"', "int main(void) {"]
    for cname, py in (("cmsis_nn_dw_conv_params", DwConvParams), ("cmsis_nn_pool_params", PoolParams),
                      ("cmsis_nn_fc_params", FcParams), ("cmsis_nn_quant_params", QuantParams),
                      ("cmsis_nn_per_channel_quant_params", PerChannelQuant), ("arm_nn_plan_target", Target),
                      ("arm_nn_plan_result", Plan)):
        lines.append(f'printf("{cname} %zu\\n", sizeof({cname}));')
        for field, _ in py._fields_:
            lines.append(f'printf("{cname}.{field} %zu\\n", offsetof({cname}, {field}));')
    lines.append("return 0; }")
    probe.write_text("\n".join(lines))
    exe = tmp / "layout"
    subprocess.run([CC, "-std=c11", f"-I{ROOT / 'Include'}", f"-I{ROOT / 'Planner/include'}", str(probe), "-o",
                    str(exe)], check=True, capture_output=True, text=True)
    got = dict(line.rsplit(" ", 1) for line in subprocess.run([str(exe)], capture_output=True, text=True,
                                                               check=True).stdout.splitlines())
    for cname, py in (("cmsis_nn_dw_conv_params", DwConvParams), ("cmsis_nn_pool_params", PoolParams),
                      ("cmsis_nn_fc_params", FcParams), ("cmsis_nn_quant_params", QuantParams),
                      ("cmsis_nn_per_channel_quant_params", PerChannelQuant), ("arm_nn_plan_target", Target),
                      ("arm_nn_plan_result", Plan)):
        assert int(got[cname]) == ctypes.sizeof(py), cname
        for field, _ in py._fields_:
            assert int(got[f"{cname}.{field}"]) == getattr(py, field).offset, f"{cname}.{field}"


def _build(tmp: Path, toolchain: int) -> ctypes.CDLL:
    sources = [str(ROOT / line) for line in (ROOT / "Planner" / "sources.txt").read_text().splitlines()
               if line.strip() and not line.startswith("#")]
    suffix = ".dylib" if sys.platform == "darwin" else ".so"
    lib = tmp / f"libplan_{toolchain}{suffix}"
    subprocess.run([CC, "-std=c11", "-O2", "-Wall", "-Wextra", "-Werror", "-fPIC", "-shared",
                    f"-DARM_NN_TOOLCHAIN_ARMCLANG={toolchain}", f"-I{ROOT / 'Include'}",
                    f"-I{ROOT / 'Planner/include'}", f"-I{ROOT / 'Planner/src'}", *sources, "-o", str(lib)],
                   check=True, capture_output=True, text=True)
    return ctypes.CDLL(str(lib))


@pytest.fixture(scope="module")
def libs(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("planner")
    _check_struct_layouts(tmp)
    return {GCC: _build(tmp, GCC), ARMCLANG: _build(tmp, ARMCLANG)}


def target(isa=MVE, autovectorize=0, single_rounding=0, toolchain=GCC):
    return Target(isa, autovectorize, single_rounding, toolchain)


def dims(n, h, w, c):
    return Dims(n, h, w, c)


def dw_params(ch_mult=1, stride=(1, 1), pad=(0, 0), dilation=(1, 1)):
    return DwConvParams(0, 0, ch_mult, Tile(*stride), Tile(*pad), Tile(*dilation), Activation(-128, 127))


def plan_dw(lib, tgt, params, i, f, o):
    plan = Plan()
    status = lib.arm_nn_plan_depthwise_conv_wrapper_s8(ctypes.byref(tgt), ctypes.byref(params), ctypes.byref(i),
                                                       ctypes.byref(f), ctypes.byref(o), ctypes.byref(plan))
    return status, plan


def dw_layer(c=8, ch_mult=1, h=6, w=6, k=(3, 3), n=1, **kw):
    params = dw_params(ch_mult, **kw)
    oh = (h + 2 * params.padding.h - (k[0] - 1) * params.dilation.h - 1) // params.stride.h + 1
    ow = (w + 2 * params.padding.w - (k[1] - 1) * params.dilation.w - 1) // params.stride.w + 1
    return params, dims(n, h, w, c), dims(1, k[0], k[1], c * ch_mult), dims(n, oh, ow, c * ch_mult)


def test_abi_and_toolchain(libs):
    assert libs[GCC].arm_nn_plan_abi_version() == 1
    assert libs[GCC].arm_nn_plan_toolchain_built() == GCC
    assert libs[ARMCLANG].arm_nn_plan_toolchain_built() == ARMCLANG


@pytest.mark.parametrize("bad", [
    target(isa=3), target(isa=-1), target(autovectorize=2), target(single_rounding=-1),
    target(isa=DSP, autovectorize=1), target(isa=SCALAR, autovectorize=1), target(toolchain=ARMCLANG),
    target(toolchain=2),
])
def test_an_invalid_or_foreign_target_is_refused(libs, bad):
    assert plan_dw(libs[GCC], bad, *dw_layer())[0] == E_TARGET


def test_null_arguments_are_refused(libs):
    lib = libs[GCC]
    params, i, f, o = dw_layer()
    plan = Plan()
    tgt = target()
    for args in ((None, params, i, f, o, plan), (tgt, None, i, f, o, plan), (tgt, params, None, f, o, plan),
                 (tgt, params, i, f, o, None)):
        refs = [ctypes.byref(a) if a is not None else None for a in args]
        assert lib.arm_nn_plan_depthwise_conv_wrapper_s8(*refs) == E_NULL


@pytest.mark.parametrize("change, status", [
    (lambda p, i, f, o: setattr(i, "c", 0), E_SHAPE),
    (lambda p, i, f, o: setattr(o, "h", -1), E_SHAPE),
    (lambda p, i, f, o: setattr(p, "ch_mult", 0), E_SHAPE),
    (lambda p, i, f, o: setattr(o, "c", 9), E_SHAPE),            # C_OUT != C_IN * ch_mult
    (lambda p, i, f, o: setattr(f, "n", 2), E_SHAPE),            # filter batch
    (lambda p, i, f, o: setattr(o, "n", 2), E_SHAPE),            # batch disagrees
    (lambda p, i, f, o: setattr(p.stride, "w", 0), E_SHAPE),
    (lambda p, i, f, o: setattr(p.padding, "h", -1), E_SHAPE),
    (lambda p, i, f, o: setattr(p.dilation, "h", 0), E_SHAPE),
    (lambda p, i, f, o: (setattr(i, "h", 65536), setattr(i, "w", 65536)), E_RANGE),  # element count past int32
])
def test_invalid_depthwise_layers_are_refused(libs, change, status):
    params, i, f, o = dw_layer()
    change(params, i, f, o)
    assert plan_dw(libs[GCC], target(), params, i, f, o)[0] == status


TO_CONV, DW3X3, OPT, DW = range(4)


@pytest.mark.parametrize("layer, gcc_routes, armclang_mve", [
    (dict(c=1, ch_mult=2), {SCALAR: DW, DSP: DW, MVE: TO_CONV}, DW),
    (dict(c=1, ch_mult=9), {SCALAR: DW, DSP: DW, MVE: TO_CONV}, TO_CONV),
    (dict(c=8, pad=(1, 1)), {SCALAR: DW3X3, DSP: DW3X3, MVE: OPT}, OPT),
    (dict(c=8, k=(5, 5), h=9, w=9), {SCALAR: OPT, DSP: OPT, MVE: OPT}, OPT),
    (dict(c=8, n=2), {SCALAR: DW, DSP: DW, MVE: DW}, DW),
    (dict(c=4, ch_mult=2), {SCALAR: DW, DSP: DW, MVE: DW}, DW),
])
def test_depthwise_routes_and_scratch_match_the_library(libs, layer, gcc_routes, armclang_mve):
    params, i, f, o = dw_layer(**layer)
    for toolchain, lib in libs.items():
        sizers = {MVE: lib.arm_depthwise_conv_wrapper_s8_get_buffer_size_mve,
                  DSP: lib.arm_depthwise_conv_wrapper_s8_get_buffer_size_dsp,
                  SCALAR: lib.arm_depthwise_conv_wrapper_s8_get_buffer_size}
        for isa in (SCALAR, DSP, MVE):
            status, plan = plan_dw(lib, target(isa, toolchain=toolchain), params, i, f, o)
            assert status == OK
            want = armclang_mve if (toolchain == ARMCLANG and isa == MVE) else gcc_routes[isa]
            assert plan.route == want, (toolchain, isa)
            assert plan.route_id == {TO_CONV: b"dw_wrapper_s8:to_conv", DW3X3: b"dw_wrapper_s8:conv_3x3",
                                     OPT: b"dw_wrapper_s8:opt", DW: b"dw_wrapper_s8:generic"}[want]
            size = sizers[isa](ctypes.byref(params), ctypes.byref(i), ctypes.byref(f), ctypes.byref(o))
            assert plan.ctx_bytes == size
            assert plan.ctx_role == (CTX_SCRATCH if size > 0 else CTX_NONE)
            assert plan.prepared_bytes == 0
    assert plan_dw(libs[GCC], target(MVE, autovectorize=1), params, i, f, o)[1].route == gcc_routes[MVE]


def plan_fc(lib, tgt, quant, i, f, o):
    plan = Plan()
    status = lib.arm_nn_plan_fully_connected_wrapper_s8(ctypes.byref(tgt), ctypes.byref(quant), ctypes.byref(i),
                                                        ctypes.byref(f), ctypes.byref(o), ctypes.byref(plan))
    return status, plan


def fc_quant(per_channel, n=4):
    mult = (ctypes.c_int32 * n)(*([1 << 30] * n))
    shift = (ctypes.c_int32 * n)(*([-1] * n))
    return QuantParams(mult, shift, per_channel), (mult, shift)


@pytest.mark.parametrize("per_channel", [True, False])
@pytest.mark.parametrize("isa", [SCALAR, DSP, MVE])
def test_fully_connected_routes_and_kernel_sums(libs, per_channel, isa):
    quant, _keep = fc_quant(per_channel, 5)
    status, plan = plan_fc(libs[GCC], target(isa), quant, dims(2, 1, 3, 7), dims(21, 1, 1, 5), dims(2, 1, 1, 5))
    assert status == OK
    assert plan.kernel == (b"arm_fully_connected_per_channel_s8" if per_channel else b"arm_fully_connected_s8")
    assert plan.route == (0 if per_channel else 1)
    assert plan.route_id == (b"fc_wrapper_s8:per_channel" if per_channel else b"fc_wrapper_s8:per_tensor")
    if isa == MVE:
        assert (plan.ctx_role, plan.ctx_bytes) == (CTX_KERNEL_SUMS, 5 * 4)
    else:
        assert (plan.ctx_role, plan.ctx_bytes) == (CTX_NONE, 0)


@pytest.mark.parametrize("i, f, o", [
    (dims(2, 1, 3, 7), dims(20, 1, 1, 5), dims(2, 1, 1, 5)),  # accumulation depth 21 != 20
    (dims(2, 1, 3, 7), dims(21, 1, 1, 5), dims(2, 1, 1, 6)),  # output depth
    (dims(2, 1, 3, 7), dims(21, 1, 1, 5), dims(1, 1, 1, 5)),  # batch
])
def test_fully_connected_shape_mismatch_is_refused(libs, i, f, o):
    quant, _keep = fc_quant(True, 6)
    assert plan_fc(libs[GCC], target(), quant, i, f, o)[0] == E_SHAPE


def plan_packed(lib, tgt, fc, quant, f):
    plan = Plan()
    status = lib.arm_nn_plan_fully_connected_per_channel_packed_s8(ctypes.byref(tgt), ctypes.byref(fc),
                                                                   ctypes.byref(quant), ctypes.byref(f),
                                                                   ctypes.byref(plan))
    return status, plan


def packed_quant(n, shift=-1):
    mult = (ctypes.c_int32 * n)(*([1 << 30] * n))
    sh = (ctypes.c_int32 * n)(*([shift] * n))
    return PerChannelQuant(mult, sh), (mult, sh)


def test_packed_fully_connected_plans_its_stream(libs):
    lib = libs[GCC]
    quant, _keep = packed_quant(9)
    f = dims(23, 1, 1, 9)
    status, plan = plan_packed(lib, target(MVE), FcParams(0, 0, 0, Activation(-128, 127)), quant, f)
    assert status == OK
    assert plan.kernel == b"arm_fully_connected_per_channel_packed_s8" and plan.route == -1
    assert plan.route_id is None
    assert (plan.ctx_role, plan.ctx_bytes) == (CTX_NONE, 0)
    assert plan.prepared_bytes == lib.arm_fully_connected_per_channel_packed_s8_get_packed_size(ctypes.byref(f))
    # ceil(9 / 4) blocks of 4 rows of 23 rounded up to 32, plus 48 bytes each.
    assert plan.prepared_bytes == 3 * (4 * 32 + 48)


@pytest.mark.parametrize("tgt, fc_offset, shift", [
    (target(SCALAR), 0, -1),
    (target(DSP), 0, -1),
    (target(MVE, autovectorize=1), 0, -1),
    (target(MVE, single_rounding=1), 0, -1),
    (target(MVE), 3, -1),   # filter offset
    (target(MVE), 0, 2),    # a left shift is outside the right-shift-only gate
])
def test_packed_fully_connected_declines(libs, tgt, fc_offset, shift):
    quant, _keep = packed_quant(9, shift)
    status, _ = plan_packed(libs[GCC], tgt, FcParams(0, fc_offset, 0, Activation(-128, 127)), quant,
                            dims(23, 1, 1, 9))
    assert status == E_UNSUPPORTED


def test_packed_fully_connected_stream_past_int32_is_a_range_error(libs):
    quant, _keep = packed_quant(1)
    status, _ = plan_packed(libs[GCC], target(MVE), FcParams(0, 0, 0, Activation(-128, 127)), quant,
                            dims(INT32_MAX // 2, 1, 1, 1))
    assert status == E_RANGE


def plan_pool(lib, tgt, is_max, params, i, f, o):
    plan = Plan()
    status = lib.arm_nn_plan_pool_s8(ctypes.byref(tgt), is_max, ctypes.byref(params), ctypes.byref(i),
                                     ctypes.byref(f), ctypes.byref(o), ctypes.byref(plan))
    return status, plan


@pytest.mark.parametrize("isa", [SCALAR, DSP, MVE])
def test_pooling_scratch_matches_the_library(libs, isa):
    lib = libs[GCC]
    params = PoolParams(Tile(2, 2), Tile(0, 0), Activation(-128, 127))
    i, f, o = dims(1, 8, 8, 13), dims(1, 2, 2, 1), dims(1, 4, 4, 13)
    status, plan = plan_pool(lib, target(isa), 0, params, i, f, o)
    sizer = {MVE: lib.arm_avgpool_s8_get_buffer_size_mve, DSP: lib.arm_avgpool_s8_get_buffer_size_dsp,
             SCALAR: lib.arm_avgpool_s8_get_buffer_size}[isa]
    assert status == OK and plan.kernel == b"arm_avgpool_s8"
    assert plan.ctx_bytes == sizer(o.w, i.c)
    assert plan.ctx_role == (CTX_SCRATCH if plan.ctx_bytes else CTX_NONE)
    status, plan = plan_pool(lib, target(isa), 1, params, i, f, o)
    assert status == OK and plan.kernel == b"arm_max_pool_s8" and (plan.ctx_role, plan.ctx_bytes) == (CTX_NONE, 0)


def test_pooling_refusals(libs):
    lib = libs[GCC]
    params = PoolParams(Tile(1, 1), Tile(0, 0), Activation(-128, 127))
    assert plan_pool(lib, target(), 2, params, dims(1, 4, 4, 3), dims(1, 2, 2, 1), dims(1, 3, 3, 3))[0] == E_SHAPE
    assert plan_pool(lib, target(), 0, params, dims(1, 4, 4, 3), dims(1, 2, 2, 1), dims(1, 3, 3, 4))[0] == E_SHAPE
    assert plan_pool(lib, target(), 0, params, dims(1, 4, 4, 3), dims(1, 0, 2, 1), dims(1, 3, 3, 3))[0] == E_SHAPE
    bad_stride = PoolParams(Tile(0, 1), Tile(0, 0), Activation(-128, 127))
    assert plan_pool(lib, target(), 0, bad_stride, dims(1, 4, 4, 3), dims(1, 2, 2, 1), dims(1, 3, 3, 3))[0] == E_SHAPE
    # The DSP scratch is C * 4 bytes; a depth whose byte count passes int32 is the sizers' -1, reported as a range error.
    wide = (1 << 29) + 1
    assert plan_pool(lib, target(DSP), 0, params, dims(1, 1, 1, wide), dims(1, 1, 1, 1), dims(1, 1, 1, wide))[0] \
        == E_RANGE


def test_the_planner_refuses_a_target_build(tmp_path):
    for flag in ("-DARM_MATH_DSP", "-DARM_MATH_MVEI"):
        out = subprocess.run([CC, "-std=c11", "-fsyntax-only", flag, f"-I{ROOT / 'Include'}",
                              f"-I{ROOT / 'Planner/include'}", f"-I{ROOT / 'Planner/src'}",
                              str(ROOT / "Planner/src/arm_nn_plan.c")], capture_output=True, text=True)
        assert out.returncode != 0 and "the ns-cmsis-nn planner is a host library" in out.stderr
