# SPDX-License-Identifier: LicenseRef-Ambiq-Apollo-SDK
"""The host planner library (Planner/): routes and sizes per target, and its refusals.

Built from Planner/sources.txt with the host compiler for both target toolchains and driven through ctypes. A plan's
sizes are checked against the library's own scratch-size queries in the same build, so the planner cannot drift
from what the firmware asks for.
"""

from __future__ import annotations

import ctypes
import json
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


class SizeOperands(ctypes.Structure):
    _fields_ = [("dw_conv_params", ctypes.POINTER(DwConvParams)), ("input_dims", ctypes.POINTER(Dims)),
                ("filter_dims", ctypes.POINTER(Dims)), ("output_dims", ctypes.POINTER(Dims)),
                ("dim_dst_width", ctypes.POINTER(ctypes.c_int32)), ("ch_src", ctypes.POINTER(ctypes.c_int32))]


def _check_struct_layouts(tmp: Path) -> None:
    """The ctypes mirrors above must match the C structs, field for field."""
    probe = tmp / "layout.c"
    lines = ['#include <stdio.h>', '#include <stddef.h>', '#include "arm_nn_plan.h"', "int main(void) {"]
    for cname, py in (("cmsis_nn_dw_conv_params", DwConvParams), ("cmsis_nn_pool_params", PoolParams),
                      ("cmsis_nn_fc_params", FcParams), ("cmsis_nn_quant_params", QuantParams),
                      ("cmsis_nn_per_channel_quant_params", PerChannelQuant), ("arm_nn_plan_target", Target),
                      ("arm_nn_plan_result", Plan), ("arm_nn_plan_size_operands", SizeOperands)):
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
                      ("arm_nn_plan_result", Plan), ("arm_nn_plan_size_operands", SizeOperands)):
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
    (lambda p, i, f, o: (setattr(i, "n", 16), setattr(i, "h", 2**30), setattr(i, "w", 2**30), setattr(o, "n", 16)),
     E_RANGE),  # element count wraps int64 to 0
])
def test_invalid_depthwise_layers_are_refused(libs, change, status):
    params, i, f, o = dw_layer()
    change(params, i, f, o)
    assert plan_dw(libs[GCC], target(), params, i, f, o)[0] == status


TO_CONV, DW3X3, OPT, DW = range(4)


# The selected kernel's own argument limits (ARM_CMSIS_NN_ARG_ERROR on the target) refuse a plan the route accepts.
# (output - 1) * stride - padding is the generic kernel's first tap index, kept in int16_t; its values are uint16_t.
@pytest.mark.parametrize("isa", [SCALAR, DSP, MVE])
@pytest.mark.parametrize("layer, route, statuses", [
    # 65536 channels route opt: the DSP channelwise kernel counts channels in uint16_t, and without DSP or MVE the
    # entry runs arm_depthwise_conv_s8(), whose generic kernel takes uint16_t values; only MVE runs it.
    (dict(c=65536, h=1, w=1, k=(1, 1)), OPT, {SCALAR: E_RANGE, DSP: E_RANGE, MVE: OK}),
    (dict(c=65535, h=1, w=1, k=(1, 1)), OPT, {SCALAR: OK, DSP: OK, MVE: OK}),
    # 257x257 taps pass uint16 as a count of channels but not as kw * kh.
    (dict(c=8, h=257, w=257, k=(257, 257)), OPT, {SCALAR: OK, DSP: E_RANGE, MVE: OK}),
    # Generic route (N = 2): a 32769-wide output puts the first tap index at 32768 on every build.
    (dict(c=1, h=1, w=32769, k=(1, 1), n=2), DW, {SCALAR: E_RANGE, DSP: E_RANGE, MVE: E_RANGE}),
    (dict(c=1, h=1, w=32768, k=(1, 1), n=2), DW, {SCALAR: OK, DSP: OK, MVE: OK}),
    # Generic route past uint16_t in a channel count...
    (dict(c=65536, h=1, w=1, k=(1, 1), n=2), DW, {SCALAR: E_RANGE, DSP: E_RANGE, MVE: E_RANGE}),
    # ...unless ch_mult is a multiple of 4 with N = 1 and no dilation: that kernel has no uint16 values.
    (dict(c=65536, ch_mult=4, h=1, w=1, k=(1, 1)), DW, {SCALAR: OK, DSP: OK, MVE: OK}),
    (dict(c=65536, ch_mult=4, h=1, w=1, k=(1, 1), n=2), DW, {SCALAR: E_RANGE, DSP: E_RANGE, MVE: E_RANGE}),
])
def test_depthwise_plan_applies_the_selected_kernels_limits(libs, isa, layer, route, statuses):
    params, i, f, o = dw_layer(**layer)
    status, plan = plan_dw(libs[GCC], target(isa), params, i, f, o)
    assert status == statuses[isa]
    if status == OK:
        assert plan.route == route


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
    # Both kernels reject a window that misses the input: the second 1x1 window of a 1-wide input starts past it.
    for max_pool in (0, 1):
        assert plan_pool(lib, target(), max_pool, params, dims(1, 1, 1, 1), dims(1, 1, 1, 1), dims(1, 1, 2, 1))[0] \
            == E_SHAPE
        assert plan_pool(lib, target(), max_pool, params, dims(1, 1, 1, 1), dims(1, 1, 1, 1), dims(1, 2, 1, 1))[0] \
            == E_SHAPE
        assert plan_pool(lib, target(), max_pool, params, dims(1, 1, 1, 1), dims(1, 1, 1, 1), dims(1, 1, 1, 1))[0] \
            == OK
    # A padding wider than the window leaves the first window empty.
    padded = PoolParams(Tile(1, 1), Tile(2, 0), Activation(-128, 127))
    assert plan_pool(lib, target(), 0, padded, dims(1, 4, 4, 3), dims(1, 2, 2, 1), dims(1, 3, 3, 3))[0] == E_SHAPE
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


PLANAR, CHANNELWISE = range(2)


def plan_dw_opt(lib, tgt, params, i, f, o, ctx_size):
    plan = Plan()
    status = lib.arm_nn_plan_depthwise_conv_s8_opt(ctypes.byref(tgt), ctypes.byref(params), ctypes.byref(i),
                                                   ctypes.byref(f), ctypes.byref(o), ctypes.c_int32(ctx_size),
                                                   ctypes.byref(plan))
    return status, plan


def opt_scratch(lib, isa, i, f):
    sizer = {MVE: lib.arm_depthwise_conv_s8_opt_get_buffer_size_mve,
             DSP: lib.arm_depthwise_conv_s8_opt_get_buffer_size_dsp,
             SCALAR: lib.arm_depthwise_conv_s8_opt_get_buffer_size}[isa]
    return sizer(ctypes.byref(i), ctypes.byref(f))


def planar_bytes(lib, params, i, f, o):
    return lib.arm_nn_depthwise_conv_s8_planar_bytes(ctypes.byref(params), ctypes.byref(i), ctypes.byref(f),
                                                     ctypes.byref(o))


# A 3x3 layer the planar path takes: an 8 + 2 by 8 + 2 plane plus 32 bytes of slack.
PLANAR_3X3 = dict(c=8, h=10, w=10)
PLANAR_3X3_PLANE = 10 * 10 + 32


@pytest.mark.parametrize("toolchain", [GCC, ARMCLANG])
@pytest.mark.parametrize("isa, autovectorize", [(SCALAR, 0), (DSP, 0), (MVE, 0), (MVE, 1)])
def test_depthwise_opt_plan_reports_the_route_and_scratch(libs, toolchain, isa, autovectorize):
    lib = libs[toolchain]
    params, i, f, o = dw_layer(**PLANAR_3X3)
    assert planar_bytes(lib, params, i, f, o) == PLANAR_3X3_PLANE
    scratch = opt_scratch(lib, isa, i, f)
    tgt = target(isa, autovectorize, toolchain=toolchain)
    for ctx_size in (0, scratch, PLANAR_3X3_PLANE):
        status, plan = plan_dw_opt(lib, tgt, params, i, f, o, ctx_size)
        planar = isa == MVE and ctx_size >= PLANAR_3X3_PLANE
        if not planar and 0 < ctx_size < scratch:
            assert status == E_RANGE, ctx_size
            continue
        assert status == OK, ctx_size
        assert plan.route == (PLANAR if planar else CHANNELWISE)
        assert plan.kernel == (b"arm_depthwise_conv_s8_opt_planar" if planar else
                               b"arm_depthwise_conv_s8_opt_channelwise")
        assert plan.route_id == (b"dw_opt_s8:planar" if planar else b"dw_opt_s8:channelwise")
        assert plan.ctx_bytes == scratch and plan.prepared_bytes == 0
        assert plan.ctx_role == (CTX_SCRATCH if scratch else CTX_NONE)


@pytest.mark.parametrize("isa, status", [(SCALAR, E_RANGE), (DSP, OK), (MVE, OK)])
def test_depthwise_opt_scalar_channelwise_is_the_generic_kernel(libs, isa, status):
    """Without DSP or MVE arm_depthwise_conv_s8_opt() runs arm_depthwise_conv_s8(), whose tap index is int16_t."""
    params, i, f, o = dw_layer(c=1, h=1, w=32769, k=(1, 1))
    assert plan_dw_opt(libs[GCC], target(isa), params, i, f, o, 0)[0] == status
    params, i, f, o = dw_layer(c=1, h=1, w=32768, k=(1, 1))
    assert plan_dw_opt(libs[GCC], target(isa), params, i, f, o, 0)[0] == OK


@pytest.mark.parametrize("isa", [MVE, DSP])
def test_depthwise_opt_ctx_one_byte_short_of_the_plane(libs, isa):
    """Planar declines and channelwise rejects a declared ctx->size below its scratch, so the call fails."""
    lib = libs[GCC]
    params, i, f, o = dw_layer(**PLANAR_3X3)
    assert opt_scratch(lib, isa, i, f) > PLANAR_3X3_PLANE
    status, _ = plan_dw_opt(lib, target(isa), params, i, f, o, PLANAR_3X3_PLANE - 1)
    assert status == E_RANGE
    status, plan = plan_dw_opt(lib, target(isa), params, i, f, o, PLANAR_3X3_PLANE)
    if isa == MVE:
        assert status == OK and plan.route == PLANAR
    else:
        assert status == E_RANGE


@pytest.mark.parametrize("width, planar", [(464, True), (465, False)])
def test_depthwise_opt_plane_against_the_mve_scratch(libs, width, planar):
    """A 1x1 filter has a 4 * 124-byte MVE scratch; a 1-row plane of W + 32 bytes fits it up to W = 464."""
    lib = libs[GCC]
    params, i, f, o = dw_layer(c=4, h=1, w=width, k=(1, 1))
    scratch = opt_scratch(lib, MVE, i, f)
    assert scratch == 4 * 124
    assert planar_bytes(lib, params, i, f, o) == width + 32
    status, plan = plan_dw_opt(lib, target(MVE), params, i, f, o, 1 << 20)
    assert status == OK and plan.route == (PLANAR if planar else CHANNELWISE)


@pytest.mark.parametrize("layer, planar", [
    (dict(c=33, h=10, w=10), False),               # more than 32 channels
    (dict(c=8, h=11, w=11, stride=(2, 2)), False),  # not the planar candidate
    (dict(c=8, h=12, w=12, dilation=(2, 1)), True),
    (dict(c=8, h=10, w=10, n=1, pad=(1, 1)), True),
    (dict(c=16, h=6, w=6), False),                 # narrow plane: not profitable
])
def test_depthwise_opt_plan_follows_the_planar_rule(libs, layer, planar):
    lib = libs[GCC]
    params, i, f, o = dw_layer(**layer)
    plane = planar_bytes(lib, params, i, f, o)
    assert (0 <= plane <= opt_scratch(lib, MVE, i, f)) == planar
    status, plan = plan_dw_opt(lib, target(MVE), params, i, f, o, 1 << 20)
    assert status == OK and plan.route == (PLANAR if planar else CHANNELWISE)


def test_depthwise_opt_dsp_counter_limits(libs):
    lib = libs[GCC]
    wide = dw_layer(c=65536, h=1, w=1, k=(1, 1))
    taps = dw_layer(c=1, h=256, w=256, k=(256, 256))
    for params, i, f, o in (wide, taps):
        assert plan_dw_opt(lib, target(DSP), params, i, f, o, 0)[0] == E_RANGE
        status, plan = plan_dw_opt(lib, target(MVE), params, i, f, o, 0)
        assert status == OK and plan.route == CHANNELWISE
    params, i, f, o = dw_layer(c=65535, h=1, w=1, k=(1, 1))
    assert plan_dw_opt(lib, target(DSP), params, i, f, o, 0)[0] == OK


@pytest.mark.parametrize("change, status", [
    (lambda p, i, f, o: (setattr(p, "ch_mult", 2), setattr(o, "c", 16), setattr(f, "c", 16)), E_SHAPE),
    (lambda p, i, f, o: setattr(p.dilation, "h", 2), E_SHAPE),
    (lambda p, i, f, o: setattr(p.dilation, "w", 0), E_SHAPE),
    (lambda p, i, f, o: setattr(f, "c", 9), E_SHAPE),
    (lambda p, i, f, o: setattr(i, "n", 0), E_SHAPE),
    (lambda p, i, f, o: (setattr(i, "h", 65536), setattr(i, "w", 65536)), E_RANGE),
])
def test_invalid_depthwise_opt_layers_are_refused(libs, change, status):
    params, i, f, o = dw_layer(**PLANAR_3X3)
    change(params, i, f, o)
    assert plan_dw_opt(libs[GCC], target(), params, i, f, o, 0)[0] == status


def test_depthwise_opt_refusals(libs):
    lib = libs[GCC]
    params, i, f, o = dw_layer(**PLANAR_3X3)
    assert plan_dw_opt(lib, target(), params, i, f, o, -1)[0] == E_RANGE
    assert plan_dw_opt(lib, target(toolchain=ARMCLANG), params, i, f, o, 0)[0] == E_TARGET
    assert plan_dw_opt(lib, target(isa=DSP, autovectorize=1), params, i, f, o, 0)[0] == E_TARGET
    tgt = target()
    for args in ((None, params, i, f, o), (tgt, None, i, f, o), (tgt, params, None, f, o), (tgt, params, i, f, None)):
        refs = [ctypes.byref(a) if a is not None else None for a in args]
        assert lib.arm_nn_plan_depthwise_conv_s8_opt(*refs, ctypes.c_int32(0), ctypes.byref(Plan())) == E_NULL
    assert lib.arm_nn_plan_depthwise_conv_s8_opt(ctypes.byref(tgt), ctypes.byref(params), ctypes.byref(i),
                                                 ctypes.byref(f), ctypes.byref(o), ctypes.c_int32(0), None) == E_NULL


def buffer_bytes(lib, tgt, symbol, **operands):
    refs = {}
    for name, value in operands.items():
        refs[name] = ctypes.pointer(ctypes.c_int32(value) if isinstance(value, int) else value)
    out = ctypes.c_int32(-12345)
    status = lib.arm_nn_plan_buffer_bytes(ctypes.byref(tgt), symbol.encode() if symbol is not None else None,
                                          ctypes.byref(SizeOperands(**refs)), ctypes.byref(out))
    return status, out.value


def direct(lib, isa, base):
    """The library's query of `base` for an ISA: its _mve or _dsp leg, or the dispatching query of this scalar build."""
    return getattr(lib, {MVE: f"{base}_mve", DSP: f"{base}_dsp", SCALAR: base}[isa])


@pytest.mark.parametrize("toolchain", [GCC, ARMCLANG])
@pytest.mark.parametrize("isa, autovectorize", [(SCALAR, 0), (DSP, 0), (MVE, 0), (MVE, 1)])
def test_buffer_bytes_equal_the_library_queries(libs, toolchain, isa, autovectorize):
    lib = libs[toolchain]
    tgt = target(isa, autovectorize, toolchain=toolchain)
    for width, ch in ((1, 1), (4, 13), (7, 300)):
        status, got = buffer_bytes(lib, tgt, "arm_avgpool_s8_get_buffer_size", dim_dst_width=width, ch_src=ch)
        assert status == OK and got == direct(lib, isa, "arm_avgpool_s8_get_buffer_size")(width, ch)
    for layer in (dict(c=8, pad=(1, 1)), dict(c=1, ch_mult=9), dict(c=13, k=(5, 5), h=9, w=9), dict(c=4, ch_mult=2)):
        params, i, f, o = dw_layer(**layer)
        status, got = buffer_bytes(lib, tgt, "arm_depthwise_conv_wrapper_s8_get_buffer_size", dw_conv_params=params,
                                   input_dims=i, filter_dims=f, output_dims=o)
        want = direct(lib, isa, "arm_depthwise_conv_wrapper_s8_get_buffer_size")(
            ctypes.byref(params), ctypes.byref(i), ctypes.byref(f), ctypes.byref(o))
        assert status == OK and got == want
        status, got = buffer_bytes(lib, tgt, "arm_depthwise_conv_s8_opt_get_buffer_size", input_dims=i, filter_dims=f)
        assert status == OK and got == opt_scratch(lib, isa, i, f)
        status, got = buffer_bytes(lib, tgt, "arm_convolve_s8_get_weights_sum_size", output_dims=o)
        assert status == OK and got == (4 * o.c if isa == MVE else 0)
        if isa != MVE:
            assert got == lib.arm_convolve_s8_get_weights_sum_size(ctypes.byref(o))
    for i in (dims(1, 5, 3, 16), dims(1, 9, 41, 64), dims(1, 1, 2048, 2048)):
        status, got = buffer_bytes(lib, tgt, "arm_depthwise_conv_s8_opt_3x3_get_buffer_size", input_dims=i)
        assert status == OK and got == lib.arm_depthwise_conv_s8_opt_3x3_get_buffer_size(ctypes.byref(i))
        assert got == 16 * (3 * 52 + 32) + i.w * i.c + 16
    for f in (dims(21, 1, 1, 5), dims(23, 1, 1, 9), dims(1, 1, 1, 1)):
        status, got = buffer_bytes(lib, tgt, "arm_fully_connected_s8_get_buffer_size", filter_dims=f)
        assert status == OK and got == direct(lib, isa, "arm_fully_connected_s8_get_buffer_size")(ctypes.byref(f))
        status, got = buffer_bytes(lib, tgt, "arm_fully_connected_per_channel_packed_s8_get_packed_size",
                                   filter_dims=f)
        assert status == OK
        assert got == lib.arm_fully_connected_per_channel_packed_s8_get_packed_size(ctypes.byref(f)) > 0


def test_every_manifest_size_query_is_answered(libs):
    manifest = json.loads((ROOT / "Tests" / "KernelContracts" / "kernel_implementations.json").read_text())
    params, i, f, o = dw_layer(**PLANAR_3X3)
    values = dict(dw_conv_params=params, input_dims=i, filter_dims=f, output_dims=o, dim_dst_width=8, ch_src=8)
    queries = {(b["size_query"]["symbol"], tuple(sorted(b["size_query"]["args"])))
               for record in manifest["implementations"] for b in record["buffers"] if b["size_query"]}
    assert queries
    for symbol, args in queries:
        assert set(args) <= set(values), (symbol, args)
        for isa in (SCALAR, DSP, MVE):
            status, got = buffer_bytes(libs[GCC], target(isa), symbol, **{a: values[a] for a in args})
            assert status == OK and got >= 0, (symbol, isa)


def test_buffer_bytes_refusals(libs):
    lib = libs[GCC]
    params, i, f, o = dw_layer(**PLANAR_3X3)
    tgt = target()
    opt = "arm_depthwise_conv_s8_opt_get_buffer_size"
    assert buffer_bytes(lib, tgt, "arm_depthwise_conv_s8_get_buffer_size", input_dims=i, filter_dims=f) == \
        (E_UNSUPPORTED, -12345)
    assert buffer_bytes(lib, tgt, None, input_dims=i, filter_dims=f)[0] == E_NULL
    assert buffer_bytes(lib, target(toolchain=ARMCLANG), opt, input_dims=i, filter_dims=f)[0] == E_TARGET
    assert buffer_bytes(lib, tgt, opt, input_dims=i) == (E_NULL, -12345)
    assert buffer_bytes(lib, tgt, opt, input_dims=i, filter_dims=f, output_dims=o) == (E_SHAPE, -12345)
    assert buffer_bytes(lib, tgt, "arm_avgpool_s8_get_buffer_size", dim_dst_width=4)[0] == E_NULL
    assert buffer_bytes(lib, tgt, "arm_avgpool_s8_get_buffer_size", dim_dst_width=4, ch_src=4, input_dims=i)[0] \
        == E_SHAPE
    out = ctypes.c_int32()
    operands = SizeOperands(input_dims=ctypes.pointer(i), filter_dims=ctypes.pointer(f))
    assert lib.arm_nn_plan_buffer_bytes(ctypes.byref(tgt), opt.encode(), None, ctypes.byref(out)) == E_NULL
    assert lib.arm_nn_plan_buffer_bytes(ctypes.byref(tgt), opt.encode(), ctypes.byref(operands), None) == E_NULL
    assert lib.arm_nn_plan_buffer_bytes(None, opt.encode(), ctypes.byref(operands), ctypes.byref(out)) == E_NULL


@pytest.mark.parametrize("isa, symbol, operands, status", [
    (MVE, "arm_depthwise_conv_s8_opt_get_buffer_size", dict(input_dims=dims(1, 4, 4, 0), filter_dims=dims(1, 3, 3, 8)),
     E_SHAPE),
    (MVE, "arm_depthwise_conv_s8_opt_get_buffer_size", dict(input_dims=dims(1, 4, 4, 8), filter_dims=dims(1, -3, 3, 8)),
     E_SHAPE),
    (MVE, "arm_avgpool_s8_get_buffer_size", dict(dim_dst_width=0, ch_src=4), E_SHAPE),
    (MVE, "arm_avgpool_s8_get_buffer_size", dict(dim_dst_width=4, ch_src=-1), E_SHAPE),
    # The element count passes int32_t.
    (MVE, "arm_depthwise_conv_s8_opt_3x3_get_buffer_size", dict(input_dims=dims(1, 1, 65536, 32768)), E_RANGE),
    # Within int32_t elements, but the size query's bytes are not: its -1.
    (MVE, "arm_depthwise_conv_s8_opt_3x3_get_buffer_size", dict(input_dims=dims(1, 1, 1, INT32_MAX - 100)), E_RANGE),
    (MVE, "arm_convolve_s8_get_weights_sum_size", dict(output_dims=dims(1, 1, 1, 1 << 29)), E_RANGE),
    (DSP, "arm_avgpool_s8_get_buffer_size", dict(dim_dst_width=1, ch_src=(1 << 29) + 1), E_RANGE),
    (MVE, "arm_fully_connected_s8_get_buffer_size", dict(filter_dims=dims(1, 1, 1, 1 << 29)), E_RANGE),
    (MVE, "arm_fully_connected_per_channel_packed_s8_get_packed_size",
     dict(filter_dims=dims(INT32_MAX // 2, 1, 1, 1)), E_RANGE),
])
def test_buffer_bytes_refuses_invalid_or_oversized_operands(libs, isa, symbol, operands, status):
    assert buffer_bytes(libs[GCC], target(isa), symbol, **operands) == (status, -12345)


def test_buffer_bytes_on_a_mismatched_depthwise_layer(libs):
    params, i, f, o = dw_layer(c=4, ch_mult=2)
    o.c = 9
    assert buffer_bytes(libs[GCC], target(), "arm_depthwise_conv_wrapper_s8_get_buffer_size", dw_conv_params=params,
                        input_dims=i, filter_dims=f, output_dims=o)[0] == E_SHAPE


def test_weights_sum_bytes_on_scalar_builds_ignore_the_channel_count(libs):
    status, got = buffer_bytes(libs[GCC], target(SCALAR), "arm_convolve_s8_get_weights_sum_size",
                               output_dims=dims(1, 1, 1, 1 << 29))
    assert (status, got) == (OK, 0)
