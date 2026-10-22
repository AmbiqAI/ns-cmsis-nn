# SPDX-License-Identifier: LicenseRef-Ambiq-Apollo-SDK
"""build_planner and Planner: the cache, the refusals of a library or manifest that does not match, every plan resolving
to a manifest implementation, and the preparation helpers against the definitions they implement."""

import ctypes
import dataclasses
import os
import random
import shutil
import subprocess

import pytest

from conftest import ROOT
from ns_cmsis_nn_contracts import (ContractError, PlannerError, PlanRefused, Planner, Target, build_planner,
                                   load_implementations, load_signatures, open_planner, route_enums)
from ns_cmsis_nn_contracts.planner import _Dims

pytestmark = pytest.mark.skipif(not (os.environ.get("CC") or shutil.which("cc") or shutil.which("gcc")),
                                reason="host C compiler required")

TARGETS = [Target("scalar"), Target("dsp"), Target("mve"), Target("mve", autovectorize=True),
           Target("mve", single_rounding=True), Target("dsp", single_rounding=True)]


@pytest.fixture(scope="module")
def cache(tmp_path_factory):
    return tmp_path_factory.mktemp("plan_cache")


@pytest.fixture(scope="module")
def implementations():
    return load_implementations(ROOT, load_signatures(ROOT))


@pytest.fixture(scope="module", params=["gcc", "armclang"])
def planner(request, cache, implementations):
    return Planner(build_planner(ROOT, request.param, cache_dir=cache), request.param, implementations)


def test_cache_reuses_and_invalidates(tree, tmp_path):
    cache = tmp_path / "cache"
    first = build_planner(tree, "gcc", cache_dir=cache)
    stamp = first.stat().st_mtime_ns
    assert build_planner(tree, "gcc", cache_dir=cache) == first
    assert first.stat().st_mtime_ns == stamp
    libraries = {first, build_planner(tree, "armclang", cache_dir=cache)}
    source = tree / "Planner" / "src" / "arm_nn_plan.c"
    source.write_text(source.read_text() + "\n")
    libraries.add(build_planner(tree, "gcc", cache_dir=cache))
    header = tree / "Include" / "arm_nnsupportfunctions.h"
    header.write_text(header.read_text() + "\n")
    libraries.add(build_planner(tree, "gcc", cache_dir=cache))
    assert len(libraries) == 4
    assert set(cache.iterdir()) == libraries


def test_failed_build_leaves_nothing(tree, tmp_path):
    (tree / "Planner" / "src" / "arm_nn_plan.c").write_text("#error broken on purpose\n")
    with pytest.raises(PlannerError, match="broken on purpose"):
        build_planner(tree, "gcc", cache_dir=tmp_path / "cache")
    assert list((tmp_path / "cache").iterdir()) == []


def test_build_refusals(tree, tmp_path):
    with pytest.raises(ValueError, match="toolchain"):
        build_planner(tree, "iar", cache_dir=tmp_path)
    with pytest.raises(PlannerError, match="does not run"):
        build_planner(tree, "gcc", cc=str(tmp_path / "no-such-cc"), cache_dir=tmp_path)
    (tree / "Planner" / "sources.txt").write_text("Planner/src/missing.c\n")
    with pytest.raises(PlannerError, match="missing.c"):
        build_planner(tree, "gcc", cache_dir=tmp_path)
    (tree / "Planner" / "sources.txt").write_text("# nothing\n")
    with pytest.raises(PlannerError, match="none listed"):
        build_planner(tree, "gcc", cache_dir=tmp_path)
    (tree / "Planner" / "sources.txt").unlink()
    with pytest.raises(PlannerError, match="no planner"):
        build_planner(tree, "gcc", cache_dir=tmp_path)


def test_library_refusals(cache, implementations, tmp_path):
    gcc = build_planner(ROOT, "gcc", cache_dir=cache)
    with pytest.raises(PlannerError, match="toolchain"):
        Planner(gcc, "armclang")
    with pytest.raises(PlannerError, match="planner ABI 2"):
        Planner(gcc, "gcc", dataclasses.replace(implementations, planner_abi=2))
    with pytest.raises(PlannerError, match="planner ABI None"):
        Planner(gcc, "gcc", dataclasses.replace(implementations, planner_abi=None))
    with pytest.raises(PlannerError, match="cannot be loaded"):
        Planner(tmp_path / "missing.so")
    (tmp_path / "junk.so").write_bytes(b"not a library")
    with pytest.raises(PlannerError, match="cannot be loaded"):
        Planner(tmp_path / "junk.so")


def test_abi_mismatch_in_the_library(tree, tmp_path):
    source = tree / "Planner" / "src" / "arm_nn_plan.c"
    text = source.read_text()
    assert "return ARM_NN_PLAN_ABI_VERSION;" in text
    source.write_text(text.replace("return ARM_NN_PLAN_ABI_VERSION;", "return ARM_NN_PLAN_ABI_VERSION + 1;"))
    with pytest.raises(PlannerError, match="planner ABI 2, this package reads 1"):
        Planner(build_planner(tree, "gcc", cache_dir=tmp_path))


def test_target_validation():
    with pytest.raises(ValueError, match="isa"):
        Target("neon")
    with pytest.raises(ValueError, match="mve target only"):
        Target("dsp", autovectorize=True)


def test_open_planner(cache):
    planner = open_planner(ROOT, "gcc", cache_dir=cache)
    assert planner.implementations.planner_abi == 1 and planner.toolchain == "gcc"


DW_GRID = [
    dict(input_dims=(1, 8, 8, 16), filter_dims=(1, 3, 3, 16), output_dims=(1, 8, 8, 16), ch_mult=1, padding=(1, 1)),
    dict(input_dims=(1, 8, 8, 16), filter_dims=(1, 3, 3, 16), output_dims=(1, 8, 8, 16), ch_mult=1, padding=(1, 1),
         dilation=(2, 2)),
    dict(input_dims=(1, 9, 9, 3), filter_dims=(1, 3, 3, 3), output_dims=(1, 7, 7, 3), ch_mult=1),
    dict(input_dims=(1, 16, 16, 8), filter_dims=(1, 5, 5, 8), output_dims=(1, 12, 12, 8), ch_mult=1),
    dict(input_dims=(1, 8, 8, 4), filter_dims=(1, 3, 3, 8), output_dims=(1, 6, 6, 8), ch_mult=2),
    dict(input_dims=(1, 1, 1, 32), filter_dims=(1, 1, 1, 32), output_dims=(1, 1, 1, 32), ch_mult=1),
    dict(input_dims=(2, 8, 8, 16), filter_dims=(1, 3, 3, 16), output_dims=(2, 6, 6, 16), ch_mult=1),
    dict(input_dims=(1, 4, 64, 16), filter_dims=(1, 1, 3, 16), output_dims=(1, 4, 62, 16), ch_mult=1),
]


@pytest.mark.parametrize("target", TARGETS, ids=str)
def test_every_plan_resolves_to_a_manifest_implementation(planner, target):
    seen = set()
    for layer in DW_GRID:
        plan = planner.depthwise_conv_wrapper_s8(target, **layer)
        assert plan.route is not None and plan.implementation.route == plan.route
        assert plan.implementation.entry == plan.kernel
        seen.add(plan.route)
    for per_channel in (False, True):
        plan = planner.fully_connected_wrapper_s8(target, input_dims=(1, 1, 1, 40), filter_dims=(40, 1, 1, 10),
                                                  output_dims=(1, 1, 1, 10), per_channel=per_channel)
        assert plan.implementation.entry == plan.kernel
        assert plan.ctx_role == ("kernel_sums" if target.isa == "mve" else "none")
        assert plan.ctx_bytes == (40 if target.isa == "mve" else 0)
    for max_pool in (False, True):
        plan = planner.pool_s8(target, max_pool=max_pool, input_dims=(1, 8, 8, 16), kernel_hw=(2, 2),
                               output_dims=(1, 4, 4, 16), stride=(2, 2))
        assert plan.route is None and plan.implementation.entry == plan.kernel
        assert plan.implementation.wrapper is None
    assert seen


def test_packed_plan_matches_the_stream(planner):
    plan = planner.fully_connected_per_channel_packed_s8(Target("mve"), filter_dims=(64, 1, 1, 16),
                                                         filter_offset=0, multipliers=[1 << 30] * 16,
                                                         shifts=[-1] * 16)
    assert plan.implementation.id == "fully_connected_per_channel_packed_s8"
    rng = random.Random(829)
    weights = [rng.randint(-128, 127) for _ in range(64 * 16)]
    sums = planner.vector_sum_s8(weights, cols=64, rows=16, lhs_offset=0, rhs_offset=0)
    stream = planner.pack_fully_connected_per_channel_packed_s8(weights, filter_dims=(64, 1, 1, 16),
                                                               kernel_sums=sums, multipliers=[1 << 30] * 16,
                                                               shifts=[-1] * 16)
    assert len(stream) == plan.prepared_bytes
    assert stream == planner.pack_fully_connected_per_channel_packed_s8(
        bytes(w & 0xFF for w in weights), filter_dims=(64, 1, 1, 16), kernel_sums=sums,
        multipliers=[1 << 30] * 16, shifts=[-1] * 16)


@pytest.mark.parametrize("target, filter_offset, shifts, status", [
    (Target("scalar"), 0, [-1] * 4, "unsupported"),
    (Target("dsp"), 0, [-1] * 4, "unsupported"),
    (Target("mve", autovectorize=True), 0, [-1] * 4, "unsupported"),
    (Target("mve", single_rounding=True), 0, [-1] * 4, "unsupported"),
    (Target("mve"), 1, [-1] * 4, "unsupported"),
    (Target("mve"), 0, [-1, -1, 0, -1], "unsupported"),
])
def test_packed_refusals(planner, target, filter_offset, shifts, status):
    with pytest.raises(PlanRefused) as refused:
        planner.fully_connected_per_channel_packed_s8(target, filter_dims=(8, 1, 1, 4), filter_offset=filter_offset,
                                                      multipliers=[1 << 30] * 4, shifts=shifts)
    assert refused.value.status == status


@pytest.mark.parametrize("change, status", [
    (dict(input_dims=(1, 8, 8, 0)), "shape"),
    (dict(output_dims=(1, -1, 8, 16)), "shape"),
    (dict(ch_mult=0), "shape"),
    (dict(output_dims=(1, 8, 8, 9)), "shape"),
    (dict(stride=(0, 1)), "shape"),
    (dict(dilation=(1, 0)), "shape"),
    (dict(input_dims=(1, 65536, 65536, 16)), "range"),
])
def test_layer_refusals(planner, change, status):
    layer = dict(DW_GRID[0], **change)
    with pytest.raises(PlanRefused) as refused:
        planner.depthwise_conv_wrapper_s8(Target("mve"), **layer)
    assert refused.value.status == status


@pytest.mark.parametrize("dims", [(1, 8, 8), (1, 8, 8, 16, 1), (1, 8, 8, 2**31), (1, 8, 8, 1.5), (1, 8, 8, True)])
def test_dims_checked_before_the_call(planner, dims):
    with pytest.raises(ValueError, match="input_dims"):
        planner.depthwise_conv_wrapper_s8(Target("mve"), **dict(DW_GRID[0], input_dims=dims))


def vector_sum(data, cols, rows, lhs_offset, rhs_offset, bias):
    return [(bias[r] if bias else 0) + lhs_offset * sum(v + rhs_offset for v in data[r * cols:(r + 1) * cols])
            for r in range(rows)]


@pytest.mark.parametrize("cols, rows, lhs_offset, rhs_offset, with_bias", [
    (1, 1, 0, 0, False), (3, 2, 1, 0, False), (17, 5, -3, 0, True), (64, 16, 128, 0, True), (33, 7, 1, 5, True),
    (255, 3, -128, -7, False),
])
def test_vector_sum_matches_its_definition(planner, cols, rows, lhs_offset, rhs_offset, with_bias):
    rng = random.Random(cols * 1000 + rows)
    data = [rng.randint(-128, 127) for _ in range(cols * rows)]
    bias = [rng.randint(-1000, 1000) for _ in range(rows)] if with_bias else None
    assert planner.vector_sum_s8(data, cols=cols, rows=rows, lhs_offset=lhs_offset, rhs_offset=rhs_offset,
                                 bias=bias) == vector_sum(data, cols, rows, lhs_offset, rhs_offset, bias)


@pytest.mark.parametrize("kwargs, match", [
    (dict(data=[1, 2, 3], cols=0, rows=1), "positive"),
    (dict(data=[1, 2, 3], cols=2, rows=1), "cols \\* rows"),
    (dict(data=[1, 2, 128], cols=3, rows=1), "int8"),
    (dict(data=[1, 2, 3], cols=3, rows=1, bias=[1, 2]), "bias"),
    (dict(data=[1, 2, 3], cols=3, rows=1, bias=[2**31]), "int32"),
])
def test_vector_sum_refusals(planner, kwargs, match):
    kwargs.setdefault("bias", None)
    with pytest.raises(ValueError, match=match):
        planner.vector_sum_s8(lhs_offset=0, rhs_offset=0, **kwargs)


@pytest.mark.parametrize("kwargs, error, match", [
    (dict(filter_data=[0] * 7), ValueError, "filter_data holds 7"),
    (dict(kernel_sums=[0] * 3), ValueError, "kernel_sums"),
    (dict(multipliers=[1] * 5), ValueError, "multipliers"),
    (dict(filter_dims=(0, 1, 1, 4)), ValueError, "positive depths"),
    (dict(shifts=[0] * 4), PlanRefused, "returned"),
])
def test_pack_refusals(planner, kwargs, error, match):
    args = dict(filter_data=[1] * 8, filter_dims=(2, 1, 1, 4), kernel_sums=[0] * 4, multipliers=[1 << 30] * 4,
                shifts=[-1] * 4)
    args.update(kwargs)
    with pytest.raises(error, match=match):
        planner.pack_fully_connected_per_channel_packed_s8(args.pop("filter_data"), **args)


def _direct(planner, target, base, *args):
    """The library's own query: its _mve or _dsp leg for the target, or the dispatching query of this scalar build."""
    name = {"mve": f"{base}_mve", "dsp": f"{base}_dsp", "scalar": base}[target.isa]
    fn = getattr(planner._lib, name)
    fn.restype = ctypes.c_int32
    return fn(*[ctypes.byref(_Dims(*a)) if isinstance(a, tuple) else a for a in args])


def _plain(planner, name, *dims):
    fn = getattr(planner._lib, name)
    fn.restype = ctypes.c_int32
    return fn(*[ctypes.byref(_Dims(*d)) for d in dims])


DW_3X3 = dict(input_dims=(1, 10, 10, 16), filter_dims=(1, 3, 3, 16), output_dims=(1, 8, 8, 16))


@pytest.mark.parametrize("target", TARGETS, ids=str)
def test_buffer_bytes_equal_the_direct_queries(planner, target):
    i, f, o = DW_3X3["input_dims"], DW_3X3["filter_dims"], DW_3X3["output_dims"]
    for impl in ("depthwise_conv_s8_opt_3x3", "depthwise_conv_s8_opt_3x3_c64_s1"):
        got = planner.buffer_bytes(target, impl, "scratch", input_dims=i)
        assert got == _plain(planner, "arm_depthwise_conv_s8_opt_3x3_get_buffer_size", i)
        assert got == 16 * (3 * 52 + 32) + 10 * 16 + 16
    for impl in ("depthwise_conv_s8_opt", "depthwise_conv_s8_opt_planar", "depthwise_conv_s8_opt_channelwise"):
        assert planner.buffer_bytes(target, impl, "scratch", input_dims=i, filter_dims=f) == \
            _direct(planner, target, "arm_depthwise_conv_s8_opt_get_buffer_size", i, f)
        assert planner.buffer_bytes(target, impl, "weight_sums", output_dims=o) == \
            (16 * 4 if target.isa == "mve" else 0)
    to_conv = dict(input_dims=(1, 6, 6, 1), filter_dims=(1, 3, 3, 9), output_dims=(1, 4, 4, 9))
    plan = planner.depthwise_conv_wrapper_s8(target, ch_mult=9, **to_conv)
    assert planner.buffer_bytes(target, "depthwise_conv_s8_to_conv", "scratch", dw_conv_params={"ch_mult": 9},
                                **to_conv) == plan.ctx_bytes
    for impl in ("fully_connected_s8", "fully_connected_per_channel_s8"):
        assert planner.buffer_bytes(target, impl, "kernel_sums", filter_dims=(40, 1, 1, 10)) == \
            _direct(planner, target, "arm_fully_connected_s8_get_buffer_size", (40, 1, 1, 10))
    packed = planner.buffer_bytes(target, "fully_connected_per_channel_packed_s8", "packed_weights",
                                  filter_dims=(23, 1, 1, 9))
    assert packed == _plain(planner, "arm_fully_connected_per_channel_packed_s8_get_packed_size", (23, 1, 1, 9))
    assert planner.buffer_bytes(target, "avg_pool_s8", "accumulators", output_dims=(1, 4, 4, 13),
                                input_dims=(1, 8, 8, 13)) == \
        _direct(planner, target, "arm_avgpool_s8_get_buffer_size", 4, 13)


def test_every_manifest_size_query_is_answered(planner, implementations):
    layer = dict(input_dims=(1, 10, 10, 8), filter_dims=(1, 3, 3, 8), output_dims=(1, 8, 8, 8),
                 dw_conv_params={"ch_mult": 1})
    asked = 0
    for impl in implementations.implementations.values():
        for buffer in impl.buffers:
            if buffer.size_query is None:
                continue
            roots = {ref.split(".")[0] for ref in buffer.size_args.values()}
            for target in TARGETS:
                assert planner.buffer_bytes(target, impl.id, buffer.name, **{r: layer[r] for r in roots}) >= 0
            asked += 1
    assert asked >= 15


@pytest.mark.parametrize("target", TARGETS, ids=str)
def test_dw_opt_plan_resolves_to_the_manifest_routes(planner, target):
    layer = dict(input_dims=(1, 10, 10, 8), filter_dims=(1, 3, 3, 8), output_dims=(1, 8, 8, 8))
    scratch = planner.buffer_bytes(target, "depthwise_conv_s8_opt", "scratch", input_dims=layer["input_dims"],
                                   filter_dims=layer["filter_dims"])
    for ctx_size in (0, scratch):
        plan = planner.depthwise_conv_s8_opt(target, ctx_size=ctx_size, **layer)
        planar = target.isa == "mve" and ctx_size > 0
        assert plan.route == ("dw_opt_s8:planar" if planar else "dw_opt_s8:channelwise")
        assert plan.implementation.id == ("depthwise_conv_s8_opt_planar" if planar
                                          else "depthwise_conv_s8_opt_channelwise")
        assert plan.implementation.entry == plan.kernel
        assert plan.ctx_bytes == scratch


def test_dw_opt_plan_at_the_plane_boundary(planner):
    """The plane of this layer is 10 x 10 + 32 bytes: ctx->size at the plane runs planar, one byte short fails."""
    layer = dict(input_dims=(1, 10, 10, 8), filter_dims=(1, 3, 3, 8), output_dims=(1, 8, 8, 8))
    assert planner.depthwise_conv_s8_opt(Target("mve"), ctx_size=132, **layer).route == "dw_opt_s8:planar"
    with pytest.raises(PlanRefused) as refused:
        planner.depthwise_conv_s8_opt(Target("mve"), ctx_size=131, **layer)
    assert refused.value.status == "range"
    wide = dict(input_dims=(1, 1, 465, 4), filter_dims=(1, 1, 1, 4), output_dims=(1, 1, 465, 4))
    assert planner.depthwise_conv_s8_opt(Target("mve"), ctx_size=1 << 20, **wide).route == "dw_opt_s8:channelwise"
    narrow = dict(input_dims=(1, 1, 464, 4), filter_dims=(1, 1, 1, 4), output_dims=(1, 1, 464, 4))
    assert planner.depthwise_conv_s8_opt(Target("mve"), ctx_size=1 << 20, **narrow).route == "dw_opt_s8:planar"


@pytest.mark.parametrize("change, status", [
    (dict(ch_mult=2, output_dims=(1, 8, 8, 16), filter_dims=(1, 3, 3, 16)), "shape"),
    (dict(dilation=(2, 1)), "shape"),
    (dict(ctx_size=-1), "range"),
    (dict(input_dims=(1, 65536, 65536, 8)), "range"),
])
def test_dw_opt_plan_refusals(planner, change, status):
    layer = dict(input_dims=(1, 10, 10, 8), filter_dims=(1, 3, 3, 8), output_dims=(1, 8, 8, 8), ctx_size=0)
    layer.update(change)
    with pytest.raises(PlanRefused) as refused:
        planner.depthwise_conv_s8_opt(Target("mve"), **layer)
    assert refused.value.status == status


@pytest.mark.parametrize("kwargs, match", [
    (dict(ctx_size=2**31), "ctx_size"),
    (dict(ctx_size=1.5), "ctx_size"),
    (dict(stride=(1,)), "stride"),
    (dict(ch_mult=True), "ch_mult"),
])
def test_dw_opt_arguments_checked_before_the_call(planner, kwargs, match):
    layer = dict(input_dims=(1, 10, 10, 8), filter_dims=(1, 3, 3, 8), output_dims=(1, 8, 8, 8), ctx_size=0)
    layer.update(kwargs)
    with pytest.raises(ValueError, match=match):
        planner.depthwise_conv_s8_opt(Target("mve"), **layer)


@pytest.mark.parametrize("args, operands, error, match", [
    (("no_such_implementation", "scratch"), {}, ContractError, "no such implementation"),
    (("depthwise_conv_s8_opt", "no_such_buffer"), {}, ContractError, "no buffer"),
    (("depthwise_conv_s8", "ctx"), {}, ContractError, "no size query"),
    (("depthwise_conv_s8_opt", "scratch"), dict(input_dims=(1, 4, 4, 8)), ValueError, "missing \\['filter_dims'\\]"),
    (("depthwise_conv_s8_opt", "scratch"),
     dict(input_dims=(1, 4, 4, 8), filter_dims=(1, 3, 3, 8), output_dims=(1, 2, 2, 8)), ValueError,
     "unexpected \\['output_dims'\\]"),
    (("depthwise_conv_s8_opt", "scratch"), dict(input_dims=(1, 4, 4), filter_dims=(1, 3, 3, 8)), ValueError,
     "input_dims"),
    (("depthwise_conv_s8_to_conv", "scratch"), dict(dw_conv_params={"ch_mult": 9, "groups": 1},
                                                    input_dims=(1, 6, 6, 1), filter_dims=(1, 3, 3, 9),
                                                    output_dims=(1, 4, 4, 9)), ValueError, "dw_conv_params"),
    (("depthwise_conv_s8_to_conv", "scratch"), dict(dw_conv_params={"stride": (1, 1)}, input_dims=(1, 6, 6, 1),
                                                    filter_dims=(1, 3, 3, 9), output_dims=(1, 4, 4, 9)), ValueError,
     "dw_conv_params"),
    (("avg_pool_s8", "accumulators"), dict(output_dims=(1, 4, 4, 13), input_dims=(1, 8, 8)), ValueError,
     "input_dims"),
])
def test_buffer_bytes_refusals(planner, args, operands, error, match):
    with pytest.raises(error, match=match):
        planner.buffer_bytes(Target("mve"), *args, **operands)


@pytest.mark.parametrize("target, args, operands, status", [
    (Target("mve"), ("depthwise_conv_s8_opt_3x3", "scratch"), dict(input_dims=(1, 1, 1, 2**31 - 100)), "range"),
    (Target("mve"), ("depthwise_conv_s8_opt_3x3", "scratch"), dict(input_dims=(1, 1, 65536, 32768)), "range"),
    (Target("mve"), ("depthwise_conv_s8_opt", "weight_sums"), dict(output_dims=(1, 1, 1, 1 << 29)), "range"),
    (Target("mve"), ("fully_connected_per_channel_packed_s8", "packed_weights"),
     dict(filter_dims=((2**31 - 1) // 2, 1, 1, 1)), "range"),
    (Target("dsp"), ("avg_pool_s8", "accumulators"),
     dict(output_dims=(1, 1, 1, 1), input_dims=(1, 1, 1, (1 << 29) + 1)), "range"),
    (Target("mve"), ("depthwise_conv_s8_opt", "scratch"), dict(input_dims=(1, 4, 4, 0), filter_dims=(1, 3, 3, 8)),
     "shape"),
    (Target("mve"), ("depthwise_conv_s8_to_conv", "scratch"), dict(dw_conv_params={"ch_mult": 9},
                                                                   input_dims=(1, 6, 6, 1), filter_dims=(1, 3, 3, 9),
                                                                   output_dims=(1, 4, 4, 8)), "shape"),
])
def test_buffer_bytes_planner_refusals(planner, target, args, operands, status):
    with pytest.raises(PlanRefused) as refused:
        planner.buffer_bytes(target, *args, **operands)
    assert refused.value.status == status


def test_buffer_bytes_needs_the_manifest(cache):
    bare = Planner(build_planner(ROOT, "gcc", cache_dir=cache), "gcc")
    with pytest.raises(PlannerError, match="manifest"):
        bare.buffer_bytes(Target("mve"), "depthwise_conv_s8_opt_3x3", "scratch", input_dims=(1, 4, 4, 16))


# The route ids of the manifest and the arm_nn_dispatch.h constant each one is.
ROUTE_NAMES = {
    "dw_wrapper_s8:to_conv": "ARM_NN_DW_CONV_S8_ROUTE_TO_CONV", "dw_wrapper_s8:conv_3x3": "ARM_NN_DW_CONV_S8_ROUTE_3X3",
    "dw_wrapper_s8:opt": "ARM_NN_DW_CONV_S8_ROUTE_OPT", "dw_wrapper_s8:generic": "ARM_NN_DW_CONV_S8_ROUTE_GENERIC",
    "fc_wrapper_s8:per_channel": "ARM_NN_FC_S8_ROUTE_PER_CHANNEL",
    "fc_wrapper_s8:per_tensor": "ARM_NN_FC_S8_ROUTE_PER_TENSOR",
    "dw_opt_s8:planar": "ARM_NN_DW_OPT_S8_ROUTE_PLANAR", "dw_opt_s8:channelwise": "ARM_NN_DW_OPT_S8_ROUTE_CHANNELWISE",
}


def test_route_enums_read_the_values_the_compiler_gives(tmp_path):
    enums = route_enums(ROOT)
    assert set(enums) >= {"arm_depthwise_conv_wrapper_s8", "arm_fully_connected_wrapper_s8",
                          "arm_depthwise_conv_s8_opt"}
    names = [name for constants in enums.values() for name in constants.values()]
    probe = tmp_path / "probe.c"
    probe.write_text('#include <stdio.h>\n#include "Internal/arm_nn_dispatch.h"\nint main(void) {\n'
                     + "".join(f'printf("{n} %d\\n", (int){n});\n' for n in names) + "return 0; }\n")
    cc = os.environ.get("CC") or shutil.which("cc") or shutil.which("gcc")
    subprocess.run([cc, "-std=c11", f"-I{ROOT / 'Include'}", str(probe), "-o", str(tmp_path / "probe")], check=True,
                   capture_output=True)
    out = subprocess.run([str(tmp_path / "probe")], check=True, capture_output=True, text=True).stdout
    compiled = dict(line.split() for line in out.splitlines())
    assert {n: int(v) for n, v in compiled.items()} == {n: v for c in enums.values() for v, n in c.items()}


@pytest.mark.parametrize("target", TARGETS, ids=str)
def test_routed_plans_carry_the_selector_value(planner, target):
    plans = [planner.depthwise_conv_wrapper_s8(target, **layer) for layer in DW_GRID]
    plans += [planner.fully_connected_wrapper_s8(target, input_dims=(1, 1, 1, 40), filter_dims=(40, 1, 1, 10),
                                                 output_dims=(1, 1, 1, 10), per_channel=per_channel)
              for per_channel in (False, True)]
    plans += [planner.depthwise_conv_s8_opt(target, input_dims=(1, 10, 10, 8), filter_dims=(1, 3, 3, 8),
                                            output_dims=(1, 8, 8, 8), ctx_size=ctx_size) for ctx_size in (0, 1 << 20)]
    enums = route_enums(ROOT)
    for plan in plans:
        assert plan.route_name == ROUTE_NAMES[plan.route]
        assert enums[plan.implementation.wrapper][plan.route_value] == plan.route_name


def test_direct_entries_have_no_route_value(planner):
    for plan in (planner.pool_s8(Target("mve"), max_pool=max_pool, input_dims=(1, 8, 8, 16), kernel_hw=(2, 2),
                                 output_dims=(1, 4, 4, 16), stride=(2, 2)) for max_pool in (False, True)):
        assert (plan.route, plan.route_value, plan.route_name) == (None, None, None)
    plan = planner.fully_connected_per_channel_packed_s8(Target("mve"), filter_dims=(64, 1, 1, 16), filter_offset=0,
                                                         multipliers=[1 << 30] * 16, shifts=[-1] * 16)
    assert (plan.route, plan.route_value, plan.route_name) == (None, None, None)


def test_a_plan_without_a_manifest_keeps_the_value_but_no_name(cache):
    bare = Planner(build_planner(ROOT, "gcc", cache_dir=cache), "gcc")
    plan = bare.depthwise_conv_wrapper_s8(Target("mve"), **DW_GRID[0])
    assert (plan.route, plan.route_value, plan.route_name) == ("dw_wrapper_s8:opt", 2, None)


@pytest.mark.parametrize("edit, match", [
    (lambda text: text.replace("ARM_NN_DW_CONV_S8_ROUTE_GENERIC = 3", "ARM_NN_DW_CONV_S8_ROUTE_GENERIC = 2"), "twice"),
    (lambda text: text.replace("The kernel arm_fully_connected_wrapper_s8() calls.", "Routes."), "no route 0"),
    (lambda text: "", "declares no route enum"),
])
def test_a_route_without_a_readable_constant_fails_closed(tree, tmp_path, edit, match):
    header = tree / "Include" / "Internal" / "arm_nn_dispatch.h"
    header.write_text(edit(header.read_text()))
    shutil.copytree(ROOT / "python" / "ns_cmsis_nn_contracts", tree / "python" / "ns_cmsis_nn_contracts")
    implementations = load_implementations(ROOT, load_signatures(ROOT))
    planner = Planner(build_planner(ROOT, "gcc", cache_dir=tmp_path), "gcc",
                      dataclasses.replace(implementations, path=tree / implementations.path.relative_to(ROOT)))
    with pytest.raises(PlannerError, match=match):
        planner.fully_connected_wrapper_s8(Target("mve"), input_dims=(1, 1, 1, 8), filter_dims=(8, 1, 1, 4),
                                           output_dims=(1, 1, 1, 4), per_channel=True)
    with pytest.raises(PlannerError, match="cannot be read"):
        route_enums(tmp_path / "nowhere")
