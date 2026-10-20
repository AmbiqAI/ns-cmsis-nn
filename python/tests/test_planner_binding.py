# SPDX-License-Identifier: LicenseRef-Ambiq-Apollo-SDK
"""build_planner and Planner: the cache, the refusals of a library or manifest that does not match, every plan resolving
to a manifest implementation, and the preparation helpers against the definitions they implement."""

import dataclasses
import os
import random
import shutil

import pytest

from conftest import ROOT
from ns_cmsis_nn_contracts import (PlannerError, PlanRefused, Planner, Target, build_planner, load_implementations,
                                   load_signatures, open_planner)

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
