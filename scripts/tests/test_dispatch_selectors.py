"""Route selection of the s8 wrappers (Include/Internal/arm_nn_dispatch.h), compiled on the host per build profile.

The selectors route for the build named by ARM_NN_ROUTE_MVEI, ARM_NN_ROUTE_AUTOVECTORIZE and
ARM_NN_TOOLCHAIN_ARMCLANG, so a host compile reproduces each target's routing without its intrinsics. The cases sit
on the boundaries of each gate; a wrapper refuses to build when the route macros disagree with its own build.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
CC = shutil.which("cc") or shutil.which("gcc") or shutil.which("clang")
pytestmark = pytest.mark.skipif(CC is None, reason="host C compiler required")

PROFILES = {
    "scalar": [],
    "mve": ["-DARM_NN_ROUTE_MVEI=1"],
    "mve_autovec": ["-DARM_NN_ROUTE_MVEI=1", "-DARM_NN_ROUTE_AUTOVECTORIZE=1"],
    "mve_armclang": ["-DARM_NN_ROUTE_MVEI=1", "-DARM_NN_TOOLCHAIN_ARMCLANG=1"],
}

DRIVER = r"""
#include <stdio.h>
#include <stdlib.h>
#include "Internal/arm_nn_dispatch.h"

static int32_t a(char **v, int i) { return (int32_t)strtol(v[i], NULL, 10); }

int main(int argc, char **argv)
{
    if (argc < 2) return 2;
    if (argv[1][0] == 'c' && argc == 19)
    {
        const cmsis_nn_dims in = {a(argv, 2), a(argv, 3), a(argv, 4), a(argv, 5)};
        const cmsis_nn_dims f = {a(argv, 6), a(argv, 7), a(argv, 8), a(argv, 9)};
        const cmsis_nn_dims out = {in.n, a(argv, 10), a(argv, 11), a(argv, 12)};
        cmsis_nn_conv_params p = {0};
        p.stride.w = a(argv, 13); p.stride.h = a(argv, 14);
        p.padding.w = a(argv, 15); p.padding.h = a(argv, 16);
        p.dilation.w = a(argv, 17); p.dilation.h = a(argv, 18);
        printf("%d\n", (int)arm_nn_convolve_wrapper_s8_select(&p, &in, &f, &out));
        return 0;
    }
    if (argv[1][0] == 'd' && argc == 18)
    {
        const cmsis_nn_dims in = {a(argv, 2), a(argv, 3), a(argv, 4), a(argv, 5)};
        const cmsis_nn_dims f = {1, a(argv, 6), a(argv, 7), a(argv, 10)};
        const cmsis_nn_dims out = {in.n, a(argv, 8), a(argv, 9), a(argv, 10)};
        cmsis_nn_dw_conv_params p = {0};
        p.ch_mult = a(argv, 11);
        p.stride.w = a(argv, 12); p.stride.h = a(argv, 13);
        p.padding.w = a(argv, 14); p.padding.h = a(argv, 15);
        p.dilation.w = a(argv, 16); p.dilation.h = a(argv, 17);
        printf("%d\n", (int)arm_nn_depthwise_conv_wrapper_s8_select(&p, &in, &f, &out));
        return 0;
    }
    if (argv[1][0] == 'o' && argc == 9)
    {
        /* plane_bytes, ctx (0 NULL, 1 set, 2 NULL buf), ctx->size, weight_sum_ctx (as ctx), filter h w, input c */
        int8_t byte = 0;
        const int32_t ctx_kind = a(argv, 3), ws_kind = a(argv, 5);
        const cmsis_nn_context ctx = {ctx_kind == 1 ? &byte : NULL, a(argv, 4)};
        const cmsis_nn_context ws = {ws_kind == 1 ? &byte : NULL, 0};
        const cmsis_nn_dims in = {1, 8, 8, a(argv, 8)};
        const cmsis_nn_dims f = {1, a(argv, 6), a(argv, 7), a(argv, 8)};
        printf("%d\n",
               (int)arm_nn_depthwise_conv_s8_opt_select(
                   a(argv, 2), ctx_kind == 0 ? NULL : &ctx, ws_kind == 0 ? NULL : &ws, &in, &f));
        return 0;
    }
    if (argv[1][0] == 'w' && argc == 3)
    {
        const cmsis_nn_dims out = {1, 1, 1, a(argv, 2)};
        printf("%d\n", (int)arm_nn_convolve_s8_weights_sum_bytes(&out));
        return 0;
    }
    if (argv[1][0] == 'f' && argc == 3)
    {
        int32_t m = 1, s = 0;
        const cmsis_nn_quant_params q = {&m, &s, (bool)a(argv, 2)};
        printf("%d\n", (int)arm_nn_fully_connected_wrapper_s8_select(&q));
        return 0;
    }
    return 2;
}
"""

# The MVE opt scratch the dw_opt selector compares against, built for the host.
SIZER_SOURCES = [ROOT / "Source" / "ConvolutionFunctions" / "arm_depthwise_conv_get_buffer_sizes_s8.c",
                 ROOT / "Source" / "ConvolutionFunctions" / "arm_convolve_get_buffer_sizes_s8.c"]

# Route enums, in arm_nn_dispatch.h order.
FAST, ONE, ONE_X_N, OUT, SMALL_CIN, C16, CONV = range(7)
INT32_MAX = 2**31 - 1
TO_CONV, DW3X3, OPT, DW = range(4)
PLANAR, CHANNELWISE = range(2)


@pytest.fixture(scope="module")
def route(tmp_path_factory):
    build = tmp_path_factory.mktemp("dispatch")
    (build / "driver.c").write_text(DRIVER)
    binaries = {}
    for name, flags in PROFILES.items():
        exe = build / name
        subprocess.run([CC, "-std=c11", "-Wall", "-Wextra", "-Werror", *flags, "-I", str(ROOT / "Include"),
                        str(build / "driver.c"), *map(str, SIZER_SOURCES), "-o", str(exe)], check=True,
                       capture_output=True, text=True)
        binaries[name] = exe

    def run(profile, kind, *values):
        out = subprocess.run([str(binaries[profile]), kind, *map(str, values)], check=True, capture_output=True,
                             text=True)
        return int(out.stdout)

    return run


# (input n h w c, filter o h w c, output h w c, stride w h, pad w h, dilation w h) -> route per profile
CONV_CASES = {
    "1x1_stride1": ((1, 4, 4, 8, 8, 1, 1, 8, 4, 4, 8, 1, 1, 0, 0, 1, 1), dict.fromkeys(PROFILES, FAST)),
    "1x1_stride2": ((1, 4, 4, 8, 8, 1, 1, 8, 2, 2, 8, 2, 2, 0, 0, 1, 1), dict.fromkeys(PROFILES, ONE)),
    "1x1_padded_is_not_1x1": ((1, 4, 4, 8, 8, 1, 1, 8, 6, 6, 8, 1, 1, 1, 1, 1, 1),
                              {"scalar": CONV, "mve": CONV, "mve_autovec": CONV, "mve_armclang": CONV}),
    "1xn": ((1, 1, 8, 4, 4, 1, 3, 4, 1, 6, 4, 1, 1, 0, 0, 1, 1), dict.fromkeys(PROFILES, ONE_X_N)),
    "1xn_odd_stride_c": ((1, 1, 9, 3, 4, 1, 3, 3, 1, 7, 4, 1, 1, 0, 0, 1, 1),
                         {"scalar": CONV, "mve": SMALL_CIN, "mve_autovec": CONV, "mve_armclang": SMALL_CIN}),
    "single_pixel_out": ((1, 3, 3, 8, 4, 3, 3, 8, 1, 1, 4, 1, 1, 0, 0, 1, 1),
                         {"scalar": CONV, "mve": OUT, "mve_autovec": OUT, "mve_armclang": OUT}),
    "small_cin": ((1, 8, 8, 3, 8, 3, 3, 3, 6, 6, 8, 1, 1, 0, 0, 1, 1),
                  {"scalar": CONV, "mve": SMALL_CIN, "mve_autovec": CONV, "mve_armclang": SMALL_CIN}),
    "small_cin_out_not_multiple_of_4": ((1, 8, 8, 3, 6, 3, 3, 3, 6, 6, 6, 1, 1, 0, 0, 1, 1),
                                        dict.fromkeys(PROFILES, CONV)),
    "small_cin_taps_over_48": ((1, 8, 8, 3, 8, 4, 5, 3, 5, 4, 8, 1, 1, 0, 0, 1, 1), dict.fromkeys(PROFILES, CONV)),
    "c16_3x3": ((1, 6, 6, 16, 8, 3, 3, 16, 4, 4, 8, 1, 1, 0, 0, 1, 1),
                {"scalar": CONV, "mve": C16, "mve_autovec": CONV, "mve_armclang": C16}),
    "c16_3x3_stride2": ((1, 6, 6, 16, 8, 3, 3, 16, 2, 2, 8, 2, 2, 0, 0, 1, 1), dict.fromkeys(PROFILES, CONV)),
    "c17_3x3": ((1, 6, 6, 17, 8, 3, 3, 17, 4, 4, 8, 1, 1, 0, 0, 1, 1), dict.fromkeys(PROFILES, CONV)),
}


@pytest.mark.parametrize("profile", PROFILES)
@pytest.mark.parametrize("case", CONV_CASES)
def test_convolve_routes(route, case, profile):
    args, expected = CONV_CASES[case]
    assert route(profile, "c", *args) == expected[profile]


# (input n h w c, filter h w, output h w c, ch_mult, stride w h, pad w h, dilation w h) -> route per profile
DW_CASES = {
    # The to-conv threshold is 1 under GCC and 8 under Arm Compiler 6.
    "one_in_ch_out_2": ((1, 6, 6, 1, 3, 3, 4, 4, 2, 2, 1, 1, 0, 0, 1, 1),
                        {"scalar": DW, "mve": TO_CONV, "mve_autovec": TO_CONV, "mve_armclang": DW}),
    "one_in_ch_out_8": ((1, 6, 6, 1, 3, 3, 4, 4, 8, 8, 1, 1, 0, 0, 1, 1),
                        {"scalar": DW, "mve": TO_CONV, "mve_autovec": TO_CONV, "mve_armclang": DW}),
    "one_in_ch_out_9": ((1, 6, 6, 1, 3, 3, 4, 4, 9, 9, 1, 1, 0, 0, 1, 1),
                        {"scalar": DW, "mve": TO_CONV, "mve_autovec": TO_CONV, "mve_armclang": TO_CONV}),
    "one_in_ch_out_1": ((1, 6, 6, 1, 3, 3, 4, 4, 1, 1, 1, 1, 0, 0, 1, 1),
                        {"scalar": DW3X3, "mve": OPT, "mve_autovec": OPT, "mve_armclang": OPT}),
    "3x3_pad1": ((1, 6, 6, 8, 3, 3, 6, 6, 8, 1, 1, 1, 1, 1, 1, 1),
                 {"scalar": DW3X3, "mve": OPT, "mve_autovec": OPT, "mve_armclang": OPT}),
    "3x3_pad2": ((1, 6, 6, 8, 3, 3, 8, 8, 8, 1, 1, 1, 2, 2, 1, 1),
                 {"scalar": OPT, "mve": OPT, "mve_autovec": OPT, "mve_armclang": OPT}),
    "5x5": ((1, 8, 8, 8, 5, 5, 4, 4, 8, 1, 1, 1, 0, 0, 1, 1), dict.fromkeys(PROFILES, OPT)),
    "batch_2": ((2, 6, 6, 8, 3, 3, 4, 4, 8, 1, 1, 1, 0, 0, 1, 1), dict.fromkeys(PROFILES, DW)),
    "ch_mult_2": ((1, 6, 6, 4, 3, 3, 4, 4, 8, 2, 1, 1, 0, 0, 1, 1), dict.fromkeys(PROFILES, DW)),
    "dilated_2d": ((1, 9, 9, 8, 3, 3, 5, 5, 8, 1, 1, 1, 0, 0, 2, 2), dict.fromkeys(PROFILES, DW)),
    "dilated_1d": ((1, 1, 12, 8, 1, 3, 1, 8, 8, 1, 1, 1, 0, 0, 2, 1), dict.fromkeys(PROFILES, OPT)),
}


@pytest.mark.parametrize("profile", PROFILES)
@pytest.mark.parametrize("case", DW_CASES)
def test_depthwise_routes(route, case, profile):
    args, expected = DW_CASES[case]
    assert route(profile, "d", *args) == expected[profile]


# The MVE opt scratch of a 3x3 filter is 4 * CH_IN_BLOCK_MVE (124) * 3 * 3 bytes on every MVE build.
SCRATCH_3X3 = 4 * 124 * 3 * 3


# (plane_bytes, ctx, ctx->size, weight_sum_ctx) with ctx kinds 0 NULL, 1 set, 2 NULL buf -> planar on MVE builds
DW_OPT_CASES = {
    "fits": ((100, 1, 100, 1), True),
    "ctx_one_byte_short": ((100, 1, 99, 1), False),
    "ctx_size_zero": ((100, 1, 0, 1), False),
    "plane_is_the_scratch": ((SCRATCH_3X3, 1, 1 << 20, 1), True),
    "plane_one_past_the_scratch": ((SCRATCH_3X3 + 1, 1, 1 << 20, 1), False),
    "declined_plane": ((-1, 1, 1 << 20, 1), False),
    "ctx_null": ((100, 0, 100, 1), False),
    "ctx_buf_null": ((100, 2, 100, 1), False),
    "weight_sums_null": ((100, 1, 100, 0), False),
    "weight_sums_buf_null": ((100, 1, 100, 2), False),
}


@pytest.mark.parametrize("profile", PROFILES)
@pytest.mark.parametrize("case", DW_OPT_CASES)
def test_depthwise_opt_routes(route, case, profile):
    (plane, ctx, size, ws), planar_on_mve = DW_OPT_CASES[case]
    expected = PLANAR if planar_on_mve and profile != "scalar" else CHANNELWISE
    assert route(profile, "o", plane, ctx, size, ws, 3, 3, 8) == expected


@pytest.mark.parametrize("profile", PROFILES)
@pytest.mark.parametrize("channels, mve_bytes", [(1, 4), (13, 52), ((1 << 29) - 1, INT32_MAX - 3), (1 << 29, -1),
                                                 (-1, -1)])
def test_weights_sum_bytes(route, profile, channels, mve_bytes):
    assert route(profile, "w", channels) == (0 if profile == "scalar" else mve_bytes)


@pytest.mark.parametrize("profile", PROFILES)
def test_fully_connected_routes(route, profile):
    assert route(profile, "f", 1) == 0
    assert route(profile, "f", 0) == 1


@pytest.mark.parametrize("source, what", [
    ("Source/FullyConnectedFunctions/arm_fully_connected_wrapper_s8.c", "wrapper"),
    ("Source/NNSupportFunctions/arm_nn_depthwise_conv_s8_planar.c", "kernel"),
])
@pytest.mark.parametrize("flags", [["-DARM_NN_ROUTE_MVEI=1"], ["-DARM_NN_ROUTE_AUTOVECTORIZE=1"]])
def test_a_routed_file_refuses_routes_for_another_build(tmp_path, flags, source, what):
    path = ROOT / source
    ok = subprocess.run([CC, "-std=c11", "-fsyntax-only", "-I", str(ROOT / "Include"), str(path)],
                        capture_output=True, text=True)
    assert ok.returncode == 0, ok.stderr
    bad = subprocess.run([CC, "-std=c11", "-fsyntax-only", *flags, "-I", str(ROOT / "Include"), str(path)],
                         capture_output=True, text=True)
    assert bad.returncode != 0
    assert f"ARM_NN_ROUTE_* must describe the build that compiles this {what}" in bad.stderr
