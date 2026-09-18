# SPDX-License-Identifier: LicenseRef-Ambiq-Apollo-SDK
"""Kernel-family patterns used by the fast header classification guard."""

import re

GROUP_PATTERNS: dict[str, tuple[str, ...]] = {
    "convolution": (
        r"^arm_convolve",
        r"^arm_depthwise_conv",
        r"^arm_depthwise_convolve",
        r"^arm_depthwise_nhwc_conv",
        r"^arm_transpose_conv",
    ),
    "fully-connected": (
        r"^arm_fully_connected",
        r"^arm_batch_matmul",
    ),
    "elementwise": (
        r"^arm_abs",
        r"^arm_add",
        r"^arm_batch_norm",
        r"^arm_elementwise",
        r"^arm_maximum",
        r"^arm_minimum",
        r"^arm_mul",
        r"^arm_nn_abs",
        r"^arm_nn_sqrt",
        r"^arm_rsqrt",
        r"^arm_select_v2",
        r"^arm_sqrt",
        r"^arm_squared_difference",
        r"^arm_sub",
    ),
    "reduction-comparison": (
        r"^arm_argmax",
        r"^arm_argmin",
        r"^arm_comparison",
        r"^arm_equal",
        r"^arm_greater",
        r"^arm_less",
        r"^arm_mean",
        r"^arm_nn_mean",
        r"^arm_not_equal",
        r"^arm_reduce",
        r"^arm_vector_sum",
        r"^arm_where",
    ),
    "activation": (
        r"^arm_clamp",
        r"^arm_hard_swish",
        r"^arm_leaky_relu",
        r"^arm_logistic",
        r"^arm_nn_activation",
        r"^arm_prelu",
        r"^arm_relu",
        r"^arm_tanh",
    ),
    "data-movement": (
        r"^arm_batch_to_space",
        r"^arm_broadcast_to",
        r"^arm_concatenation",
        r"^arm_depth_to_space",
        r"^arm_dynamic_update_slice",
        r"^arm_gather",
        r"^arm_mirror_pad",
        r"^arm_nn_fill",
        r"^arm_pack",
        r"^arm_pad",
        r"^arm_reshape",
        r"^arm_resize",
        r"^arm_reverse_sequence",
        r"^arm_scatter_nd",
        r"^arm_space_to",
        r"^arm_split",
        r"^arm_strided_slice",
        r"^arm_tile",
        r"^arm_transpose_f",
        r"^arm_transpose_s",
        r"^arm_unpack",
    ),
    "classifier-tail": (
        r"^arm_avg_?pool",
        r"^arm_dequantize",
        r"^arm_max_pool",
        r"^arm_q7_to_q15",
        r"^arm_quantize",
        r"^arm_requantize",
        r"^arm_softmax",
    ),
    "sequence": (
        r"^arm_lstm",
        r"^arm_nn_lstm",
        r"^arm_gru",
        r"^arm_nn_gru",
        r"^arm_svdf",
    ),
}

def _matches(name: str, patterns: tuple[str, ...]) -> bool:
    return any(re.search(pattern, name) for pattern in patterns)


def _dtype(name: str) -> str:
    # `fp16` is a legacy spelling of the same half-precision bucket, still
    # used by arm_elementwise_add_fp16. It has to be probed separately --
    # `(^|_)f16($|_)` cannot match `_fp16`, because the `f16` there is
    # preceded by `p` rather than `_` -- and then folded into `f16`, since
    # docs/_static/api-filter.js compares dtype for exact equality against
    # the chip values in docs/reference/api-groups.md. Without the fold the
    # kernel falls through to "mixed" and no chip on the page reaches it.
    for dtype in ("s4", "s8", "s16", "s32", "u8", "q7", "q15", "f16", "fp16", "f32"):
        if re.search(rf"(^|_){dtype}($|_)", name):
            return "f16" if dtype == "fp16" else dtype
    return "mixed"
