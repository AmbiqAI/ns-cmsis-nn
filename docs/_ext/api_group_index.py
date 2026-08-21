from __future__ import annotations

import html
import re
from dataclasses import dataclass
from pathlib import Path

# docutils/sphinx are only needed for the actual Sphinx directive below.
# GROUP_PATTERNS and _matches (the pure data/logic this module owns) have
# no dependency on either, so scripts/check_api_group_classification.py
# can `importlib` this module with a bare `python3` -- no docs venv, no
# Sphinx install -- and stay a millisecond, no-build CI check instead of
# requiring the full doxygen+sphinx toolchain just to read a dict.
try:
    from docutils import nodes
    from docutils.parsers.rst import Directive
    from sphinx.application import Sphinx
except ModuleNotFoundError:
    nodes = None  # type: ignore[assignment]
    Directive = object  # type: ignore[assignment,misc]
    Sphinx = None  # type: ignore[assignment,misc]


@dataclass(frozen=True)
class ApiFunction:
    name: str
    href: str


# Every public top-level kernel is expected to match at least one pattern
# here; scripts/check_api_group_classification.py enforces that against
# Include/arm_nnfunctions*.h on every PR (see #283) so a drifted or missing
# pattern fails fast instead of silently dropping a kernel off the
# customer-facing "Browse By Kernel Family" page. That check imports
# GROUP_PATTERNS and _matches from this module directly, so this stays the
# single source of truth for both the rendered grouping and the guard.
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


class ApiGroupIndex(Directive):
    required_arguments = 1
    optional_arguments = 0
    final_argument_whitespace = False
    has_content = False

    def run(self) -> list[nodes.Node]:
        env = self.state.document.settings.env
        group_key = self.arguments[0]
        patterns = GROUP_PATTERNS.get(group_key)

        if patterns is None:
            message = self.state_machine.reporter.warning(
                f"Unknown api-group-index group '{group_key}'",
                line=self.lineno,
            )
            return [message]

        api_dir = Path(env.srcdir) / "api"
        functions = _load_functions(api_dir)
        matched = [function for function in functions if _matches(function.name, patterns)]

        if not matched:
            message = self.state_machine.reporter.warning(
                f"No API functions matched group '{group_key}'",
                line=self.lineno,
            )
            return [message]

        env.note_dependency(str(api_dir))
        items = "\n".join(
            '<a class="api-function-link" '
            f'href="{function.href}" '
            f'data-api-function="{html.escape(function.name)}" '
            f'data-api-family="{html.escape(group_key)}" '
            f'data-api-dtype="{html.escape(_dtype(function.name))}" '
            f'data-api-role="{html.escape(_role(function.name))}">'
            f"<code>{html.escape(function.name)}</code></a>"
            for function in matched
        )
        plural = "function" if len(matched) == 1 else "functions"
        markup = (
            '<div class="api-function-summary">'
            f'<span class="api-function-count">{len(matched)} {plural}</span>'
            '</div>'
            '<div class="api-function-list">'
            f"{items}"
            '</div>'
        )
        return [nodes.raw("", markup, format="html")]


def _load_functions(api_dir: Path) -> list[ApiFunction]:
    functions: list[ApiFunction] = []
    for path in sorted(api_dir.glob("function_*.rst")):
        text = path.read_text(encoding="utf-8", errors="replace")
        match = re.search(r"^Function\s+(\S+)\s*$", text, re.MULTILINE)
        if match:
            functions.append(ApiFunction(name=match.group(1), href=f"../api/{path.stem}.html"))
    return sorted(functions, key=lambda function: function.name)


def _matches(name: str, patterns: tuple[str, ...]) -> bool:
    return any(re.search(pattern, name) for pattern in patterns)


def _dtype(name: str) -> str:
    # `fp16` is a legacy spelling of the same half-precision bucket, still
    # used by arm_fully_connected_fp16. It has to be probed separately --
    # `(^|_)f16($|_)` cannot match `_fp16`, because the `f16` there is
    # preceded by `p` rather than `_` -- and then folded into `f16`, since
    # docs/_static/api-filter.js compares dtype for exact equality against
    # the chip values in docs/reference/api-groups.md. Without the fold the
    # kernel falls through to "mixed" and no chip on the page reaches it.
    for dtype in ("s4", "s8", "s16", "s32", "u8", "q7", "q15", "f16", "fp16", "f32"):
        if re.search(rf"(^|_){dtype}($|_)", name):
            return "f16" if dtype == "fp16" else dtype
    return "mixed"


def _role(name: str) -> str:
    if "get_buffer_size" in name:
        return "buffer"
    if "wrapper" in name:
        return "wrapper"
    if re.search(r"(_fast|_opt|_mve|_dsp|_core)", name):
        return "optimized"
    return "kernel"


def setup(app: Sphinx) -> dict[str, bool]:
    app.add_directive("api-group-index", ApiGroupIndex)
    return {"parallel_read_safe": True, "parallel_write_safe": True}
