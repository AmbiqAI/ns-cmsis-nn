from __future__ import annotations

import html
import re
from dataclasses import dataclass
from pathlib import Path

from docutils import nodes
from docutils.parsers.rst import Directive
from sphinx.application import Sphinx
from sphinx.util import logging

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ApiFunction:
    name: str
    href: str
    docname: str


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
        r"^arm_select_v2",
        r"^arm_space_to",
        r"^arm_split",
        r"^arm_strided_slice",
        r"^arm_tile",
        r"^arm_transpose_f",
        r"^arm_transpose_s",
        r"^arm_where",
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

# The headers that make up the *public* top-level kernel API, as opposed to
# Include/arm_nnsupportfunctions*.h and Include/Internal/*.h, which hold
# internal helpers that are never expected to appear on the customer-facing
# "Browse By Kernel Family" page. Every function declared (or @copydoc/@ref
# referenced) in these two files is expected to land in some GROUP_PATTERNS
# bucket; see _check_unmatched_public_functions below.
PUBLIC_API_HEADERS: tuple[str, ...] = (
    "arm_nnfunctions.h",
    "arm_nnfunctions_flt.h",
)

# Matches a declaration/reference of the form `arm_some_name(` in a header.
# Deliberately permissive: it also picks up @copydoc/@ref mentions inside
# doc comments, not just the declarations themselves. That over-matching is
# harmless because every such mention necessarily names a real function that
# is itself declared somewhere in these same headers, so it can only add
# genuine public names, never fabricate one -- and it sidesteps having to
# parse multi-line C prototypes (return type and name often sit on different
# lines in these headers).
_PUBLIC_NAME_RE = re.compile(r"\b(arm_[A-Za-z0-9_]*[A-Za-z0-9])\s*\(")


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
            functions.append(
                ApiFunction(
                    name=match.group(1),
                    href=f"../api/{path.stem}.html",
                    docname=f"api/{path.stem}",
                )
            )
    return sorted(functions, key=lambda function: function.name)


def _matches(name: str, patterns: tuple[str, ...]) -> bool:
    return any(re.search(pattern, name) for pattern in patterns)


def _matched_group(name: str) -> str | None:
    """The first GROUP_PATTERNS key whose patterns match `name`, or None.

    None means `name` matches zero groups -- the condition
    _check_unmatched_public_functions warns about for public kernels.
    """
    for group_key, patterns in GROUP_PATTERNS.items():
        if _matches(name, patterns):
            return group_key
    return None


def _public_api_names(include_dir: Path) -> frozenset[str]:
    """Function names declared (or doc-referenced) in the public top-level
    kernel headers, i.e. the set a generated function is checked against to
    decide whether an empty _matched_group() is a real gap or an internal
    support function that legitimately has no customer-facing group.
    """
    names: set[str] = set()
    for filename in PUBLIC_API_HEADERS:
        path = include_dir / filename
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        names.update(match.group(1) for match in _PUBLIC_NAME_RE.finditer(text))
    return frozenset(names)


def _dtype(name: str) -> str:
    for dtype in ("s4", "s8", "s16", "s32", "u8", "q7", "q15", "fp16", "f32"):
        if re.search(rf"(^|_){dtype}($|_)", name):
            return dtype
    return "mixed"


def _role(name: str) -> str:
    if "get_buffer_size" in name:
        return "buffer"
    if "wrapper" in name:
        return "wrapper"
    if re.search(r"(_fast|_opt|_mve|_dsp|_core)", name):
        return "optimized"
    return "kernel"


def _check_unmatched_public_functions(app: Sphinx, env) -> None:
    """Warn about public kernels that GROUP_PATTERNS silently drops.

    ApiGroupIndex.run() only ever reports a *group* coming up empty. A
    function that simply drifts out of every pattern -- a rename, or a new
    operator family that never got a pattern -- matches zero groups and
    disappears from the "Browse By Kernel Family" page with no warning at
    all (see #283, where 26 public kernels including every float pooling,
    batch-norm, transpose and depthwise-conv variant went missing this way).
    Cross-referencing against the public kernel headers is what lets this
    warn only for genuine gaps instead of the ~150 internal support
    functions that legitimately have no customer-facing group -- a warning
    that fired for all of them would be noise nobody would act on.

    Hooked to env-check-consistency (fires once per build, after the full
    read phase) rather than builder-inited so it is guaranteed to run after
    exhale's builder-inited hook has populated docs/api/, regardless of
    extension registration order.
    """
    api_dir = Path(env.srcdir) / "api"
    if not api_dir.is_dir():
        return
    functions = _load_functions(api_dir)
    if not functions:
        return

    include_dir = Path(env.srcdir).parent / "Include"
    public_names = _public_api_names(include_dir)

    for function in functions:
        if function.name not in public_names:
            continue
        if _matched_group(function.name) is not None:
            continue
        logger.warning(
            f"public API function '{function.name}' (declared in Include/arm_nnfunctions.h "
            "or arm_nnfunctions_flt.h) does not match any GROUP_PATTERNS group in "
            "docs/_ext/api_group_index.py -- it is missing from the 'Browse By Kernel "
            "Family' page. Add a pattern for it.",
            location=function.docname,
            type="api_group_index",
            subtype="unmatched_public_function",
        )


def setup(app: Sphinx) -> dict[str, bool]:
    app.add_directive("api-group-index", ApiGroupIndex)
    app.connect("env-check-consistency", _check_unmatched_public_functions)
    return {"parallel_read_safe": True, "parallel_write_safe": True}
