# SPDX-License-Identifier: LicenseRef-Ambiq-Apollo-SDK
"""Vocabulary of the implementation manifest, shared by its producer (scripts/check_kernel_implementations.py) and
the loaders here, so the two read one definition."""

import re

SCHEMA = "ns-cmsis-nn/kernel-implementations/1"
BASE_FEATURES = ("mvei", "dsp", "mvef", "mve_f16", "f32_api", "f16_api", "autovectorize", "single_rounding")
FEATURES = frozenset(BASE_FEATURES[:6]) | {"!" + feature for feature in BASE_FEATURES}
VARIES_WITH = frozenset({"toolchain"})
OPS = frozenset({"fully_connected", "depthwise_conv", "avg_pool", "max_pool"})
QUANTIZATIONS = frozenset({"per_tensor", "per_channel", "none"})
DTYPES = frozenset({"int8", "int16", "int32", "int64", "float16", "float32"})
LAYOUTS = frozenset({"NHWC", "NC", "OI", "1HWO"})
LIFETIMES = frozenset({"call", "prepared", "prepare"})
DECLINE_CAUSES = frozenset({"build", "layer", "buffer"})
CLAIMS = frozenset({"bit_exact", "bounded"})
ID_RE = re.compile(r"^[a-z][a-z0-9_]*$")
ROUTE_RE = re.compile(r"^([a-z0-9_]+):([a-z0-9_]+)$")
FORMAT_RE = re.compile(r"^[a-z][a-z0-9_]*/[1-9][0-9]*$")
REFERENCE_RE = re.compile(r"^hct_ref:[a-z][a-z0-9_]*$")
VERSION_RE = re.compile(r"^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)$")

# Keys of the exported records (the producer adds family, consumes, fills and predicate_static to what is authored).
EXPORT_ENVELOPE_KEYS = frozenset({"schema", "library_version", "signature_schema", "signature_sha256", "planner_abi",
                           "implementations"})
EXPORT_RECORD_KEYS = frozenset({"id", "family", "op", "quantization", "dtypes", "layout", "entry", "operands", "routed_by",
                         "requires", "varies_with", "eligibility", "buffers", "prepare", "declines", "since",
                         "numerics", "note"})
EXPORT_ROUTE_KEYS = frozenset({"wrapper", "route", "requires", "when"})
EXPORT_BUFFER_KEYS = frozenset({"name", "param", "size_query", "units", "align", "lifetime", "optional", "format", "note"})
EXPORT_QUERY_KEYS = frozenset({"symbol", "args"})
EXPORT_STEP_KEYS = frozenset({"symbol", "args", "consumes", "fills", "note"})
EXPORT_ELIGIBILITY_KEYS = frozenset({"when", "predicate", "predicate_static", "args"})
EXPORT_DECLINE_KEYS = frozenset({"cause", "when"})
EXPORT_NUMERICS_KEYS = frozenset({"reference", "claim", "holds_with"})
