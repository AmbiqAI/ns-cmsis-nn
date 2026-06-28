# SPDX-License-Identifier: LicenseRef-Ambiq-Apollo-SDK

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "docs" / "_ext"))
DOXYGEN_XML = ROOT / "Documentation" / "xml"

# Fast iteration mode for local previews. When DOCS_FAST is truthy, skip the
# heavy Doxygen/Breathe/Exhale C API generation (hundreds of generated pages)
# so landing-page and CSS edits rebuild in a fraction of the time. The full API
# reference still builds in CI and in the default (unset) configuration.
DOCS_FAST = os.environ.get("DOCS_FAST", "").strip().lower() in {"1", "true", "yes", "on"}

project = "heliaCORE"
author = "Ambiq Micro, Inc."
copyright = "Ambiq Micro, Inc. Built on Arm CMSIS-NN."

extensions = [
    "api_group_index",
    "breathe",
    "exhale",
    "myst_parser",
    "sphinx_copybutton",
    "sphinx_design",
]

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}
master_doc = "index"
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "scripts", "assets/README.md"]

if DOCS_FAST:
    # Drop the extensions that parse Doxygen XML and emit the generated C API
    # tree, and exclude the API pages that depend on them. This avoids reading
    # and writing the ~400 generated API documents on every rebuild.
    extensions = [ext for ext in extensions if ext not in {"api_group_index", "breathe", "exhale"}]
    exclude_patterns += ["api", "reference/api-groups.md", "reference/doxygen.md"]
    # The hidden landing-page toctree still references the excluded API pages;
    # keep those expected warnings quiet so fast rebuilds stay readable.
    suppress_warnings = ["toc.excluded", "toc.not_readable", "ref.ref", "myst.xref_missing"]

html_theme = "furo"
html_title = "heliaCORE"
html_short_title = "heliaCORE"
html_logo = "assets/helia-core-logo-dark.png"
html_favicon = "assets/helia-core-icon-color.svg"
html_static_path = ["_static", "assets"]
html_css_files = ["heliacore.css"]
html_show_sourcelink = False
html_js_files = [
    "https://cdn.jsdelivr.net/npm/chart.js",
    "chart-init.js",
    "benchmark-charts.js",
    "api-filter.js",
]
html_theme_options = {}

myst_enable_extensions = [
    "attrs_block",
    "attrs_inline",
    "colon_fence",
    "deflist",
]
myst_heading_anchors = 3
myst_links_external_new_tab = True

primary_domain = "c"
highlight_language = "c"

breathe_projects = {
    "heliaCORE": str(DOXYGEN_XML),
}
breathe_default_project = "heliaCORE"
breathe_domain_by_extension = {
    "h": "c",
}

exhale_args = {
    "containmentFolder": "./api",
    "rootFileName": "library_root.rst",
    "rootFileTitle": "C API Reference",
    "doxygenStripFromPath": str(ROOT),
    # Exhale's raw tree view emits hash links that do not always match the
    # final Sphinx/Breathe HTML anchors. Keep the generated API index simple;
    # the grouped/searchable API page is the primary browsing surface.
    "createTreeView": False,
    "exhaleExecutesDoxygen": False,
    "verboseBuild": False,
}

# Exhale can emit very large C API trees. Keep warnings meaningful without
# requiring every generated target to participate in Sphinx nitpicky linking.
nitpicky = False

# Make generated pages deterministic across local and CI builds.
os.environ.setdefault("PYTHONHASHSEED", "0")
