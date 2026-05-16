# SPDX-License-Identifier: LicenseRef-Ambiq-Apollo-SDK

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "docs" / "_ext"))
DOXYGEN_XML = ROOT / "Documentation" / "xml"

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

html_theme = "furo"
html_title = "heliaCORE"
html_short_title = "heliaCORE"
html_logo = "assets/helia-core-logo-dark.png"
html_favicon = "assets/helia-core-icon-color.svg"
html_static_path = ["_static"]
html_css_files = ["heliacore.css"]
html_js_files = [
    "https://cdn.jsdelivr.net/npm/chart.js",
    "chart-init.js",
    "api-filter.js",
]
html_theme_options = {
    "source_repository": "https://github.com/AmbiqAI/ns-cmsis-nn/",
    "source_branch": "main",
    "source_directory": "docs/",
}

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
    "createTreeView": True,
    "exhaleExecutesDoxygen": False,
    "verboseBuild": False,
}

# Exhale can emit very large C API trees. Keep warnings meaningful without
# requiring every generated target to participate in Sphinx nitpicky linking.
nitpicky = False

# Make generated pages deterministic across local and CI builds.
os.environ.setdefault("PYTHONHASHSEED", "0")
