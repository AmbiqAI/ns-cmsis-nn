#!/usr/bin/env python3
#
# SPDX-FileCopyrightText: Copyright 2026 Ambiq <opensource@ambiq.com>
# SPDX-License-Identifier: Apache-2.0
#
# Tests for scripts/check_pdsc.py's check_header_closure() (#514): every
# heliaCORE component must declare every in-repo header its declared files
# reach through #include "...". Imports the script as a module and runs the
# check against a small synthetic tree, as test_check_pdsc.py does.
#
# Run with: python3 scripts/tests/test_check_pdsc_header_closure.py
from __future__ import annotations

import importlib.util
import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "check_pdsc.py"

COMPONENT = 'Cclass="Machine Learning" Cgroup="NN Lib" Csub="heliaCORE" Cvendor="Ambiq"'

TREE = {
    "Include/a.h": '#include "Internal/b.h"\n#include <stdint.h>\n#include "intrin.h"\n',
    "Include/Internal/b.h": '#include "c.h"\n',
    "Include/Internal/c.h": '#include "b.h"\n',
    "Source/Group/x.c": '#include "a.h"\n#include "local.h"\n',
    "Source/Group/local.h": "",
}


def _load_module():
    spec = importlib.util.spec_from_file_location("check_pdsc_closure_under_test", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _pdsc(*components: str) -> ET.Element:
    return ET.fromstring("<package><components>" + "".join(components) + "</components></package>")


def _component(variant: str, headers: list[str], sources: list[str], attrs: str = COMPONENT) -> str:
    files = "".join(f'<file category="header" name="{h}"/>' for h in headers)
    files += "".join(f'<file category="source" name="{s}"/>' for s in sources)
    return f'<component {attrs} Cvariant="{variant}"><files>{files}</files></component>'


ALL_HEADERS = ["Include/a.h", "Include/Internal/b.h", "Include/Internal/c.h", "Source/Group/local.h"]


class CheckHeaderClosure(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name).resolve()
        for rel, text in TREE.items():
            path = self.root / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text)
        self.mod = _load_module()
        self.mod.REPO = self.root

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _run(self, pkg: ET.Element) -> list[str]:
        self.mod.check_header_closure(pkg)
        return list(self.mod.failures)

    def test_complete_component_passes(self) -> None:
        self.assertEqual([], self._run(_pdsc(_component("Source", ALL_HEADERS, ["Source/Group/x.c"]))))

    def test_missing_transitive_internal_header_fails(self) -> None:
        headers = [h for h in ALL_HEADERS if h != "Include/Internal/c.h"]
        failures = self._run(_pdsc(_component("Source", headers, ["Source/Group/x.c"])))
        self.assertEqual(1, len(failures))
        self.assertIn("Include/Internal/c.h", failures[0])
        self.assertIn("Include/Internal/b.h", failures[0])

    def test_missing_source_local_header_fails(self) -> None:
        headers = [h for h in ALL_HEADERS if h != "Source/Group/local.h"]
        failures = self._run(_pdsc(_component("Source", headers, ["Source/Group/x.c"])))
        self.assertEqual(1, len(failures))
        self.assertIn("Source/Group/local.h", failures[0])

    def test_header_only_component_follows_its_headers(self) -> None:
        failures = self._run(_pdsc(_component("Prebuilt", ["Include/a.h"], [])))
        self.assertEqual(2, len(failures))
        self.assertTrue(all("Cvariant='Prebuilt'" in f for f in failures))
        self.assertIn("Include/Internal/b.h", failures[0])
        self.assertIn("Include/Internal/c.h", failures[1])

    def test_only_the_incomplete_component_is_reported(self) -> None:
        pkg = _pdsc(
            _component("Source", ALL_HEADERS, ["Source/Group/x.c"]),
            _component("Prebuilt", ["Include/a.h", "Include/Internal/b.h"], []),
        )
        failures = self._run(pkg)
        self.assertEqual(1, len(failures))
        self.assertIn("Cvariant='Prebuilt'", failures[0])
        self.assertIn("Include/Internal/c.h", failures[0])

    def test_includer_directory_takes_precedence_over_include_dir(self) -> None:
        # Source/Group/shadow.h and Include/shadow.h share a name; x2.c's include resolves beside it, as the compiler's.
        (self.root / "Source/Group/shadow.h").write_text("")
        (self.root / "Include/shadow.h").write_text("")
        (self.root / "Source/Group/x2.c").write_text('#include "shadow.h"\n')
        failures = self._run(_pdsc(_component("Source", ["Include/shadow.h"], ["Source/Group/x2.c"])))
        self.assertEqual(1, len(failures))
        self.assertIn("Source/Group/shadow.h", failures[0])

    def test_indented_include_under_a_condition_is_followed(self) -> None:
        (self.root / "Include/cond.h").write_text('#if defined(X)\n  #include "Internal/c.h"\n#endif\n')
        failures = self._run(_pdsc(_component("Prebuilt", ["Include/cond.h"], [])))
        self.assertEqual(2, len(failures))
        self.assertIn("Include/Internal/c.h", failures[1])

    def test_angle_include_of_an_in_repo_file_is_not_followed(self) -> None:
        (self.root / "Include/angle.h").write_text("#include <Internal/c.h>\n")
        self.assertEqual([], self._run(_pdsc(_component("Prebuilt", ["Include/angle.h"], []))))

    def test_include_that_leaves_the_repo_is_not_a_declared_header(self) -> None:
        outside = self.root.parent / (self.root.name + "-outside.h")
        outside.write_text("")
        self.addCleanup(outside.unlink)
        (self.root / "Include/escape.h").write_text(f'#include "../../{outside.name}"\n')
        failures = self._run(_pdsc(_component("Prebuilt", ["Include/escape.h"], [])))
        self.assertEqual(1, len(failures))
        self.assertIn("resolves neither", failures[0])

    def test_unresolved_quoted_include_fails(self) -> None:
        (self.root / "Source/Group/y.c").write_text('#include "arm_nn_missing.h"\n')
        failures = self._run(_pdsc(_component("Source", [], ["Source/Group/y.c"])))
        self.assertEqual(1, len(failures))
        self.assertIn('#include "arm_nn_missing.h"', failures[0])

    def test_main_runs_the_check(self) -> None:
        self.assertIn("check_header_closure", self.mod.main.__code__.co_names)

    def test_other_components_are_ignored(self) -> None:
        other = 'Cclass="Machine Learning" Cgroup="NN Lib" Csub="Other" Cvendor="Ambiq"'
        self.assertEqual([], self._run(_pdsc(_component("Source", [], ["Source/Group/x.c"], attrs=other))))


if __name__ == "__main__":
    unittest.main()
