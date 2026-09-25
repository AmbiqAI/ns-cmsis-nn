#!/usr/bin/env python3
# SPDX-License-Identifier: LicenseRef-Ambiq-Apollo-SDK
"""Finished-pack license checks, including the repository's staging selection."""

from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import xml.etree.ElementTree as ET
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[2]
CHECKER = ROOT / 'scripts/check_pack_license_paths.py'
PACKAGE = '<license>LICENSE</license>'
SETS = ('<licenseSets><licenseSet id="test">'
        '<license name="LICENSES/Apache-2.0.txt"/>'
        '</licenseSet></licenseSets>')


class PackLicenseTests(unittest.TestCase):
    def run_check(self, declaration=PACKAGE + SETS, files=None, pdsc=None):
        if files is None:
            files = {'LICENSE': 'root license', 'LICENSES/Apache-2.0.txt': 'Apache'}
        with tempfile.TemporaryDirectory() as directory:
            pack = Path(directory) / 'test.pack'
            with ZipFile(pack, 'w') as archive:
                if pdsc is None:
                    pdsc = {'Test.pdsc': f'<package>{declaration}</package>'}
                for name, content in {**pdsc, **files}.items():
                    archive.writestr(name, content)
            return subprocess.run([sys.executable, str(CHECKER), str(pack)],
                                  capture_output=True, text=True)

    def test_both_declaration_forms(self):
        for declaration in (PACKAGE, SETS, PACKAGE + SETS):
            with self.subTest(declaration=declaration):
                self.assertEqual(self.run_check(declaration).returncode, 0)

    def test_each_missing_license(self):
        for missing, files in (
                ('LICENSE', {'LICENSES/Apache-2.0.txt': 'Apache'}),
                ('LICENSES/Apache-2.0.txt', {'LICENSE': 'root'})):
            with self.subTest(missing=missing):
                result = self.run_check(files=files)
                self.assertEqual(result.returncode, 1)
                self.assertIn(missing, result.stderr)

    def test_invalid_declarations(self):
        for declaration in ('', '<license/>', '<licenseSets><licenseSet>'
                            '<license/></licenseSet></licenseSets>',
                            '<license>../LICENSE</license>',
                            '<license>/LICENSE</license>',
                            '<license>LICENSE/</license>',
                            '<license>C:\\LICENSE</license>'):
            with self.subTest(declaration=declaration):
                self.assertEqual(self.run_check(declaration).returncode, 1)

    def test_directory_is_not_license_file(self):
        result = self.run_check(PACKAGE, files={'LICENSE/': ''})
        self.assertEqual(result.returncode, 1)
        self.assertIn('LICENSE', result.stderr)

    def test_invalid_pdsc(self):
        for pdsc in ({}, {'nested/Test.pdsc': '<package/>'},
                     {'Test.pdsc': '<package'}, {'Test.pdsc': '<other/>'},
                     {'One.pdsc': '<package/>', 'Two.pdsc': '<package/>'}):
            with self.subTest(pdsc=pdsc):
                self.assertEqual(self.run_check(pdsc=pdsc).returncode, 1)

    def test_invalid_archive(self):
        with tempfile.TemporaryDirectory() as directory:
            pack = Path(directory) / 'bad.pack'
            pack.write_text('not a ZIP')
            result = subprocess.run([sys.executable, str(CHECKER), str(pack)],
                                    capture_output=True, text=True)
            self.assertEqual(result.returncode, 1)
            self.assertIn('ERROR:', result.stderr)

    def test_real_pack_configuration_selects_declared_licenses(self):
        # Evaluate only configuration, before gen-pack dependency acquisition.
        config = (ROOT / 'gen_pack.sh').read_text().split(
            '############ DO NOT EDIT BELOW ###########', 1)[0]
        result = subprocess.run(
            ['bash', '-c', config + '\nprintf "%s\\n" "$PACK_DIRS" "$PACK_BASE_FILES"'],
            cwd=ROOT, capture_output=True, text=True, check=True)
        selected = result.stdout.split()
        pdsc = next(ROOT.glob('*.pdsc'))
        root = ET.parse(pdsc).getroot()
        paths = [node.text for node in root.findall('license')]
        paths += [node.get('name') for node in
                  root.findall('licenseSets/licenseSet/license')]
        for name in paths:
            with self.subTest(path=name):
                self.assertTrue((ROOT / name).is_file())
                self.assertTrue(any(name == entry or name.startswith(entry + '/')
                                    for entry in selected),
                                f'{name} omitted by gen_pack.sh staging configuration')


if __name__ == '__main__':
    unittest.main()
