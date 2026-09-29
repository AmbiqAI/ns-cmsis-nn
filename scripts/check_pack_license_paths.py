#!/usr/bin/env python3
# SPDX-License-Identifier: LicenseRef-Ambiq-Apollo-SDK
"""Check that PDSC license declarations resolve inside a finished CMSIS-Pack."""

import argparse
from pathlib import PurePosixPath
import sys
import xml.etree.ElementTree as ET
from zipfile import BadZipFile, ZipFile


def check_pack(pack):
    """Return the distinct license paths, or raise ValueError on invalid input."""
    with ZipFile(pack) as archive:
        pdscs = [entry for entry in archive.infolist()
                 if not entry.is_dir() and '/' not in entry.filename
                 and entry.filename.endswith('.pdsc')]
        if len(pdscs) != 1:
            raise ValueError('expected exactly one root PDSC file')
        root = ET.fromstring(archive.read(pdscs[0]))
        if root.tag != 'package':
            raise ValueError('PDSC root must be package')
        declarations = [node.text for node in root.findall('license')]
        declarations += [node.get('name') for node in
                         root.findall('licenseSets/licenseSet/license')]
        if not declarations:
            raise ValueError('PDSC has no license declarations')
        paths = set()
        for declaration in declarations:
            name = (declaration or '').strip()
            path = PurePosixPath(name)
            if (not name or path.is_absolute() or '..' in path.parts
                    or '\\' in name or ':' in name or name.endswith('/')
                    or str(path) == '.'):
                raise ValueError(f'unusable license path: {name!r}')
            name = str(path)
            matches = [entry for entry in archive.infolist()
                       if entry.filename == name and not entry.is_dir()]
            if len(matches) != 1:
                raise ValueError(f'license path must resolve to one archive file: {name}')
            paths.add(name)
        return sorted(paths)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('pack', help='finished .pack ZIP archive')
    args = parser.parse_args()
    try:
        paths = check_pack(args.pack)
    except (OSError, BadZipFile, ET.ParseError, ValueError) as error:
        print(f'ERROR: {args.pack}: {error}', file=sys.stderr)
        return 1
    print(f'OK: all {len(paths)} declared license files exist: {", ".join(paths)}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
