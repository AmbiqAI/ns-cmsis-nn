#!/usr/bin/env python3
# SPDX-License-Identifier: LicenseRef-Ambiq-Apollo-SDK
"""Export the public kernel signatures as a machine-readable contract, and check it is fresh.

Every declaration in the public functions headers becomes one JSON record: name, header,
line, the preprocessor conditions it sits under, its return type, and its parameters with the
normalized C type, array extent and the [in]/[out]/[in,out] direction resolved from the
Doxygen block (through @copydoc chains). The doc block is the source; the parser is the one
scripts/check_doxygen_params.py already gates, so nothing here needs doxygen or a build.

    python3 scripts/check_kernel_contract.py export    # rewrite Tests/KernelContracts/kernel_contracts.json
    python3 scripts/check_kernel_contract.py           # fail if the committed file is stale

Fail-loud by construction: an undocumented parameter, a declaration the parser cannot
classify, a duplicate name across headers or a header set that shrinks below the floor aborts
the export with no file written, because a contract with a silently missing kernel would be
consumed as if that kernel did not exist. Refs AmbiqAI/ns-cmsis-nn#525, #386, #526.
"""

import argparse
import json
import os
from pathlib import Path
import re
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parent))
from check_doxygen_params import (  # noqa: E402
    PUBLIC_HEADER_GLOB, check_decl, parse_header, public_headers, resolve_tags)

REPO = Path(__file__).resolve().parents[1]
SCHEMA = 'ns-cmsis-nn/kernel-contracts/1'
DEFAULT_OUTPUT = REPO / 'Tests' / 'KernelContracts' / 'kernel_contracts.json'
# Include guards and the extern "C" wrapper say nothing about when a kernel exists.
NOISE_GUARD_RE = re.compile(r'^!?defined\((\w+_H|__cplusplus)\)$')


class ExportError(ValueError):
    pass


def function_record(decl, by_name, include_dir):
    if decl.error:
        raise ExportError(f'{decl.path.name}:{decl.line}: {decl.name}: {decl.error}')
    try:
        check_decl(decl, by_name)
        directions = dict(resolve_tags(decl, by_name, {decl.name}))
    except ValueError as error:
        raise ExportError(f'{decl.path.name}:{decl.line}: {decl.name}: {error}') from None
    try:
        header = decl.path.resolve().relative_to(REPO).as_posix()
    except ValueError:
        header = decl.path.resolve().relative_to(Path(include_dir).resolve().parent).as_posix()
    return {
        'name': decl.name,
        'header': header,
        'line': decl.line,
        'guards': [guard for guard in decl.guards if not NOISE_GUARD_RE.match(guard)],
        'returns': decl.ret,
        'static': False,
        'params': [{
            'name': param.name,
            'type': param.type,
            'extent': param.extent,
            'direction': directions[param.name],
        } for param in decl.params],
    }


def build_contract(include_dir):
    """Return the contract document for every public functions header under include_dir."""
    decls = []
    for path in public_headers(include_dir):
        found = parse_header(path)
        if not found:
            raise ExportError(f'{path}: no function declarations found')
        decls.extend(found)
    # One record per symbol: a name declared twice would let a @copydoc resolve against
    # whichever twin the scanner saw last, so refuse before any record is built.
    seen = {}
    for decl in decls:
        if decl.name in seen:
            raise ExportError(f'{decl.name} is declared twice: {seen[decl.name]} and '
                              f'{decl.path.name}:{decl.line}; the contract needs one record per symbol')
        seen[decl.name] = f'{decl.path.name}:{decl.line}'
    by_name = {decl.name: decl for decl in decls}
    functions = [function_record(decl, by_name, include_dir) for decl in decls if not decl.is_static]
    if not functions:
        raise ExportError('no exportable (non-static) declarations found')
    functions.sort(key=lambda record: (record['header'], record['name']))
    return {'schema': SCHEMA, 'functions': functions}


def render(document):
    return json.dumps(document, indent=2, sort_keys=True) + '\n'


def write_atomically(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile('w', dir=path.parent, prefix=path.name + '.', delete=False)
    try:
        with handle:
            handle.write(text)
        os.replace(handle.name, path)
    except BaseException:
        Path(handle.name).unlink(missing_ok=True)
        raise


def comparable(document):
    """The document with the fields that legitimately drift (source lines) removed."""
    return {
        'schema': document.get('schema'),
        'functions': [{key: value for key, value in record.items() if key != 'line'}
                      for record in document.get('functions', [])],
    }


def check(include_dir, output):
    fresh = build_contract(include_dir)
    if not output.is_file():
        raise ExportError(f'{output}: missing; run `{Path(sys.argv[0]).name} export`')
    try:
        committed = json.loads(output.read_text(encoding='utf-8'))
    except (OSError, ValueError) as error:
        raise ExportError(f'{output}: cannot be read as JSON ({error})') from None
    if committed.get('schema') != SCHEMA:
        raise ExportError(f'{output}: schema {committed.get("schema")!r}, expected {SCHEMA!r}')
    if output.read_text(encoding='utf-8') != render(committed):
        raise ExportError(f'{output}: not in canonical form; run `{Path(sys.argv[0]).name} export`')
    if comparable(committed) != comparable(fresh):
        fresh_names = {record['name']: record for record in fresh['functions']}
        old_names = {record['name']: record for record in committed['functions']}
        details = []
        for name in sorted(set(fresh_names) - set(old_names)):
            details.append(f'  + {name}')
        for name in sorted(set(old_names) - set(fresh_names)):
            details.append(f'  - {name}')
        for name in sorted(set(fresh_names) & set(old_names)):
            before = {k: v for k, v in old_names[name].items() if k != 'line'}
            after = {k: v for k, v in fresh_names[name].items() if k != 'line'}
            if before != after:
                details.append(f'  ~ {name}')
        raise ExportError(f'{output}: stale; run `{Path(sys.argv[0]).name} export`\n'
                          + '\n'.join(details[:20]))
    return len(fresh['functions'])


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('command', nargs='?', choices=('export', 'check'), default='check',
                        help='export rewrites the contract file; check (default) fails if it is stale')
    parser.add_argument('--include-dir', default=str(REPO / 'Include'),
                        help=f'directory whose {PUBLIC_HEADER_GLOB} headers are exported')
    parser.add_argument('--output', default=str(DEFAULT_OUTPUT), help='contract file path')
    args = parser.parse_args()
    include_dir, output = Path(args.include_dir), Path(args.output)
    try:
        if args.command == 'export':
            document = build_contract(include_dir)
            write_atomically(output, render(document))
            print(f'OK: exported {len(document["functions"])} public function signatures to {output}')
        else:
            count = check(include_dir, output)
            print(f'OK: {output} is fresh ({count} public function signatures)')
    except (OSError, ValueError) as error:
        print(f'ERROR: {error}', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
