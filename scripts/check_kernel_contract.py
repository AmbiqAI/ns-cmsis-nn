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
    python3 scripts/check_kernel_contract.py verify-xml [--xml-dir DIR]   # cross-check against Doxygen XML

verify-xml is the independent check: the docs build runs doxygen itself and the XML it emits
must describe the same functions, parameters, types and directions as the export, so a
parser blind spot in the scanner and a doxygen blind spot cannot both hide the same thing.

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
import xml.etree.ElementTree as ET

sys.path.insert(0, str(Path(__file__).resolve().parent))
from check_doxygen_params import (  # noqa: E402
    PUBLIC_HEADER_GLOB, check_decl, parse_header, public_headers, resolve_tags)

REPO = Path(__file__).resolve().parents[1]
SCHEMA = 'ns-cmsis-nn/kernel-contracts/1'
DEFAULT_OUTPUT = REPO / 'Tests' / 'KernelContracts' / 'kernel_contracts.json'
# Where astro-site/scripts/build-reference.mjs leaves the Doxygen XML it generates.
DEFAULT_XML_DIR = REPO / 'astro-site' / '.cache' / 'reference' / 'xml'
XML_DIRECTIONS = {'in': 'in', 'out': 'out', 'inout': 'in,out'}
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
    params = []
    for param in decl.params:
        record = {'name': param.name, 'type': param.type, 'direction': directions[param.name]}
        if param.extent:
            record['extent'] = param.extent
        params.append(record)
    return {
        'name': decl.name,
        'header': header,
        'line': decl.line,
        'guards': [guard for guard in decl.guards if not NOISE_GUARD_RE.match(guard)],
        'returns': decl.ret,
        'params': params,
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


RECORD_KEYS = ('name', 'header', 'line', 'guards', 'returns', 'params')


def render(document):
    """The one canonical text form: one line per parameter, fixed key order, so the file
    stays reviewable in diffs and well under the pre-commit large-file limit."""
    lines = ['{', f'  "schema": {json.dumps(document["schema"])},', '  "functions": [']
    records = document['functions']
    for i, record in enumerate(records):
        lines.append('    {')
        for key in RECORD_KEYS[:-1]:
            lines.append(f'      {json.dumps(key)}: {json.dumps(record[key])},')
        params = record['params']
        if not params:
            lines.append('      "params": []')
        else:
            lines.append('      "params": [')
            for j, param in enumerate(params):
                item = json.dumps(param, sort_keys=True, separators=(', ', ': '))
                lines.append(f'        {item}' + (',' if j + 1 < len(params) else ''))
            lines.append('      ]')
        lines.append('    }' + (',' if i + 1 < len(records) else ''))
    lines += ['  ]', '}']
    return '\n'.join(lines) + '\n'


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
    try:
        canonical = render(committed)
    except (KeyError, TypeError) as error:
        raise ExportError(f'{output}: malformed record ({error!r})') from None
    if output.read_text(encoding='utf-8') != canonical:
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


class XmlError(ValueError):
    """The XML cannot be read at all, as opposed to disagreeing with the export."""


def _text(node):
    return ' '.join(''.join(node.itertext()).split()) if node is not None else ''


def _squash(ctype):
    """Types compared without whitespace, and without the name inside a function-pointer
    declarator: doxygen spaces `*` differently and prints `void(*)(int32_t)` for a parameter
    the scanner keeps as `void (*fn)(int32_t)`."""
    return re.sub(r'\(\*\w+\)', '(*)', re.sub(r'\s+', '', ctype))


DEFAULT_DOXYFILE = REPO / 'Documentation' / 'Doxygen' / 'nn.dxy.in'


def doxyfile_defines(path):
    """The PREDEFINED macros of a Doxyfile as {name: value}; doxygen sees only the branches
    they satisfy, so a declaration under any other #if is legitimately absent from its XML."""
    try:
        text = Path(path).read_text(encoding='utf-8')
    except OSError as error:
        raise XmlError(f'{path}: cannot read the Doxyfile ({error})') from None
    match = re.search(r'^PREDEFINED\s*\+?=\s*(.*?)(?<!\\)$', text, re.M | re.S)
    if match is None:
        raise XmlError(f'{path}: no PREDEFINED line')
    defines = {}
    for token in match.group(1).replace('\\\n', ' ').split():
        name, _, value = token.partition('=')
        defines[re.sub(r'\(.*$', '', name)] = value
    return defines


def guard_holds(guard, defines):
    """True/False when the guard is decidable from the Doxyfile defines, else None."""
    negated = guard.startswith('!')
    body = guard[1:] if negated else guard
    if body.startswith('(') and body.endswith(')'):
        body = body[1:-1].strip()
    if re.fullmatch(r'\d+', body):
        value = body != '0'
    elif re.fullmatch(r'defined\((\w+)\)', body):
        value = re.fullmatch(r'defined\((\w+)\)', body).group(1) in defines
    elif re.fullmatch(r'\w+', body):
        value = defines.get(body, '') not in ('', '0')
    else:
        return None
    return (not value) if negated else value


def doxygen_sees(record, defines):
    """Whether doxygen's preprocessor keeps this declaration; raises on a guard it cannot judge."""
    for guard in record.get('guards', []):
        holds = guard_holds(guard, defines)
        if holds is None:
            raise XmlError(f'{record["name"]}: guard {guard!r} cannot be evaluated against the '
                           'Doxyfile PREDEFINED set; teach guard_holds() this shape')
        if not holds:
            return False
    return True


def load_xml_functions(xml_dir):
    """Every non-static function memberdef declared in a public header, keyed by name.

    Members live in whichever compound doxygen chose (a file compound, or the group of an
    @addtogroup block, which is where every float kernel ends up), so every compound
    index.xml lists is walked and members are deduplicated by their id.
    """
    xml_dir = Path(xml_dir)
    index = xml_dir / 'index.xml'
    if not index.is_file():
        raise XmlError(f'{index}: not found; run the docs build (astro-site: npm run reference)')
    try:
        compounds = [c.get('refid') for c in ET.parse(index).getroot().iter('compound')
                     if c.get('kind') in ('file', 'group')]
    except ET.ParseError as error:
        raise XmlError(f'{index}: {error}') from None
    functions, seen_ids, public = {}, set(), re.compile(r'^arm_nn[^/]*functions[^/]*\.h$')
    for refid in compounds:
        path = xml_dir / f'{refid}.xml'
        try:
            root = ET.parse(path).getroot()
        except (OSError, ET.ParseError) as error:
            raise XmlError(f'{path}: {error}') from None
        for member in root.iter('memberdef'):
            if member.get('kind') != 'function' or member.get('id') in seen_ids:
                continue
            seen_ids.add(member.get('id'))
            location = member.find('location')
            declared_in = Path(location.get('declfile') or location.get('file') or '') if location is not None else Path()
            if not public.match(declared_in.name) or member.get('static') == 'yes':
                continue
            name = member.findtext('name')
            directions = {}
            for item in member.iter('parameteritem'):
                for pname in item.iter('parametername'):
                    directions[_text(pname)] = pname.get('direction')
            params = []
            for param in member.findall('param'):
                pname = param.findtext('declname') or param.findtext('defname') or ''
                if not pname and _squash(_text(param.find('type'))) == 'void':
                    continue
                params.append({
                    'name': pname,
                    'type': _text(param.find('type')),
                    'extent': _text(param.find('array')),
                    'direction': directions.get(pname),
                })
            if name in functions:
                raise XmlError(f'{path}: {name} appears twice in the XML with different ids')
            functions[name] = {'header': declared_in.name, 'returns': _text(member.find('type')),
                               'params': params}
    if not functions:
        raise XmlError(f'{xml_dir}: no public function members found')
    return functions


def verify_xml(xml_dir, output, doxyfile):
    """Return (disagreements, compared_count, excluded_count) for the export vs the XML."""
    try:
        committed = json.loads(Path(output).read_text(encoding='utf-8'))
    except (OSError, ValueError) as error:
        raise XmlError(f'{output}: cannot be read ({error})') from None
    if committed.get('schema') != SCHEMA:
        raise XmlError(f'{output}: schema {committed.get("schema")!r}, expected {SCHEMA!r}')
    defines = doxyfile_defines(doxyfile)
    exported = {record['name']: record for record in committed['functions']}
    visible = {name for name, record in exported.items() if doxygen_sees(record, defines)}
    documented = load_xml_functions(xml_dir)
    problems = []
    for name in sorted(set(documented) - set(exported)):
        problems.append(f'{name}: in the Doxygen XML ({documented[name]["header"]}) but not in the export')
    for name in sorted(visible - set(documented)):
        problems.append(f'{name}: in the export ({exported[name]["header"]}) but not in the Doxygen XML')
    for name in sorted((set(exported) - visible) & set(documented)):
        problems.append(f'{name}: in the Doxygen XML although its guards {exported[name]["guards"]} '
                        'are false under the Doxyfile PREDEFINED set')
    for name in sorted(set(exported) & set(documented)):
        ours, theirs = exported[name], documented[name]
        if Path(ours['header']).name != theirs['header']:
            problems.append(f'{name}: header {ours["header"]} vs XML {theirs["header"]}')
        if _squash(ours['returns']) != _squash(theirs['returns']):
            problems.append(f'{name}: returns {ours["returns"]!r} vs XML {theirs["returns"]!r}')
        our_names = [p['name'] for p in ours['params']]
        their_names = [p['name'] for p in theirs['params']]
        if our_names != their_names:
            problems.append(f'{name}: parameters {our_names} vs XML {their_names}')
            continue
        for mine, docs in zip(ours['params'], theirs['params']):
            if _squash(mine['type']) != _squash(docs['type']):
                problems.append(f'{name}.{mine["name"]}: type {mine["type"]!r} vs XML {docs["type"]!r}')
            if _squash(mine.get('extent', '')) != _squash(docs['extent']):
                problems.append(f'{name}.{mine["name"]}: extent {mine.get("extent", "")!r} vs XML {docs["extent"]!r}')
            xml_direction = XML_DIRECTIONS.get(docs['direction'])
            if xml_direction is None:
                problems.append(f'{name}.{mine["name"]}: no direction in the Doxygen XML')
            elif xml_direction != mine['direction']:
                problems.append(f'{name}.{mine["name"]}: direction {mine["direction"]} vs XML {xml_direction}')
    return problems, len(set(exported) & set(documented)), len(exported) - len(visible)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('command', nargs='?', choices=('export', 'check', 'verify-xml'), default='check',
                        help='export rewrites the contract file; check (default) fails if it is stale; '
                             'verify-xml compares it with the Doxygen XML of the docs build')
    parser.add_argument('--include-dir', default=str(REPO / 'Include'),
                        help=f'directory whose {PUBLIC_HEADER_GLOB} headers are exported')
    parser.add_argument('--output', default=str(DEFAULT_OUTPUT), help='contract file path')
    parser.add_argument('--xml-dir', default=str(DEFAULT_XML_DIR),
                        help='Doxygen XML directory for verify-xml')
    parser.add_argument('--doxyfile', default=str(DEFAULT_DOXYFILE),
                        help='Doxyfile whose PREDEFINED set decides which #if branches the XML holds')
    args = parser.parse_args()
    include_dir, output = Path(args.include_dir), Path(args.output)
    try:
        if args.command == 'export':
            document = build_contract(include_dir)
            write_atomically(output, render(document))
            print(f'OK: exported {len(document["functions"])} public function signatures to {output}')
        elif args.command == 'verify-xml':
            problems, compared, excluded = verify_xml(args.xml_dir, output, args.doxyfile)
            for problem in problems:
                print(f'ERROR: {problem}', file=sys.stderr)
            if problems:
                print(f'ERROR: {len(problems)} disagreements between {output} and {args.xml_dir}',
                      file=sys.stderr)
                return 1
            print(f'OK: {output} agrees with the Doxygen XML in {args.xml_dir} '
                  f'({compared} functions compared, {excluded} under #if branches the Doxyfile does not define)')
        else:
            count = check(include_dir, output)
            print(f'OK: {output} is fresh ({count} public function signatures)')
    except XmlError as error:
        print(f'ERROR: {error}', file=sys.stderr)
        return 2
    except (OSError, ValueError) as error:
        print(f'ERROR: {error}', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
