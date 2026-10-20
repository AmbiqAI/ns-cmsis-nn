#!/usr/bin/env python3
# SPDX-License-Identifier: LicenseRef-Ambiq-Apollo-SDK
"""Export the kernel implementation manifest, and check it is fresh.

The signature export (Tests/KernelContracts/kernel_contracts.json) says what each public function looks like. This
manifest says how they fit together: which entry implements an operator, which wrapper route reaches it, which build
features it needs, which sizer sizes each of its buffers, how its constant operands are prepared, and when it declines
with ARM_CMSIS_NN_NO_IMPL_ERROR. It is authored as YAML, one file per family, and exported as canonical JSON:

    python3 scripts/check_kernel_implementations.py export   # rewrite Tests/KernelContracts/kernel_implementations.json
    python3 scripts/check_kernel_implementations.py          # fail if the committed file is stale or not canonical

Tests/KernelContracts/implementations/<family>.yaml holds `family: <file stem>` and `implementations:`, a list of
records with exactly these keys (a key marked ? may be left out):

    id            unique, [a-z][a-z0-9_]*
    op            fully_connected | depthwise_conv | avg_pool | max_pool
    quantization  per_tensor | per_channel | none
    dtypes        {operand pointer: int8 | int16 | int32 | int64 | float16 | float32}
    layout        {operand pointer: NHWC | NC | OI | 1HWO}
    entry         the function to call: an exported symbol returning arm_cmsis_nn_status
    operands?     the exported function whose parameters are the operator's operands; defaults to the entry. Every
                  entry parameter is either an operand of the same name and type or the param of a buffer.
    routed_by     `direct`, or {wrapper, route, requires, when}: the wrapper that selects this entry, a stable route
                  id `<wrapper stem>:<route>`, the build features under which the wrapper has the route, and the
                  shape condition in prose. An entry equal to its wrapper is reachable only through that route.
    requires      build features under which the entry computes the operator (all must hold)
    varies_with?  [toolchain]: the route or a size query answers differently under armclang
    eligibility?  {when, predicate?, args?}: the layer gate in prose, and the function that evaluates it (exported, or
                  a static inline helper of the public headers that the planner exposes)
    buffers       list of {name, param, size_query, units, align, lifetime, optional, format?, note?}
                    param       the entry parameter that carries it: a cmsis_nn_context (its buf), a const pointer
                                operand replaced by prepared data (needs a format), or null for a buffer that only
                                feeds a later prepare step
                    size_query  `none` (a context with no scratch), or {symbol, args}: an exported sizer returning
                                int32_t, every parameter bound
                    units       bytes
                    align       minimum alignment of the memory, a power of two
                    lifetime    call (scratch for one call) | prepared (filled once by prepare, kept for every call) |
                                prepare (exists only while the prepare steps run)
                    optional    true when buf may be NULL whenever the size query returns 0 for a shape with
                                positive dimensions (a zero channel count is not covered)
    prepare       ordered list of {symbol, args, note?}: exported functions returning arm_cmsis_nn_status that fill
                  one buffer each. A step runs when the buffer it fills has a non-zero size.
    declines      list of {cause: build | layer | buffer, when}: when the entry returns ARM_CMSIS_NN_NO_IMPL_ERROR
    since         the first release whose headers declare every referenced symbol as they are today
    numerics      {reference: hct_ref:<helia-core-tester C reference entry>, claim: bit_exact | bounded, holds_with}
    note?         prose

Argument bindings (size_query, prepare and eligibility `args`) name, for every parameter of the called function, an
operand (`filter_data`), a field path into a struct operand (`fc_params.input_offset`, `output_dims.c`), a buffer
(`buffer:<name>`) or null. They are references, never expressions: formulas stay in the C the bindings point at.

The export adds `family` to each record and, to each prepare step, `fills` (the buffer it writes) and `consumes` (the
operands and fields it reads apart from shapes: a change to any of them invalidates the prepared data). Optional keys
are always written, as null or [].

Feature tokens: mvei, dsp, mvef, mve_f16, f32_api, f16_api, their negations (!mvei, ...), !autovectorize and
!single_rounding. Every #if guard on a referenced symbol must be implied by `requires`; a float dtype or an
ARM_NN_ENABLE_F32/F16 guard needs the matching f32_api/f16_api, and the reverse.

The JSON envelope binds the manifest to its inputs: library_version (Include/arm_nn_types.h), signature_schema and
signature_sha256 (the bytes of kernel_contracts.json) and planner_abi (ARM_NN_PLAN_ABI_VERSION in
Planner/include/arm_nn_plan.h, null without a planner). The canonical form is JSON.stringify(document, null, 2) plus a
newline, so release-please's json updater can bump library_version without un-canonicalizing the file.

Fail-closed: any unresolved reference, unknown key or token, or type mismatch aborts the export with every problem
listed and no file written. Refs AmbiqAI/ns-cmsis-nn#885, #829.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parent))
from check_doxygen_params import parse_header, public_headers  # noqa: E402
from check_kernel_contract import NOISE_GUARD_RE, validate_document  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'python'))
from ns_cmsis_nn_contracts.vocabulary import (  # noqa: E402
    SCHEMA,
    BASE_FEATURES,
    FEATURES,
    VARIES_WITH,
    OPS,
    QUANTIZATIONS,
    DTYPES,
    LAYOUTS,
    LIFETIMES,
    DECLINE_CAUSES,
    CLAIMS,
    ID_RE,
    ROUTE_RE,
    FORMAT_RE,
    REFERENCE_RE,
    VERSION_RE,
)
SOURCES = Path('Tests/KernelContracts/implementations')
SIGNATURES = Path('Tests/KernelContracts/kernel_contracts.json')
OUTPUT = Path('Tests/KernelContracts/kernel_implementations.json')
TYPES_H = Path('Include/arm_nn_types.h')
PLAN_H = Path('Planner/include/arm_nn_plan.h')

GUARD_TOKENS = {
    'defined(ARM_MATH_MVEI)': 'mvei',
    '!defined(ARM_MATH_MVEI)': '!mvei',
    'defined(ARM_MATH_DSP)': 'dsp',
    '!defined(ARM_MATH_DSP)': '!dsp',
    'defined(ARM_MATH_MVEF)': 'mvef',
    'defined(ARM_MATH_MVE_FLOAT16)': 'mve_f16',
    '!defined(ARM_MATH_AUTOVECTORIZE)': '!autovectorize',
    '!defined(CMSIS_NN_USE_SINGLE_ROUNDING)': '!single_rounding',
    'ARM_NN_ENABLE_F32': 'f32_api',
    'ARM_NN_ENABLE_F16': 'f16_api',
}
FLOAT_API = {'float32': 'f32_api', 'float16': 'f16_api'}
STATUS = 'arm_cmsis_nn_status'
CONTEXT = 'cmsis_nn_context *'
DIMS = 'cmsis_nn_dims *'
INT_ALIASES = {'int': 'int32_t'}

RECORD_KEYS = ('id', 'op', 'quantization', 'dtypes', 'layout', 'entry', 'operands', 'routed_by', 'requires',
               'varies_with', 'eligibility', 'buffers', 'prepare', 'declines', 'since', 'numerics', 'note')
RECORD_OPTIONAL = frozenset({'operands', 'varies_with', 'eligibility', 'note'})
ROUTE_KEYS = frozenset({'wrapper', 'route', 'requires', 'when'})
BUFFER_KEYS = ('name', 'param', 'size_query', 'units', 'align', 'lifetime', 'optional', 'format', 'note')
BUFFER_OPTIONAL = frozenset({'format', 'note'})
QUERY_KEYS = frozenset({'symbol', 'args'})
STEP_AUTHORED = frozenset({'symbol', 'args'})
STEP_OPTIONAL = frozenset({'note'})
ELIGIBILITY_KEYS = frozenset({'when', 'predicate', 'args'})
DECLINE_KEYS = frozenset({'cause', 'when'})
NUMERICS_KEYS = frozenset({'reference', 'claim', 'holds_with'})

FIELD_PATH_RE = re.compile(r'^[A-Za-z_]\w*(\.[A-Za-z_]\w*)+$')
NAME_RE = re.compile(r'^[A-Za-z_]\w*$')


class ExportError(ValueError):
    pass


class Problems:
    def __init__(self):
        self.items = []

    def add(self, where, message):
        self.items.append(f'{where}: {message}')

    def raise_if_any(self):
        if self.items:
            raise ExportError(f'{len(self.items)} problem(s) in the implementation manifest:\n  '
                              + '\n  '.join(self.items))


def _nonempty_str(value):
    return isinstance(value, str) and bool(value.strip())


def _member(value, vocabulary):
    return isinstance(value, str) and value in vocabulary


def norm_type(ctype):
    return re.sub(r'\s*\*\s*', ' *', re.sub(r'\s+', ' ', ctype)).replace('* *', '**').strip()


def is_pointer(ctype):
    return norm_type(ctype).endswith('*')


def _value_type(ctype):
    """A by-value type without its top-level const, with int spelled as int32_t."""
    ctype = norm_type(ctype)
    if ctype.startswith('const '):
        ctype = ctype[len('const '):]
    return INT_ALIASES.get(ctype, ctype)


def pointee(ctype):
    return norm_type(ctype)[:-1].strip()


def accepts(param_type, arg_type):
    """Whether an argument of arg_type may be passed to a parameter of param_type."""
    if not is_pointer(param_type):
        return not is_pointer(arg_type) and _value_type(param_type) == _value_type(arg_type)
    if not is_pointer(arg_type):
        return False
    wanted, given = pointee(param_type), pointee(arg_type)
    if wanted.startswith('const '):
        return _value_type(wanted) == _value_type(given)
    return wanted == given


def struct_name(ctype):
    """The struct a `[const] S *` parameter points at."""
    return _value_type(pointee(ctype)) if is_pointer(ctype) else None


def parse_structs(text):
    """{typedef name: {field: type}} for every `typedef struct {...} name;` in a header."""
    text = re.sub(r'/\*.*?\*/|//[^\n]*', ' ', text, flags=re.S)
    structs = {}
    for body, name in re.findall(r'typedef\s+struct\s*\w*\s*\{(.*?)\}\s*(\w+)\s*;', text, flags=re.S):
        fields = {}
        for decl in body.split(';'):
            decl = ' '.join(decl.split())
            if not decl:
                continue
            match = re.fullmatch(r'(.+?)\s*\b(\w+)\s*(\[[^\]]*\])?', decl)
            if match is None:
                raise ExportError(f'struct {name}: cannot parse field {decl!r}')
            fields[match.group(2)] = norm_type(match.group(1)) + (' *' if match.group(3) else '')
        structs[name] = fields
    return structs


def library_version(root):
    text = (root / TYPES_H).read_text(encoding='utf-8')
    parts = []
    for scope in ('MAJOR', 'MINOR', 'PATCH'):
        match = re.search(rf'#define\s+NS_CMSIS_NN_VERSION_{scope}\s*\((\d+)\)', text)
        if match is None:
            raise ExportError(f'{TYPES_H}: no NS_CMSIS_NN_VERSION_{scope}')
        parts.append(match.group(1))
    return '.'.join(parts), text


def planner_abi(root):
    path = root / PLAN_H
    if not path.is_file():
        return None
    match = re.search(r'#define\s+ARM_NN_PLAN_ABI_VERSION\s*\(?\s*(\d+)', path.read_text(encoding='utf-8'))
    if match is None:
        raise ExportError(f'{PLAN_H}: exists but defines no ARM_NN_PLAN_ABI_VERSION')
    return int(match.group(1))


def _version_key(version):
    return tuple(int(part) for part in version.split('.'))


def load_yaml(path):
    try:
        import yaml
    except ImportError:
        raise ExportError('PyYAML is required: pip install pyyaml==6.0.3') from None

    class UniqueKeyLoader(yaml.SafeLoader):
        pass

    def mapping(loader, node, deep=False):
        keys = set()
        for key_node, _ in node.value:
            key = loader.construct_object(key_node, deep=deep)
            if key in keys:
                raise ExportError(f'{path}:{key_node.start_mark.line + 1}: duplicate key {key!r}')
            keys.add(key)
        return loader.construct_mapping(node, deep=deep)

    UniqueKeyLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, mapping)
    text = path.read_text(encoding='utf-8')
    if not text.isascii():
        raise ExportError(f'{path}: non-ASCII text; the export must stay byte-identical under JSON.stringify')
    try:
        return yaml.load(text, Loader=UniqueKeyLoader)
    except yaml.YAMLError as error:
        raise ExportError(f'{path}: {error}') from None


class Tree:
    """The inputs of one export: the signature export, the public headers and the authored YAML under one root."""

    def __init__(self, root):
        self.root = Path(root)
        signature_path = self.root / SIGNATURES
        try:
            self.signature_bytes = signature_path.read_bytes()
            signatures = json.loads(self.signature_bytes)
        except (OSError, ValueError) as error:
            raise ExportError(f'{signature_path}: cannot be read as JSON ({error})') from None
        validate_document(signatures, signature_path)
        self.signature_schema = signatures['schema']
        self.functions = {record['name']: record for record in signatures['functions']}
        self.statics = {}
        for header in public_headers(self.root / 'Include'):
            for decl in parse_header(header):
                if decl.is_static and not decl.error:
                    self.statics[decl.name] = {
                        'name': decl.name,
                        'returns': decl.ret,
                        'guards': [g for g in decl.guards if not NOISE_GUARD_RE.match(g)],
                        'params': [{'name': p.name, 'type': p.type, 'direction': 'in'} for p in decl.params],
                    }
        self.version, types_text = library_version(self.root)
        self.structs = parse_structs(types_text)
        self.planner_abi = planner_abi(self.root)


def _check_keys(problems, where, value, keys, optional=frozenset()):
    if not isinstance(value, dict):
        problems.add(where, f'expected a mapping, got {type(value).__name__}')
        return False
    unknown = sorted(map(str, set(value) - set(keys)))
    missing = sorted(set(keys) - set(optional) - set(value))
    if unknown:
        problems.add(where, f'unknown key(s) {unknown}')
    if missing:
        problems.add(where, f'missing key(s) {missing}')
    return not unknown and not missing


def _check_tokens(problems, where, tokens):
    if not isinstance(tokens, list) or not all(isinstance(token, str) for token in tokens):
        problems.add(where, 'expected a list of feature tokens')
        return set()
    if len(set(tokens)) != len(tokens):
        problems.add(where, f'duplicate feature token in {tokens}')
    unknown = sorted(set(tokens) - FEATURES)
    if unknown:
        problems.add(where, f'unknown feature token(s) {unknown}; the vocabulary is {sorted(FEATURES)}')
    clash = sorted(token for token in tokens if '!' + token in tokens)
    if clash:
        problems.add(where, f'contradictory feature tokens {clash} and their negations')
    return set(tokens) & FEATURES


class RecordChecker:
    def __init__(self, tree, problems, where, record):
        self.tree, self.problems, self.where, self.record = tree, problems, where, record
        self.referenced = []
        self.buffers = {}

    def add(self, message, where=None):
        self.problems.add(where or self.where, message)

    def function(self, name, where, returns=None, allow_static=False):
        if not _nonempty_str(name):
            self.add('expected a function name', where)
            return None
        record = self.tree.functions.get(name)
        if record is None and allow_static:
            record = self.tree.statics.get(name)
        if record is None:
            scope = 'the signature export or a static inline helper' if allow_static else 'the signature export'
            self.add(f'{name} is not in {scope}', where)
            return None
        if returns is not None and norm_type(record['returns']) not in returns:
            self.add(f'{name} returns {record["returns"]}, expected {" or ".join(sorted(returns))}', where)
        self.referenced.append(record)
        return record

    def resolve(self, binding, where):
        """(kind, type, operand) for one argument binding, or None after reporting why it is invalid."""
        if binding is None:
            return 'null', None, None
        if not isinstance(binding, str):
            self.add(f'binding {binding!r} is not a string or null', where)
            return None
        if binding.startswith('buffer:'):
            name = binding[len('buffer:'):]
            if name not in self.buffers:
                self.add(f'binding {binding!r} names no buffer of this record', where)
                return None
            return 'buffer', None, name
        if FIELD_PATH_RE.match(binding):
            head, *fields = binding.split('.')
            if head not in self.operands:
                self.add(f'binding {binding!r}: {head} is not an operand of {self.operand_symbol}', where)
                return None
            ctype = self.operands[head]['type']
            for field in fields:
                struct = struct_name(ctype) if is_pointer(ctype) else _value_type(ctype)
                if struct not in self.tree.structs or field not in self.tree.structs[struct]:
                    self.add(f'binding {binding!r}: {struct} has no field {field}', where)
                    return None
                ctype = self.tree.structs[struct][field]
            return 'field', ctype, head
        if NAME_RE.match(binding):
            if binding not in self.operands:
                self.add(f'binding {binding!r} is not an operand of {self.operand_symbol}', where)
                return None
            return 'operand', self.operands[binding]['type'], binding
        self.add(f'binding {binding!r} is not an operand, a field path, buffer:<name> or null', where)
        return None

    def bind(self, function, args, where, allow_buffers):
        """Check that args binds every parameter of function; returns [(param, kind, type, target)]."""
        if not isinstance(args, dict) or not all(isinstance(key, str) for key in args):
            self.add('args must be a mapping of parameter name to binding', where)
            return []
        names = [param['name'] for param in function['params']]
        if sorted(args) != sorted(names):
            self.add(f'args must bind exactly the parameters of {function["name"]} {names}, got {sorted(args)}',
                     where)
        bound = []
        for param in function['params']:
            if param['name'] not in args:
                continue
            here = f'{where}.{param["name"]}'
            resolved = self.resolve(args[param['name']], here)
            if resolved is None:
                continue
            kind, ctype, target = resolved
            if kind == 'buffer' and not allow_buffers:
                self.add('a buffer cannot be bound here', here)
                continue
            if kind == 'null' and not is_pointer(param['type']):
                self.add(f'null bound to {param["type"]}, which is not a pointer', here)
                continue
            if kind == 'buffer' and not is_pointer(param['type']):
                self.add(f'buffer bound to {param["type"]}, which is not a pointer', here)
                continue
            if kind in ('operand', 'field') and not accepts(param['type'], ctype):
                self.add(f'{args[param["name"]]} is {ctype}, which {param["type"]} does not accept', here)
                continue
            bound.append((param, kind, ctype, target))
        return bound

    def check(self):
        record, add = self.record, self.add
        if not _check_keys(self.problems, self.where, record, RECORD_KEYS, RECORD_OPTIONAL):
            return None
        if not isinstance(record['id'], str) or not ID_RE.match(record['id']):
            add(f'id {record["id"]!r} must match {ID_RE.pattern}')
        if not _member(record['op'], OPS):
            add(f'op {record["op"]!r} is not one of {sorted(OPS)}')
        if not _member(record['quantization'], QUANTIZATIONS):
            add(f'quantization {record["quantization"]!r} is not one of {sorted(QUANTIZATIONS)}')

        entry = self.function(record['entry'], f'{self.where}.entry', {STATUS})
        self.operand_symbol = record.get('operands', record['entry'])
        operand_fn = entry if 'operands' not in record else self.function(
            record['operands'], f'{self.where}.operands', {STATUS})
        if 'operands' in record and record['operands'] == record['entry']:
            add('operands repeats the entry; leave it out')
        if entry is None or operand_fn is None:
            return None
        self.operands = {param['name']: param for param in operand_fn['params']}
        entry_params = {param['name']: param for param in entry['params']}

        requires = _check_tokens(self.problems, f'{self.where}.requires', record['requires'])
        self.check_operand_maps(requires)
        self.check_route(entry, requires)
        varies_with = record.get('varies_with', [])
        if not isinstance(varies_with, list) or not all(_member(v, VARIES_WITH) for v in varies_with) \
                or len(set(varies_with)) != len(varies_with):
            add(f'varies_with must be a list of distinct tokens from {sorted(VARIES_WITH)}')
        eligibility = self.check_eligibility()
        buffers = self.check_buffers(entry_params)
        steps = self.check_prepare()
        self.check_entry_params(entry_params)
        self.check_declines(requires)
        self.check_since()
        self.check_numerics()
        if 'note' in record and not _nonempty_str(record['note']):
            add('note must be a nonempty string')
        self.check_guards(requires)
        return {
            'id': record['id'],
            'op': record['op'],
            'quantization': record['quantization'],
            'dtypes': self.ordered(record['dtypes']),
            'layout': self.ordered(record['layout']),
            'entry': record['entry'],
            'operands': self.operand_symbol,
            'routed_by': record['routed_by'] if record['routed_by'] == 'direct' else {
                key: record['routed_by'][key] for key in ('wrapper', 'route', 'requires', 'when')},
            'requires': record['requires'],
            'varies_with': varies_with,
            'eligibility': eligibility,
            'buffers': buffers,
            'prepare': steps,
            'declines': [{'cause': d['cause'], 'when': d['when']} for d in record['declines']],
            'since': record['since'],
            'numerics': {key: record['numerics'][key] for key in ('reference', 'claim', 'holds_with')},
            'note': record.get('note'),
        }

    def ordered(self, mapping):
        order = list(self.operands)
        return {key: mapping[key] for key in sorted(mapping, key=order.index)}

    def check_operand_maps(self, requires):
        for field, vocabulary in (('dtypes', DTYPES), ('layout', LAYOUTS)):
            mapping = self.record[field]
            where = f'{self.where}.{field}'
            if not isinstance(mapping, dict) or not mapping:
                self.add('expected a nonempty mapping of operand to value', where)
                self.record[field] = {}
                continue
            for operand, value in mapping.items():
                param = self.operands.get(operand) if isinstance(operand, str) else None
                if param is None:
                    self.add(f'{operand} is not an operand of {self.operand_symbol}', where)
                elif not is_pointer(param['type']) or struct_name(param['type']) in self.tree.structs:
                    self.add(f'{operand} ({param["type"]}) is not a data operand', where)
                if not _member(value, vocabulary):
                    self.add(f'{operand}: {value!r} is not one of {sorted(vocabulary)}', where)
            self.record[field] = {k: v for k, v in mapping.items() if _member(k, self.operands)}
        for dtype, api in FLOAT_API.items():
            if dtype in self.record['dtypes'].values() and api not in requires:
                self.add(f'a {dtype} operand needs {api} in requires', f'{self.where}.requires')

    def check_route(self, entry, requires):
        routed_by, where = self.record['routed_by'], f'{self.where}.routed_by'
        if routed_by == 'direct':
            return
        if not _check_keys(self.problems, where, routed_by, ROUTE_KEYS):
            self.record['routed_by'] = 'direct'
            return
        wrapper = self.function(routed_by['wrapper'], f'{where}.wrapper', {STATUS})
        if not isinstance(routed_by['route'], str) or not ROUTE_RE.match(routed_by['route']):
            self.add(f'route {routed_by["route"]!r} must match {ROUTE_RE.pattern}', where)
        route_requires = _check_tokens(self.problems, f'{where}.requires', routed_by['requires'])
        if not _nonempty_str(routed_by['when']):
            self.add('when must describe the route condition', where)
        if wrapper is None:
            return
        wrapper_params = {param['name'] for param in wrapper['params']}
        missing = [param['name'] for param in entry['params'] if param['name'] not in wrapper_params]
        if routed_by['wrapper'] != entry['name'] and missing:
            self.add(f'{entry["name"]} parameters {missing} are not parameters of {wrapper["name"]}', where)
        if routed_by['wrapper'] == entry['name'] and not route_requires <= requires:
            self.add('an entry reachable only through its route must require the route\'s features '
                     f'{sorted(route_requires - requires)}', where)

    def check_eligibility(self):
        if 'eligibility' not in self.record:
            return None
        eligibility, where = self.record['eligibility'], f'{self.where}.eligibility'
        if not _check_keys(self.problems, where, eligibility, ELIGIBILITY_KEYS, {'predicate', 'args'}):
            return None
        if not _nonempty_str(eligibility['when']):
            self.add('when must describe the gate', where)
        if ('predicate' in eligibility) != ('args' in eligibility):
            self.add('predicate and args go together', where)
            return None
        result = {'when': eligibility['when'], 'predicate': None, 'predicate_static': None, 'args': None}
        if 'predicate' in eligibility:
            predicate = self.function(eligibility['predicate'], f'{where}.predicate', {'int32_t', 'bool'},
                                      allow_static=True)
            if predicate is not None:
                self.bind(predicate, eligibility['args'], f'{where}.args', allow_buffers=False)
                result.update(predicate=eligibility['predicate'],
                              predicate_static=eligibility['predicate'] not in self.tree.functions,
                              args=self.ordered_args(predicate, eligibility['args']))
        return result

    @staticmethod
    def ordered_args(function, args):
        return {param['name']: args.get(param['name']) for param in function['params']} \
            if isinstance(args, dict) else {}

    def check_buffers(self, entry_params):
        buffers, where = self.record['buffers'], f'{self.where}.buffers'
        if not isinstance(buffers, list):
            self.add('expected a list', where)
            return []
        for index, buffer in enumerate(buffers):
            here = f'{where}[{index}]'
            if not _check_keys(self.problems, here, buffer, BUFFER_KEYS, BUFFER_OPTIONAL):
                continue
            if not isinstance(buffer['name'], str) or not ID_RE.match(buffer['name']):
                self.add(f'name {buffer["name"]!r} must match {ID_RE.pattern}', here)
                continue
            if buffer['name'] in self.buffers:
                self.add(f'duplicate buffer name {buffer["name"]}', here)
                continue
            self.buffers[buffer['name']] = dict(buffer, _kind=None, _filled=False, _read=False,
                                                _where=f'{where}.{buffer["name"]}')
        out = []
        params_seen = set()
        for name, buffer in self.buffers.items():
            here = buffer['_where']
            param, lifetime = buffer['param'], buffer['lifetime']
            if not _member(lifetime, LIFETIMES):
                self.add(f'lifetime {lifetime!r} is not one of {sorted(LIFETIMES)}', here)
            if param is None:
                buffer['_kind'] = 'internal'
                if lifetime != 'prepare':
                    self.add('a buffer passed to no entry parameter must have lifetime prepare', here)
            elif not _member(param, entry_params):
                self.add(f'param {param!r} is not a parameter of {self.record["entry"]}', here)
            elif param in params_seen:
                self.add(f'param {param} carries two buffers', here)
            else:
                params_seen.add(param)
                ptype, direction = norm_type(entry_params[param]['type']), entry_params[param]['direction']
                if _value_type(ptype) == CONTEXT:
                    buffer['_kind'] = 'context'
                    if lifetime == 'prepare':
                        self.add('a context parameter cannot have lifetime prepare', here)
                    if lifetime == 'prepared' and direction != 'in':
                        self.add(f'a prepared context is only read by the entry, but {param} is [{direction}]', here)
                    if direction == 'out':
                        self.add(f'{param} is documented [out]; a context the caller fills cannot be [out]', here)
                else:
                    buffer['_kind'] = 'operand'
                    if not is_pointer(ptype) or not pointee(ptype).startswith('const ') or direction != 'in':
                        self.add(f'{param} is {ptype} [{direction}]: a buffer must be a cmsis_nn_context * or a '
                                 'const pointer [in] that prepared data replaces', here)
                    if lifetime != 'prepared':
                        self.add('a buffer that replaces an operand must have lifetime prepared', here)
                    if 'format' not in buffer:
                        self.add('a buffer that replaces an operand needs a format', here)
            if 'format' in buffer and (not isinstance(buffer['format'], str) or not FORMAT_RE.match(buffer['format'])):
                self.add(f'format {buffer["format"]!r} must match {FORMAT_RE.pattern}', here)
            if buffer['units'] != 'bytes':
                self.add(f'units {buffer["units"]!r} must be bytes', here)
            align = buffer['align']
            if isinstance(align, bool) or not isinstance(align, int) or align < 1 or align > 64 or align & (align - 1):
                self.add(f'align {align!r} must be a power of two from 1 to 64', here)
            if not isinstance(buffer['optional'], bool):
                self.add('optional must be true or false', here)
            if 'note' in buffer and not _nonempty_str(buffer['note']):
                self.add('note must be a nonempty string', here)
            query = self.check_size_query(buffer, here)
            out.append({
                'name': name,
                'param': param,
                'size_query': query,
                'units': buffer['units'],
                'align': align,
                'lifetime': lifetime,
                'optional': buffer['optional'],
                'format': buffer.get('format'),
                'note': buffer.get('note'),
            })
        return out

    def check_size_query(self, buffer, where):
        query = buffer['size_query']
        if query == 'none':
            if buffer['_kind'] != 'context' or buffer['lifetime'] != 'call' or buffer['optional'] is not True:
                self.add('size_query none is only for an optional call-lifetime context with no scratch', where)
            return None
        if not _check_keys(self.problems, f'{where}.size_query', query, QUERY_KEYS):
            return None
        sizer = self.function(query['symbol'], f'{where}.size_query.symbol', {'int32_t'})
        if sizer is None:
            return None
        self.bind(sizer, query['args'], f'{where}.size_query.args', allow_buffers=False)
        return {'symbol': query['symbol'], 'args': self.ordered_args(sizer, query['args'])}

    def check_prepare(self):
        steps, where = self.record['prepare'], f'{self.where}.prepare'
        if not isinstance(steps, list):
            self.add('expected a list', where)
            return []
        out = []
        for index, step in enumerate(steps):
            here = f'{where}[{index}]'
            if not _check_keys(self.problems, here, step, STEP_AUTHORED | STEP_OPTIONAL, STEP_OPTIONAL):
                continue
            function = self.function(step['symbol'], f'{here}.symbol', {STATUS})
            if function is None:
                continue
            fills, consumes = [], []
            for param, kind, _, target in self.bind(function, step['args'], f'{here}.args', allow_buffers=True):
                writes = is_pointer(param['type']) and param['direction'] in ('out', 'in,out') \
                    and not pointee(param['type']).startswith('const ')
                if kind == 'buffer':
                    buffer = self.buffers[target]
                    if writes:
                        if buffer['_filled']:
                            self.add(f'buffer {target} is filled twice', here)
                        fills.append(target)
                        buffer['_filled'] = True
                    else:
                        if not buffer['_filled']:
                            self.add(f'buffer {target} is read before a step fills it', here)
                        buffer['_read'] = True
                elif writes and kind != 'null':
                    self.add(f'{param["name"]} is written, so it must be bound to a buffer, not {target}', here)
                elif kind in ('operand', 'field') and _value_type(self.operands[target]['type']) != DIMS:
                    ref = step['args'][param['name']]
                    if ref not in consumes:
                        consumes.append(ref)
            if len(fills) != 1:
                self.add(f'a prepare step must fill exactly one buffer, {function["name"]} fills {fills}', here)
            if 'note' in step and not _nonempty_str(step['note']):
                self.add('note must be a nonempty string', here)
            out.append({'symbol': step['symbol'], 'args': self.ordered_args(function, step['args']),
                        'consumes': consumes, 'fills': fills[0] if len(fills) == 1 else None,
                        'note': step.get('note')})
        for name, buffer in self.buffers.items():
            if buffer['lifetime'] in ('prepared', 'prepare') and not buffer['_filled']:
                self.add(f'buffer {name} has lifetime {buffer["lifetime"]} but no prepare step fills it',
                         buffer['_where'])
            if buffer['lifetime'] == 'call' and buffer['_filled']:
                self.add(f'buffer {name} is call scratch, so no prepare step may fill it', buffer['_where'])
            if buffer['lifetime'] == 'prepare' and not buffer['_read']:
                self.add(f'buffer {name} exists only for preparation but no later step reads it', buffer['_where'])
        return out

    def check_entry_params(self, entry_params):
        carried = {buffer['param'] for buffer in self.buffers.values()}
        for name, param in entry_params.items():
            if name in carried:
                continue
            if _value_type(param['type']) == CONTEXT:
                self.add(f'context parameter {name} of {self.record["entry"]} has no buffer; '
                         'use size_query none if it needs no scratch', f'{self.where}.buffers')
            elif name not in self.operands:
                self.add(f'entry parameter {name} is neither an operand of {self.operand_symbol} nor a buffer param',
                         f'{self.where}.entry')
            elif norm_type(self.operands[name]['type']) != norm_type(param['type']):
                self.add(f'entry parameter {name} is {param["type"]} but operand {name} is '
                         f'{self.operands[name]["type"]}', f'{self.where}.entry')

    def check_declines(self, requires):
        declines, where = self.record['declines'], f'{self.where}.declines'
        if not isinstance(declines, list):
            self.add('expected a list', where)
            self.record['declines'] = []
            return
        causes = []
        for index, decline in enumerate(declines):
            here = f'{where}[{index}]'
            if not _check_keys(self.problems, here, decline, DECLINE_KEYS):
                continue
            if not _member(decline['cause'], DECLINE_CAUSES):
                self.add(f'cause {decline["cause"]!r} is not one of {sorted(DECLINE_CAUSES)}', here)
            if decline['cause'] in causes:
                self.add(f'cause {decline["cause"]} listed twice; describe it in one entry', here)
            causes.append(decline['cause'])
            if not _nonempty_str(decline['when']):
                self.add('when must describe the decline', here)
        if 'build' in causes and not requires:
            self.add('a build decline needs the features it lacks in requires', where)
        self.record['declines'] = [d for d in declines if isinstance(d, dict) and DECLINE_KEYS <= set(d)]

    def check_since(self):
        since = self.record['since']
        if not isinstance(since, str) or not VERSION_RE.match(since):
            self.add(f'since {since!r} must be a MAJOR.MINOR.PATCH version string', f'{self.where}.since')
        elif _version_key(since) > _version_key(self.tree.version):
            self.add(f'since {since} is after the library version {self.tree.version}', f'{self.where}.since')

    def check_numerics(self):
        numerics, where = self.record['numerics'], f'{self.where}.numerics'
        if not _check_keys(self.problems, where, numerics, NUMERICS_KEYS):
            self.record['numerics'] = {key: None for key in NUMERICS_KEYS}
            return
        if not isinstance(numerics['reference'], str) or not REFERENCE_RE.match(numerics['reference']):
            self.add(f'reference {numerics["reference"]!r} must match {REFERENCE_RE.pattern}', where)
        if not _member(numerics['claim'], CLAIMS):
            self.add(f'claim {numerics["claim"]!r} is not one of {sorted(CLAIMS)}', where)
        _check_tokens(self.problems, f'{where}.holds_with', numerics['holds_with'])

    def check_guards(self, requires):
        """Every #if guard of a referenced symbol must follow from requires, and f*_api only from a float guard."""
        float_guarded = set()
        for function in self.referenced:
            for guard in function['guards']:
                token = GUARD_TOKENS.get(guard.replace(' ', ''))
                if token is None:
                    self.add(f'{function["name"]} is declared under #if {guard}, which has no feature token; '
                             'teach GUARD_TOKENS this guard', f'{self.where}.requires')
                    continue
                if token in ('f32_api', 'f16_api'):
                    float_guarded.add(token)
                if token not in requires:
                    self.add(f'{function["name"]} is declared under #if {guard}, so requires needs {token}',
                             f'{self.where}.requires')
        for token in sorted({'f32_api', 'f16_api'} & requires - float_guarded):
            self.add(f'requires {token} but no referenced symbol is declared under its ARM_NN_ENABLE_* guard',
                     f'{self.where}.requires')


def build_document(root):
    tree = Tree(root)
    problems = Problems()
    source_dir = tree.root / SOURCES
    if not source_dir.is_dir():
        raise ExportError(f'{source_dir}: missing')
    paths = sorted(source_dir.iterdir())
    records, seen_ids, routes, route_prefix = [], {}, {}, {}
    for path in paths:
        if path.suffix != '.yaml' or not path.is_file():
            problems.add(str(path.relative_to(tree.root)), 'only <family>.yaml files belong here')
            continue
        where_file = path.relative_to(tree.root).as_posix()
        document = load_yaml(path)
        if not _check_keys(problems, where_file, document, ('family', 'implementations')):
            continue
        if document['family'] != path.stem:
            problems.add(where_file, f'family {document["family"]!r} must equal the file stem {path.stem!r}')
        if not isinstance(document['implementations'], list) or not document['implementations']:
            problems.add(where_file, 'implementations must be a nonempty list')
            continue
        for index, record in enumerate(document['implementations']):
            ident = record.get('id') if isinstance(record, dict) else None
            where = f'{where_file}:{ident if isinstance(ident, str) else f"implementations[{index}]"}'
            checked = RecordChecker(tree, problems, where, record).check()
            if checked is None:
                continue
            if checked['id'] in seen_ids:
                problems.add(where, f'duplicate id (also in {seen_ids[checked["id"]]})')
            seen_ids[checked['id']] = where_file
            route = checked['routed_by']
            if isinstance(route, dict) and isinstance(route['route'], str) and ROUTE_RE.match(route['route']):
                key = (route['wrapper'], route['route'])
                if key in routes:
                    problems.add(where, f'route {route["route"]} of {route["wrapper"]} is also claimed by '
                                        f'{routes[key]}')
                routes[key] = checked['id']
                prefix = ROUTE_RE.match(route['route']).group(1)
                other = route_prefix.setdefault(route['wrapper'], prefix)
                if other != prefix:
                    problems.add(where, f'routes of {route["wrapper"]} must share one prefix, {other}: and {prefix}:')
                owner = [w for w, p in route_prefix.items() if p == prefix and w != route['wrapper']]
                if owner:
                    problems.add(where, f'route prefix {prefix}: already belongs to {owner[0]}')
            records.append(dict(checked, family=document['family']))
    if not records and not problems.items:
        problems.add(str(SOURCES), 'no implementation records; an empty manifest would read as "no kernels"')
    problems.raise_if_any()
    order = ('id', 'family') + RECORD_KEYS[1:]
    records = [{key: record[key] for key in order} for record in sorted(records, key=lambda r: r['id'])]
    return {
        'schema': SCHEMA,
        'library_version': tree.version,
        'signature_schema': tree.signature_schema,
        'signature_sha256': hashlib.sha256(tree.signature_bytes).hexdigest(),
        'planner_abi': tree.planner_abi,
        'implementations': records,
    }


def render(document):
    """JSON.stringify(document, null, 2) + newline, byte for byte, for an ASCII document."""
    return json.dumps(document, indent=2, ensure_ascii=True) + '\n'


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


def check(root, output):
    fresh = build_document(root)
    if not output.is_file():
        raise ExportError(f'{output}: missing; run `{Path(sys.argv[0]).name} export`')
    text = output.read_text(encoding='utf-8')
    try:
        committed = json.loads(text)
    except ValueError as error:
        raise ExportError(f'{output}: cannot be read as JSON ({error})') from None
    if render(committed) != text:
        raise ExportError(f'{output}: not in canonical form; run `{Path(sys.argv[0]).name} export`')
    if committed == fresh:
        return len(fresh['implementations'])
    details = []
    if not isinstance(committed, dict):
        committed = {}
    for key in ('schema', 'library_version', 'signature_schema', 'signature_sha256', 'planner_abi'):
        if committed.get(key) != fresh[key]:
            details.append(f'  {key}: {committed.get(key)!r} -> {fresh[key]!r}')
    old = committed.get('implementations')
    old = {r.get('id'): r for r in old if isinstance(r, dict)} if isinstance(old, list) else {}
    new = {r['id']: r for r in fresh['implementations']}
    details += [f'  + {ident}' for ident in sorted(set(new) - set(old), key=str)]
    details += [f'  - {ident}' for ident in sorted(set(old) - set(new), key=str)]
    details += [f'  ~ {ident}' for ident in sorted(set(old) & set(new), key=str) if old[ident] != new[ident]]
    raise ExportError(f'{output}: stale; run `{Path(sys.argv[0]).name} export`\n' + '\n'.join(details[:20]))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('command', nargs='?', choices=('export', 'check'), default='check',
                        help='export rewrites the manifest; check (default) fails if it is stale or not canonical')
    parser.add_argument('--root', default=str(REPO), help='repository tree to read (default: this checkout)')
    parser.add_argument('--output', help=f'manifest path (default: <root>/{OUTPUT})')
    args = parser.parse_args()
    root = Path(args.root)
    output = Path(args.output) if args.output else root / OUTPUT
    try:
        if args.command == 'export':
            document = build_document(root)
            write_atomically(output, render(document))
            print(f'OK: exported {len(document["implementations"])} kernel implementations to {output}')
        else:
            count = check(root, output)
            print(f'OK: {output} is fresh ({count} kernel implementations)')
    except (OSError, ValueError) as error:
        print(f'ERROR: {error}', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
