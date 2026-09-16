#!/usr/bin/env python3
# SPDX-License-Identifier: LicenseRef-Ambiq-Apollo-SDK
"""Check that every public header function has complete, direction-correct @param docs.

The doc block above each declaration in Include/ is the machine-readable half of the
kernel contract: the C types cannot say whether a mutable pointer is written only or
read first, so the @param[out] / @param[in,out] tags have to be trustworthy. Doxygen's
own WARN_NO_PARAMDOC only fires when a block already carries at least one tag, so a
brief-only block passes it silently; this check reads the headers directly, needs no
doxygen and no build, and fails on any of: a declaration with no doc block, a missing
or extra @param, a tag without a direction, a direction that contradicts the
const-ness of the parameter type, or a @copydoc whose target is unknown, cyclic, or
itself incomplete. See AmbiqAI/ns-cmsis-nn#526.
"""

import argparse
from pathlib import Path
import re
import sys

REPO = Path(__file__).resolve().parents[1]
DEFAULT_HEADERS = (
    'arm_nnfunctions.h',
    'arm_nnfunctions_flt.h',
    'arm_nnsupportfunctions.h',
    'arm_nnsupportfunctions_flt.h',
)
CONDITIONAL_RE = re.compile(r'^#\s*(if|ifdef|ifndef|elif|else|endif)\b')
DECL_RE = re.compile(r'^(?P<ret>[^;{}()]*?)\b(?P<name>[A-Za-z_]\w*)\s*\((?P<params>.*)\)\s*$', re.S)
NOT_A_FUNCTION_RE = re.compile(r'^(typedef|struct|union|enum)\b')
PARAM_TAG_RE = re.compile(r'[@\\]param\s*(?:\[(?P<dir>[^\]]*)\])?\s*(?P<name>[A-Za-z_]\w*)')
COPYDOC_RE = re.compile(r'[@\\]copy(?:doc|details)\s+(?P<name>[A-Za-z_]\w*)')
QUALIFIER_RE = re.compile(r'\b(__RESTRICT|__restrict|restrict)\b')
STATIC_RE = re.compile(r'\b(static|__STATIC_INLINE|__STATIC_FORCEINLINE)\b')
DIRECTIONS = ('in', 'out', 'in,out')
COMMENT_MARKER = '@@COMMENT@@'
# The buffer behind a const context pointer is written, so both tags are honest.
DIRECTION_EXCEPTIONS = {'const cmsis_nn_context *': {'in', 'in,out'}}


class Param:
    def __init__(self, text):
        text = ' '.join(QUALIFIER_RE.sub(' ', text).split())
        self.is_array = text.endswith(']')
        if self.is_array:
            text = text[:text.rindex('[')].rstrip()
        match = re.search(r'([A-Za-z_]\w*)$', text)
        if match is None or match.group(1) == text:
            raise ValueError(f'cannot parse parameter {text!r}')
        self.name = match.group(1)
        self.type = text[:match.start()].strip()
        self.type = re.sub(r'\s*\*\s*', ' * ', self.type).replace('  ', ' ').strip()

    def expected_directions(self):
        normalized = self.type.replace(' * ', ' *').strip()
        if normalized in DIRECTION_EXCEPTIONS:
            return DIRECTION_EXCEPTIONS[normalized]
        levels = self.type.split('*')
        if self.is_array:
            levels.append('')
        if len(levels) == 1:
            return {'in'}
        if all('const' in level.split() for level in levels[:-1]):
            return {'in'}
        return {'out', 'in,out'}


class Decl:
    def __init__(self, path, line, name, params, doc, is_static):
        self.path, self.line, self.name, self.params, self.doc = path, line, name, params, doc
        self.is_static = is_static


def blank_non_doc(text):
    """Replace // and /* */ comments (not /** */) and string literals with spaces."""
    out, i, n = [], 0, len(text)
    while i < n:
        two = text[i:i + 2]
        marker = False
        if two == '/*' and not text.startswith('/**', i):
            end = text.find('*/', i + 2)
            end = n if end < 0 else end + 2
            marker = True
        elif two == '//':
            end = text.find('\n', i)
            end = n if end < 0 else end
            marker = True
        elif text[i] == '"':
            end = i + 1
            while end < n and text[end] != '"' and text[end] != '\n':
                end += 2 if text[end] == '\\' else 1
            end = min(end + 1, n)
        else:
            out.append(text[i])
            i += 1
            continue
        blanked = re.sub(r'[^\n]', ' ', text[i:end])
        # A comment that starts its own line is a boundary between a doc block and whatever
        # follows; a trailing comment inside a statement is not.
        if marker and not text[text.rfind('\n', 0, i) + 1:i].strip():
            blanked = COMMENT_MARKER + blanked
        out.append(blanked)
        i = end
    return ''.join(out)


def logical_lines(text):
    """Yield (first_line_number, line) with backslash continuations joined."""
    lines = text.split('\n')
    i = 0
    while i < len(lines):
        start, line = i + 1, lines[i]
        while line.endswith('\\') and i + 1 < len(lines):
            i += 1
            line = line[:-1] + lines[i]
        yield start, line
        i += 1


def split_params(params):
    params = params.strip()
    if not params or params == 'void':
        return []
    parts, depth, current = [], 0, []
    for char in params:
        if char == ',' and depth == 0:
            parts.append(''.join(current))
            current = []
            continue
        depth += (char in '([') - (char in ')]')
        current.append(char)
    parts.append(''.join(current))
    return [Param(part) for part in parts]


def parse_header(path):
    """Return every function declaration in the header with its preceding doc block."""
    text = blank_non_doc(path.read_text())
    decls, pending_doc, depth = [], None, 0
    lines = list(logical_lines(text))
    i = 0
    while i < len(lines):
        lineno, line = lines[i]
        s = line.strip()
        i += 1
        if depth > 0:
            depth += s.count('{') - s.count('}')
            continue
        if s.startswith('#'):
            if not CONDITIONAL_RE.match(s):
                pending_doc = None
            continue
        if s.startswith(COMMENT_MARKER):
            pending_doc = None
            continue
        if s.startswith('/**'):
            block = [s]
            while '*/' not in block[-1] and i < len(lines):
                block.append(lines[i][1].strip())
                i += 1
            pending_doc = (lineno, '\n'.join(block))
            continue
        if not s or s == '}' or re.match(r'^extern\s*\{$', s):
            continue
        stmt, paren, opened = [], 0, False
        while True:
            for char in s:
                paren += (char == '(') - (char == ')')
            stmt.append(s)
            if paren == 0 and (s.endswith(';') or '{' in s):
                opened = '{' in s
                break
            if i >= len(lines):
                break
            s = lines[i][1].strip()
            i += 1
        joined = ' '.join(stmt)
        if opened:
            tail = joined[joined.index('{'):]
            depth = tail.count('{') - tail.count('}')
            joined = joined[:joined.index('{')]
        joined = joined.rstrip('; ').strip()
        match = DECL_RE.match(joined)
        if match and not NOT_A_FUNCTION_RE.match(joined):
            decls.append(Decl(path, lineno, match.group('name'),
                              split_params(match.group('params')), pending_doc,
                              bool(STATIC_RE.search(match.group('ret')))))
        pending_doc = None
    return decls


def doc_tags(doc):
    """Return (copydoc_target, [(name, direction)]) from a doc block."""
    copydocs = COPYDOC_RE.findall(doc)
    if len(copydocs) > 1:
        raise ValueError('multiple copy directives in one block: ' + ', '.join(copydocs))
    tags = [(m.group('name'), None if m.group('dir') is None else m.group('dir').replace(' ', ''))
            for m in PARAM_TAG_RE.finditer(doc)]
    return (copydocs[0] if copydocs else None), tags


def resolve_tags(decl, by_name, visiting):
    """Follow @copydoc chains and return the terminal tag list, raising on bad chains."""
    if decl.doc is None:
        raise ValueError('no doc block immediately before the declaration')
    target, tags = doc_tags(decl.doc[1])
    if target and tags:
        raise ValueError('doc block mixes @copydoc with @param tags')
    if target is None:
        return tags
    if target in visiting:
        raise ValueError(f'@copydoc cycle through {target}')
    if target not in by_name:
        raise ValueError(f'unknown @copydoc target {target}')
    if by_name[target].is_static:
        raise ValueError(f'@copydoc target {target} is a static function, which doxygen cannot resolve; '
                         'write the tags inline')
    return resolve_tags(by_name[target], by_name, visiting | {decl.name})


def check_decl(decl, by_name):
    tags = resolve_tags(decl, by_name, {decl.name})
    problems = []
    seen = {}
    for name, direction in tags:
        if name in seen:
            problems.append(f'duplicate @param {name}')
        seen[name] = direction
    params = {param.name: param for param in decl.params}
    for name in params:
        if name not in seen:
            problems.append(f'missing @param {name}')
    for name in seen:
        if name not in params:
            problems.append(f'@param {name}: no such parameter')
    for name, direction in seen.items():
        if name not in params:
            continue
        if direction is None:
            problems.append(f'@param {name}: missing direction')
            continue
        if direction not in DIRECTIONS:
            problems.append(f'@param {name}: unrecognized direction [{direction}]')
            continue
        expected = params[name].expected_directions()
        if direction not in expected:
            wanted = ' or '.join(f'[{d}]' for d in sorted(expected))
            problems.append(f'@param {name} ({params[name].type}): tagged [{direction}], expects {wanted}')
    if problems:
        raise ValueError('; '.join(problems))


def check_headers(paths, list_decls=False):
    """Return (declaration_count, [error strings]); raise ValueError on degenerate input."""
    decls = []
    for path in paths:
        if not path.is_file():
            raise ValueError(f'{path}: header not found')
        found = parse_header(path)
        if not found:
            raise ValueError(f'{path}: no function declarations found')
        decls.extend(found)
    by_name = {decl.name: decl for decl in decls}
    errors = []
    for decl in decls:
        try:
            rel = decl.path.relative_to(Path.cwd())
        except ValueError:
            rel = decl.path
        if list_decls:
            fields = []
            for param in decl.params:
                declared = '?'
                if decl.doc is not None:
                    try:
                        declared = dict(resolve_tags(decl, by_name, {decl.name})).get(param.name) or '?'
                    except ValueError:
                        pass
                fields.append(f'{param.name}={param.type}:{declared}')
            print(f'{rel}:{decl.line}\t{decl.name}\t' + '\t'.join(fields))
        try:
            check_decl(decl, by_name)
        except ValueError as error:
            errors.append(f'{rel}:{decl.line}: {decl.name}: {error}')
    return len(decls), errors


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--include-dir', default=str(REPO / 'Include'),
                        help='directory that relative header names resolve against')
    parser.add_argument('--list', action='store_true',
                        help='print every parsed declaration with its parameter types and tags')
    parser.add_argument('header', nargs='*', default=list(DEFAULT_HEADERS),
                        help='headers to check (default: the public CMSIS-NN headers)')
    args = parser.parse_args()
    include_dir = Path(args.include_dir)
    paths = [Path(h) if Path(h).is_absolute() or Path(h).exists() else include_dir / h
             for h in args.header]
    try:
        count, errors = check_headers(paths, list_decls=args.list)
    except (OSError, ValueError) as error:
        print(f'ERROR: {error}', file=sys.stderr)
        return 1
    for error in errors:
        print(f'ERROR: {error}', file=sys.stderr)
    if errors:
        print(f'ERROR: {len(errors)} of {count} declarations have incomplete or wrong @param docs',
              file=sys.stderr)
        return 1
    print(f'OK: {count} declarations in {len(paths)} headers have complete, '
          'direction-correct @param documentation')
    return 0


if __name__ == '__main__':
    sys.exit(main())
