#!/usr/bin/env python3
# SPDX-License-Identifier: LicenseRef-Ambiq-Apollo-SDK
"""Check that every public header function has complete, direction-correct @param docs.

The doc block above each declaration in Include/ is the machine-readable half of the
kernel contract: the C types cannot say whether a mutable pointer is written only or
read first, so the @param[out] / @param[in,out] tags have to be trustworthy. The docs
build sets EXTRACT_ALL = YES, which makes doxygen disable WARN_NO_PARAMDOC, and its
WARN_IF_INCOMPLETE_DOC only fires once a block already carries at least one tag, so a
brief-only block passes the docs build silently; this check reads the headers directly,
needs no doxygen and no build, and fails on any of: a declaration with no doc block, a
missing or extra @param, a tag without a direction, a direction that contradicts the
const-ness of the parameter type, or a @copydoc whose target is unknown, cyclic, or
itself incomplete. See AmbiqAI/ns-cmsis-nn#526.

The scanner is deliberately fail-loud: anything it cannot parse (a parameter shape it
does not know, an inline body whose braces do not balance) is reported as an error
against the declaration rather than skipped, because a silently dropped declaration
would pass the gate forever and vanish from the --list contract as well.
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
# Aggregate bodies are cut at their opening brace before this is applied, so only a typedef
# can still look like a function here; `struct x *f(...)` and `enum e f(...)` are functions.
NOT_A_FUNCTION_RE = re.compile(r'^typedef\b')
FUNCTION_POINTER_RE = re.compile(r'^(?P<ret>.+?)\(\s*\*\s*(?P<name>[A-Za-z_]\w*)\s*\)\s*\(.*\)$')
PARAM_TAG_RE = re.compile(r'[@\\]param\b\s*(?:\[(?P<dir>[^\]]*)\])?\s*(?P<name>[A-Za-z_]\w*)')
# Example code inside a block is rendered, not interpreted, so a tag in it documents nothing.
VERBATIM_RE = re.compile(r'[@\\](code|verbatim)\b.*?(?:[@\\]end(?:code|verbatim)\b|\Z)', re.S)
# A block that opens or closes a doxygen group documents the group, not the next declaration.
GROUP_MARKER_RE = re.compile(r'[@\\](\{|\}|addtogroup\b|defgroup\b)')
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
        self.is_function_pointer = False
        extent = re.search(r'(\s*\[[^\]]*\])+$', text)
        self.extent = extent.group(0).strip() if extent else ''
        self.is_array = bool(self.extent)
        if self.is_array:
            text = text[:extent.start()].rstrip()
        pointer = FUNCTION_POINTER_RE.match(text)
        if pointer:
            self.is_function_pointer = True
            self.name = pointer.group('name')
            self.type = text
            return
        match = re.search(r'([A-Za-z_]\w*)$', text)
        if match is None or match.group(1) == text or '(' in text or ')' in text:
            raise ValueError(f'cannot parse parameter {text!r}')
        self.name = match.group(1)
        self.type = text[:match.start()].strip()
        self.type = re.sub(r'\s*\*\s*', ' * ', self.type).replace('  ', ' ').strip()

    @property
    def spelling(self):
        """The type as declared, including an array extent, for listings and messages."""
        return f'{self.type} {self.extent}'.strip()

    def expected_directions(self):
        # A function pointer is a value the callee calls, never a buffer it writes.
        if self.is_function_pointer:
            return {'in'}
        # A const on the pointer itself does not change what the callee may write through it.
        normalized = re.sub(r'\s*\bconst$', '', self.type.replace(' * ', ' *').strip())
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
    def __init__(self, path, line, name, params, doc, is_static, error=None):
        self.path, self.line, self.name, self.params, self.doc = path, line, name, params, doc
        self.is_static, self.error = is_static, error


def is_doc_block_start(text, i):
    """A /** that opens its own line starts a doc block; /**/ and trailing /**< member docs do not."""
    if not text.startswith('/**', i) or text.startswith('/**/', i) or text.startswith('/**<', i):
        return False
    return not text[text.rfind('\n', 0, i) + 1:i].strip()


def blank_non_doc(text):
    """Replace // and /* */ comments, string and character literals with spaces.

    Doc blocks are copied through verbatim as a unit so that a URL or a quote inside one can
    never be mistaken for the start of a comment or literal in the code that follows.
    """
    out, i, n = [], 0, len(text)
    while i < n:
        two = text[i:i + 2]
        marker = False
        if two == '/*':
            end = text.find('*/', i + 2)
            end = n if end < 0 else end + 2
            if is_doc_block_start(text, i):
                out.append(text[i:end])
                i = end
                continue
            marker = True
        elif two == '//':
            end = text.find('\n', i)
            end = n if end < 0 else end
            marker = True
        elif text[i] in '"\'':
            quote = text[i]
            end = i + 1
            while end < n and text[end] != quote and text[end] != '\n':
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


def strip_attributes(stmt):
    """Remove __attribute__((...)) so the declarator, not the attribute, names the function."""
    while True:
        start = stmt.find('__attribute__')
        if start < 0:
            return stmt
        open_paren = stmt.find('(', start)
        if open_paren < 0:
            return stmt
        depth = 0
        for k in range(open_paren, len(stmt)):
            depth += (stmt[k] == '(') - (stmt[k] == ')')
            if depth == 0:
                break
        else:
            return stmt
        stmt = stmt[:start] + ' ' + stmt[k + 1:]


def looks_like_call(stmt):
    """True when a statement has an identifier followed by a parameter list outside any
    array extent, and is not an initialised variable."""
    if '=' in stmt:
        return False
    return re.search(r'\b[A-Za-z_]\w*\s*\(', re.sub(r'\[[^\]]*\]', '', stmt)) is not None


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


class BodySkipper:
    """Track brace depth through an inline body, counting only the first branch of each
    preprocessor conditional opened inside it: the branches share one closing brace, so
    counting every branch would leave the depth stuck above zero and drop every later
    declaration in the file."""

    def __init__(self, opened_at, tail):
        self.opened_at = opened_at
        self.depth = tail.count('{') - tail.count('}')
        self.branches = []

    def feed(self, s):
        directive = CONDITIONAL_RE.match(s)
        if directive:
            keyword = directive.group(1)
            if keyword in ('if', 'ifdef', 'ifndef'):
                self.branches.append(False)
            elif keyword in ('elif', 'else') and self.branches:
                self.branches[-1] = True
            elif keyword == 'endif' and self.branches:
                self.branches.pop()
        elif not s.startswith('#') and not any(self.branches):
            self.depth += s.count('{') - s.count('}')
        return self.depth <= 0


def parse_header(path):
    """Return every function declaration in the header with its preceding doc block."""
    text = blank_non_doc(path.read_text(encoding='utf-8'))
    decls, pending_doc, body, extern_blocks = [], None, None, []
    # One entry per open file-scope conditional: the doc block a declaration in an earlier
    # branch consumed, so the same block can document its #else/#elif twin.
    conditionals, doc_depth = [], 0
    lines = list(logical_lines(text))
    i = 0
    while i < len(lines):
        lineno, line = lines[i]
        s = line.strip()
        i += 1
        # A line can hold several tokens of interest (`/* x */ int f(void);`), so each
        # branch that consumes only part of the line loops on the remainder.
        while s:
            if body is not None:
                if body.feed(s):
                    if body.depth < 0:
                        raise ValueError(f'{path}:{body.opened_at}: braces in the body opened '
                                         'here do not balance')
                    body = None
                break
            if s.startswith('#'):
                directive = CONDITIONAL_RE.match(s)
                if not directive:
                    pending_doc = None
                elif directive.group(1) in ('if', 'ifdef', 'ifndef'):
                    conditionals.append(None)
                elif directive.group(1) in ('elif', 'else'):
                    if conditionals and conditionals[-1] is not None:
                        pending_doc, doc_depth = conditionals[-1], len(conditionals) - 1
                elif conditionals:
                    conditionals.pop()
                break
            if s.startswith(COMMENT_MARKER):
                pending_doc = None
                s = s[len(COMMENT_MARKER):].strip()
                continue
            if s.startswith('/**'):
                block, current = [], s
                while '*/' not in current and i < len(lines):
                    block.append(current)
                    current = lines[i][1].strip()
                    i += 1
                head, terminator, rest = current.partition('*/')
                block.append(head + terminator)
                doc = '\n'.join(block)
                pending_doc = None if GROUP_MARKER_RE.search(doc) else (lineno, doc)
                doc_depth = len(conditionals)
                s = rest.strip()
                continue
            # The only bare braces legal at file scope are an extern "C" block's; an
            # unmatched one means a body above was miscounted, so refuse the file.
            if re.match(r'^extern\s*\{$', s):
                extern_blocks.append(lineno)
                break
            if s == '}':
                if not extern_blocks:
                    raise ValueError(f'{path}:{lineno}: unexpected closing brace at file scope')
                extern_blocks.pop()
                break
            stmt, paren = [], 0
            while True:
                for char in s:
                    paren += (char == '(') - (char == ')')
                stmt.append(s)
                if paren == 0 and (s.endswith(';') or '{' in s):
                    break
                if i >= len(lines):
                    break
                s = lines[i][1].strip()
                i += 1
            joined = ' '.join(stmt)
            if '{' in joined:
                cut = joined.index('{')
                body = BodySkipper(lineno, joined[cut:])
                if body.depth < 0:
                    raise ValueError(f'{path}:{lineno}: braces in the body opened here do not balance')
                if body.depth == 0:
                    body = None
                joined = joined[:cut]
            joined = strip_attributes(joined.rstrip('; ').strip())
            match = DECL_RE.match(joined)
            if NOT_A_FUNCTION_RE.match(joined):
                pass
            elif match:
                params, error = [], None
                try:
                    params = split_params(match.group('params'))
                except ValueError as failure:
                    error = str(failure)
                decls.append(Decl(path, lineno, match.group('name'), params, pending_doc,
                                  bool(STATIC_RE.search(match.group('ret'))), error))
            elif looks_like_call(joined):
                # Something with a parameter list that the grammar above does not cover, such
                # as a macro after the closing parenthesis: report it rather than lose it.
                name = re.search(r'([A-Za-z_]\w*)\s*\(', joined).group(1)
                decls.append(Decl(path, lineno, name, [], pending_doc, False,
                                  f'unrecognized declaration {joined!r}'))
            if pending_doc is not None:
                for level in range(doc_depth, len(conditionals)):
                    conditionals[level] = pending_doc
            pending_doc = None
            break
    if body is not None:
        raise ValueError(f'{path}:{body.opened_at}: the body opened here is never closed')
    if extern_blocks:
        raise ValueError(f'{path}:{extern_blocks[-1]}: the extern block opened here is never closed')
    return decls


def doc_tags(doc):
    """Return (copydoc_target, [(name, direction)]) from a doc block."""
    doc = VERBATIM_RE.sub(' ', doc)
    copydocs = COPYDOC_RE.findall(doc)
    if len(copydocs) > 1:
        raise ValueError('multiple copy directives in one block: ' + ', '.join(copydocs))
    tags = [(m.group('name'), None if m.group('dir') is None else m.group('dir').replace(' ', ''))
            for m in PARAM_TAG_RE.finditer(doc)]
    return (copydocs[0] if copydocs else None), tags


def resolve_tags(decl, by_name, visiting):
    """Follow @copydoc chains and return the accumulated tag list, raising on bad chains."""
    if decl.doc is None:
        raise ValueError('no doc block immediately before the declaration')
    target, tags = doc_tags(decl.doc[1])
    if target is None:
        return tags
    if target in visiting:
        raise ValueError(f'@copydoc cycle through {target}')
    if target not in by_name:
        raise ValueError(f'unknown @copydoc target {target}')
    if by_name[target].is_static:
        raise ValueError(f'@copydoc target {target} is a static function, which doxygen cannot resolve; '
                         'write the tags inline')
    # Doxygen merges the copied tags with the block's own, so a wrapper may document only
    # the parameters its base lacks; an overlap shows up as a duplicate tag.
    return resolve_tags(by_name[target], by_name, visiting | {decl.name}) + tags


def check_decl(decl, by_name):
    if decl.error:
        raise ValueError(decl.error)
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
            problems.append(f'@param {name} ({params[name].spelling}): tagged [{direction}], expects {wanted}')
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
            fields = [f'!{decl.error}'] if decl.error else []
            for param in decl.params:
                declared = '?'
                if decl.doc is not None:
                    try:
                        declared = dict(resolve_tags(decl, by_name, {decl.name})).get(param.name) or '?'
                    except ValueError:
                        pass
                fields.append(f'{param.name}={param.spelling}:{declared}')
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
