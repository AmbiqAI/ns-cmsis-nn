#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright 2026 Ambiq <opensource@ambiq.com>
# SPDX-License-Identifier: Apache-2.0
#
# Negative test for scripts/smoke/check_archive_index.sh.
#
# The check exists because a broken archive symbol index links clean
# (AmbiqAI/ns-cmsis-nn#291), so nothing in the smoke gate would fail if
# the check itself silently stopped asserting. This drives it against
# archives that are deliberately broken in each of the ways it claims to
# catch. Host cc/ar/nm only: the assertion is on archive structure, not
# on the target, so it needs no cross toolchain.
#
# Usage: scripts/tests/test_check_archive_index.sh
# Override the tools with AR=, NM=, CC= if the defaults are unsuitable.

set -euo pipefail

CC="${CC:-cc}"
AR="${AR:-ar}"
NM="${NM:-nm}"

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
check="${repo_root}/smoke/check_archive_index.sh"
[[ -f "${check}" ]] || { echo "checker missing: ${check}" >&2; exit 3; }

for t in "${CC}" "${AR}" "${NM}"; do
  command -v "${t}" >/dev/null || { echo "${t} not on PATH" >&2; exit 3; }
done

work="$(mktemp -d)"
trap 'rm -rf "${work}"' EXIT
cd "${work}"

printf 'int ns_probe_a(void) { return 1; }\n' > a.c
printf 'int ns_probe_b(void) { return 2; }\n' > b.c

# Weak and common definitions are where the completeness check is most
# likely to fire spuriously: a weak one has to be indexed everywhere,
# while GNU ar records a common one and the Mach-O __.SYMDEF does not.
# Both belong in the passing fixture so either mistake fails a case.
cat > c.c <<'EOC'
int ns_probe_common;
__attribute__((weak)) int ns_probe_weak(void) { return 3; }
EOC
"${CC}" -c a.c b.c
# Without -fcommon a tentative definition lands in bss instead; compilers
# that reject the flag still emit a definition, just not a common one.
"${CC}" -fcommon -c c.c 2>/dev/null || "${CC}" -c c.c

failures=0
expect() { # <expected-exit> <name> <args...>
  local want="$1" name="$2"; shift 2
  local got=0 out
  out="$("${check}" "$@" 2>&1)" || got=$?
  if (( got == want )); then
    echo "ok   ${name} (exit ${got})"
    [[ "${want}" -eq 0 ]] && echo "     ${out}"
  else
    echo "FAIL ${name}: expected exit ${want}, got ${got}" >&2
    printf '     %s\n' "${out}" >&2
    failures=$(( failures + 1 ))
  fi
  return 0
}

"${AR}" rc good.a a.o b.o c.o

# Mach-O hosts prefix C symbols with an underscore, so read the names the
# index actually carries instead of assuming a spelling.
index_syms="$("${NM}" --print-armap good.a \
  | sed -n 's/^\([^[:space:]][^[:space:]]*\) in [^[:space:]][^[:space:]]*$/\1/p')"
probe_a="$(grep -E '^_?ns_probe_a$' <<<"${index_syms}" | head -1 || true)"
first_sym="$(head -1 <<<"${index_syms}")"
[[ -n "${probe_a}" && -n "${first_sym}" ]] \
  || { echo "could not read index entries from good.a" >&2; exit 3; }

expect 0 "indexed archive passes" \
  --library good.a --nm "${NM}" --require-symbol "${probe_a}"

expect 7 "absent symbol is reported" \
  --library good.a --nm "${NM}" --require-symbol ns_probe_absent

# `S` suppresses the symbol table, so the first member is a plain object.
# This is the shape a packaging change could plausibly produce.
"${AR}" rcS nosym.a a.o b.o
expect 5 "archive without an index fails" --library nosym.a --nm "${NM}"

cp good.a empty.a
cp good.a stale.a
python3 - "${first_sym}" <<'PY'
import sys

name = sys.argv[1].encode() + b"\0"


def index_payload(blob):
    """Offset of the first member's content, past any BSD long name."""
    start = 8 + 60
    member = blob[8:24].split(b"\0")[0].split()[0]
    if member.startswith(b"#1/"):
        start += int(member[3:])
    return start


with open("empty.a", "r+b") as f:
    blob = f.read()
    # The SysV symbol count and the BSD table byte size both sit in the
    # first four bytes of the payload, so one patch empties either.
    f.seek(index_payload(blob))
    f.write(b"\0\0\0\0")

with open("stale.a", "r+b") as f:
    blob = f.read()
    end = 8 + 60 + int(blob[56:66].decode().strip())
    f.seek(blob.index(name, index_payload(blob), end))
    f.write(b"Z")
PY

expect 6 "archive with an empty index fails" --library empty.a --nm "${NM}"

# A stale index: counts and offsets stay healthy while an entry names a
# symbol no member defines. The header check alone would pass this. It is
# also short an entry now, so this case pins the stale check ahead of the
# completeness one.
expect 7 "archive with a stale index fails" --library stale.a --nm "${NM}"

# An undercounting index: every entry it does carry resolves, so only the
# completeness check sees it. Spliced by hand rather than with `ar qS`,
# whose treatment of an already-present symbol table differs between GNU
# ar and llvm-ar.
"${AR}" rc undercount.a a.o
"${AR}" rc tail.a b.o
python3 - <<'SPLICE'
# (start, end) of the final ar member, header and padding included.
def last_member(blob):
    pos, last = 8, None
    while pos + 60 <= len(blob):
        size = int(blob[pos + 48:pos + 58].decode().strip())
        end = pos + 60 + size + (size % 2)
        last = (pos, end)
        pos = end
    return last


with open("tail.a", "rb") as f:
    blob = f.read()
start, end = last_member(blob)
with open("undercount.a", "ab") as f:
    f.write(blob[start:end])
SPLICE

expect 8 "archive whose index misses a member fails" \
  --library undercount.a --nm "${NM}"

if (( failures > 0 )); then
  echo "${failures} case(s) failed" >&2
  exit 1
fi
echo "all archive-index cases passed"
