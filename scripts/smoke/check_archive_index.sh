#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright 2026 Ambiq <opensource@ambiq.com>
# SPDX-License-Identifier: Apache-2.0
#
# Assert that a static archive carries a usable symbol index.
#
# smoke_staticlib.sh links every published archive for real, but neither
# linker path consults the index: the GNU side passes --whole-archive and
# the armlink side names every extracted member on the link line, so both
# resolve without ever asking the archive "which member defines X?". An
# index-less or stale archive passed that gate on all three toolchains
# while being unusable to a consumer linking it the normal way.
# See AmbiqAI/ns-cmsis-nn#291.
#
# Split out of smoke_staticlib.sh so the assertion can be exercised
# against a deliberately broken archive without a cross toolchain; see
# scripts/tests/test_check_archive_index.sh.
#
# Usage:
#   check_archive_index.sh --library <path/to/lib.a> \
#                          [--nm <nm-tool>] \
#                          [--require-symbol <name>]...
#
# --nm enables the cross-checks against `nm --print-armap`; without it
# only the on-disk index header is parsed.
#
# Exit codes: 2 usage, 3 missing input, 5 index absent, 6 index empty,
# 7 index contents wrong (an entry names a symbol no member defines, or a
# --require-symbol is absent), 8 index incomplete (a member exports a
# symbol the index does not name).

set -euo pipefail

LIBRARY=""
NM=""
require_syms=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --library)        LIBRARY="${2:?}"; shift 2 ;;
    --nm)             NM="${2:?}";      shift 2 ;;
    --require-symbol) require_syms+=("${2:?}"); shift 2 ;;
    *)
      echo "unknown arg: $1" >&2
      exit 2
      ;;
  esac
done

[[ -n "${LIBRARY}" ]] || { echo "--library required" >&2; exit 2; }
[[ -f "${LIBRARY}" ]] || { echo "library not found: ${LIBRARY}" >&2; exit 3; }

# ar(5) layout: an 8-byte global magic, then 60-byte member headers whose
# name field is at +0 and whose payload starts at +60. The symbol index,
# when it exists at all, is always the first member -- that is what makes
# "first member is not an index" a sound absence test.
readonly AR_MAGIC_LEN=8
readonly AR_HDR_LEN=60
readonly INDEX_PAYLOAD=$((AR_MAGIC_LEN + AR_HDR_LEN))

# The index symbol count is big-endian in the SysV/GNU format regardless
# of the target's endianness; the BSD format instead stores a
# little-endian byte size for a table of 8-byte entries.
read_bytes() { # offset length -> decimal bytes, one per line
  od -An -tu1 -N"$2" -j "$1" -v "${LIBRARY}" | tr -s ' ' '\n' | grep -v '^$'
}

be_uint() { # offset length -> value
  local v=0 b
  while read -r b; do v=$(( (v << 8) | b )); done < <(read_bytes "$1" "$2")
  echo "${v}"
}

le_uint() { # offset length -> value
  local v=0 shift_by=0 b
  while read -r b; do
    v=$(( v | (b << shift_by) ))
    shift_by=$(( shift_by + 8 ))
  done < <(read_bytes "$1" "$2")
  echo "${v}"
}

magic="$(dd if="${LIBRARY}" bs=1 count="${AR_MAGIC_LEN}" 2>/dev/null)"
if [[ "${magic}" != '!<arch>' ]]; then
  echo "not an ar archive: ${LIBRARY}" >&2
  exit 3
fi

# The name field is blank- (SysV) or NUL-padded to 16 bytes.
first_member="$(dd if="${LIBRARY}" bs=1 skip="${AR_MAGIC_LEN}" count=16 2>/dev/null | tr -d '\0')"
first_member="${first_member%%[[:space:]]*}"
index_payload="${INDEX_PAYLOAD}"

# BSD ar stores any name that does not fit the header -- which includes
# "__.SYMDEF SORTED" -- as "#1/<length>", with the name occupying the
# first <length> bytes of the payload.
if [[ "${first_member}" == '#1/'* ]]; then
  name_len="${first_member#\#1/}"
  first_member="$(dd if="${LIBRARY}" bs=1 skip="${INDEX_PAYLOAD}" count="${name_len}" 2>/dev/null | tr -d '\0')"
  first_member="${first_member%%[[:space:]]*}"
  index_payload=$(( INDEX_PAYLOAD + name_len ))
fi

case "${first_member}" in
  '/')
    index_format="sysv"
    symbol_count="$(be_uint "${index_payload}" 4)"
    ;;
  '/SYM64/')
    index_format="sysv64"
    symbol_count="$(be_uint "${index_payload}" 8)"
    ;;
  '__.SYMDEF')
    # BSD stores the byte size of a table of 8-byte ranlib entries.
    index_format="bsd"
    symbol_count=$(( $(le_uint "${index_payload}" 4) / 8 ))
    ;;
  *)
    echo "archive has no symbol index: first member is '${first_member}', expected '/', '/SYM64/' or '__.SYMDEF'" >&2
    echo "  ${LIBRARY}" >&2
    exit 5
    ;;
esac

if (( symbol_count == 0 )); then
  echo "archive symbol index is empty (${index_format}): ${LIBRARY}" >&2
  exit 6
fi

echo ">>> archive symbol index: ${index_format}, ${symbol_count} symbols"

[[ -n "${NM}" ]] || exit 0

# `nm --print-armap` prints one "<symbol> in <member>" line per index
# entry, under a header that differs between GNU nm and llvm-nm; the
# entry lines themselves are the same, and neither a symbol nor a member
# name can contain a space.
armap_syms="$("${NM}" --print-armap "${LIBRARY}" \
  | sed -n 's/^\([^[:space:]][^[:space:]]*\) in [^[:space:]][^[:space:]]*$/\1/p' \
  | LC_ALL=C sort -u)"

if [[ -z "${armap_syms}" ]]; then
  echo "index header claims ${symbol_count} symbols but ${NM} --print-armap read none" >&2
  exit 7
fi

# --extern-only drops locals, which an index is entitled to omit. Weak
# definitions stay in the set: every ar dialect indexes them, and nm
# spells them 'W'/'V' or plain 'T' depending on the object format.
nm_defined="$("${NM}" --defined-only --extern-only "${LIBRARY}")"
defined_syms="$(awk 'NF == 3 { print $3 }' <<<"${nm_defined}" | LC_ALL=C sort -u)"

# Common symbols are defined for the purpose of the stale check but must
# not be demanded of the index: GNU ar records them, the Mach-O
# __.SYMDEF does not, so requiring one would fail on macOS hosts only.
indexable_syms="$(awk 'NF == 3 && $2 != "C" { print $3 }' <<<"${nm_defined}" | LC_ALL=C sort -u)"

# A stale index is the failure the header check cannot see: the counts
# still look healthy while the entries point at symbols no member
# defines.
phantom="$(LC_ALL=C comm -23 <(printf '%s\n' "${armap_syms}") <(printf '%s\n' "${defined_syms}"))"
if [[ -n "${phantom}" ]]; then
  echo "archive symbol index names symbols no member defines:" >&2
  while IFS= read -r sym; do echo "  ${sym}"; done <<<"${phantom}" >&2
  exit 7
fi

# The mirror failure, and the one an undercounting index actually
# produces: every entry resolves, so the checks above are happy, while a
# member appended after the index was built is invisible to any consumer
# that resolves through the armap. Reported separately from the stale
# case because the two have different causes -- and the stale check runs
# first, so an entry that is both wrong and missing is named as stale.
unindexed="$(LC_ALL=C comm -13 <(printf '%s\n' "${armap_syms}") <(printf '%s\n' "${indexable_syms}"))"
if [[ -n "${unindexed}" ]]; then
  echo "archive members export symbols the index does not name:" >&2
  while IFS= read -r sym; do echo "  ${sym}"; done <<<"${unindexed}" >&2
  exit 8
fi

missing=()
for s in ${require_syms[@]+"${require_syms[@]}"}; do
  if ! grep -qxF "${s}" <<<"${armap_syms}"; then
    missing+=("${s}")
  fi
done
if (( ${#missing[@]} > 0 )); then
  echo "archive symbol index is missing expected symbols:" >&2
  printf '  %s\n' "${missing[@]}" >&2
  exit 7
fi
