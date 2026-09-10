#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright 2026 Ambiq <opensource@ambiq.com>
# SPDX-License-Identifier: LicenseRef-Ambiq-Apollo-SDK
#
# Fail on deferred-work markers that carry no issue reference, so that
# nothing untrackable lands; see AmbiqAI/ns-cmsis-nn#421.

set -euo pipefail

# The marker words are assembled rather than written out so that this script
# is not a hit for its own check. The match is case sensitive: folding case
# would flag ordinary text such as the Doxygen `@todo` tag and this config's
# own `todo-needs-issue` hook id.
markers="$(printf 'TO%s|FIX%s|HA%s' 'DO' 'ME' 'CK')"

(($# > 0)) || exit 0

# awk reads a `name=value` operand as a variable assignment, so a relative
# path containing `=` would be skipped silently. `./` in front makes every
# relative operand a file; awk strips the one prefix back off when reporting.
operands=()
for path in "$@"; do
  case "${path}" in
    /*) operands+=("${path}") ;;
    *) operands+=("./${path}") ;;
  esac
done

hits=$(
  awk -v markers="${markers}" '
    BEGIN {
      allowed = "(" markers ")\\((#[0-9]+|verify)\\)"
      bare = "[^A-Za-z0-9_](" markers ")[^A-Za-z0-9_]"
    }
    {
      line = " " $0 " "
      gsub(allowed, "", line)
      name = FILENAME
      sub(/^\.\//, "", name)
      pos = 1
      while (match(substr(line, pos), bare)) {
        start = pos + RSTART - 1
        word = substr(line, start + 1, RLENGTH - 2)
        printf "%s:%d: bare %s marker; use %s(#<issue>) or %s(verify)\n", \
          name, FNR, word, word, word
        # Back up one so a shared delimiter can open the next match.
        pos = start + RLENGTH - 1
      }
    }
  ' "${operands[@]}"
)

if [[ -n "${hits}" ]]; then
  printf '%s\n' "${hits}" >&2
  exit 1
fi
