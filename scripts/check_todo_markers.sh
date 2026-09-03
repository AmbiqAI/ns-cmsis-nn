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

hits=$(
  awk -v markers="${markers}" '
    BEGIN {
      allowed = "(" markers ")\\((#[0-9]+|verify)\\)"
      bare = "[^A-Za-z0-9_](" markers ")[^A-Za-z0-9_]"
    }
    {
      line = " " $0 " "
      gsub(allowed, "", line)
      if (match(line, bare)) {
        word = substr(line, RSTART + 1, RLENGTH - 2)
        printf "%s:%d: bare %s marker; use %s(#<issue>) or %s(verify)\n", \
          FILENAME, FNR, word, word, word
      }
    }
  ' "$@"
)

if [[ -n "${hits}" ]]; then
  printf '%s\n' "${hits}" >&2
  exit 1
fi
