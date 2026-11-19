#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright 2026 Ambiq <opensource@ambiq.com>
# SPDX-License-Identifier: Apache-2.0
#
# Create, update or close one rolling status issue for a scheduled workflow.
# A red run comments on the open issue (or opens it); the first green run
# closes it. One issue per workflow rather than one per red run.
#
#   rolling_status_issue.sh --title <exact title> --label <label> --label-description <text> \
#                           --what <"Nightly run"|...> --run-url <url> <callee>=<result> ...
#
# Needs GH_TOKEN with issues: write and REPO=<owner/name>.

set -euo pipefail

TITLE=""
LABEL=""
LABEL_DESCRIPTION=""
WHAT=""
RUN_URL=""
while [[ $# -gt 0 && "$1" == --* ]]; do
  case "$1" in
    --title) TITLE="${2:?}"; shift 2 ;;
    --label) LABEL="${2:?}"; shift 2 ;;
    --label-description) LABEL_DESCRIPTION="${2:?}"; shift 2 ;;
    --what) WHAT="${2:?}"; shift 2 ;;
    --run-url) RUN_URL="${2:?}"; shift 2 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done
[[ -n "$TITLE" && -n "$LABEL" && -n "$WHAT" && -n "$RUN_URL" && $# -gt 0 ]] || {
  echo "usage: $0 --title T --label L --label-description D --what W --run-url U callee=result ..." >&2
  exit 2
}
: "${REPO:?REPO must be set}"
today="$(date -u +%Y-%m-%d)"

# The label is part of the search guard below, so it must exist before
# `gh issue create --label` runs (gh does not create missing labels). Create
# only when missing: --force would re-assert color and description on every
# run, silently reverting manual edits.
if ! gh label list -R "$REPO" --limit 200 --json name --jq '.[].name' | grep -qxF "$LABEL"; then
  gh label create "$LABEL" -R "$REPO" --color B60205 --description "$LABEL_DESCRIPTION"
fi

# Guard: exact-title match AND the label, open issues only. `gh issue list
# --search` is fuzzy, so filter exactly.
issue_number="$(gh issue list -R "$REPO" --state open --label "$LABEL" --json number,title |
  python3 -c 'import json, sys; t = sys.argv[1]; print(next((str(i["number"]) for i in json.load(sys.stdin) if i["title"] == t), ""))' "$TITLE")"

failed_callees=''
for pair in "$@"; do
  name="${pair%%=*}"
  result="${pair#*=}"
  [[ "$result" == success ]] || failed_callees+="- ${name}: ${result}"$'\n'
done

if [[ -n "$failed_callees" ]]; then
  body="${WHAT} on \`main\` failed on ${today}.

Run: ${RUN_URL}

Failing callees:
${failed_callees}
This is the rolling status issue: subsequent red runs comment here, and the first green run closes it."
  if [[ -n "$issue_number" ]]; then
    echo "Red; commenting on open rolling issue #${issue_number}"
    gh issue comment "$issue_number" -R "$REPO" --body "$body"
  else
    echo 'Red; no open rolling issue -- creating one'
    gh issue create -R "$REPO" --title "$TITLE" --label "$LABEL" --body "$body"
  fi
  exit 0
fi

if [[ -n "$issue_number" ]]; then
  echo "Green; closing rolling issue #${issue_number}"
  gh issue close "$issue_number" -R "$REPO" --comment "${WHAT} on \`main\` is green again as of ${today}: ${RUN_URL}"
else
  echo 'Green; no rolling issue open -- nothing to do'
fi
