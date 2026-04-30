#!/usr/bin/env bash

set -euo pipefail

usage() {
  cat <<'EOF'
Publish ns-cmsis-nn release assets to a GitHub Release in a distribution repository.

Usage:
  publish_helia_core_release.sh --repo <owner/name> --tag <tag> [--title <title>]
                                [--draft]
                                --asset <path> [--asset <path> ...]
EOF
}

REPO=""
TAG=""
TITLE=""
CREATE_AS_DRAFT=0
ASSETS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --repo)
      REPO="${2:?missing value for --repo}"
      shift 2
      ;;
    --tag)
      TAG="${2:?missing value for --tag}"
      shift 2
      ;;
    --title)
      TITLE="${2:?missing value for --title}"
      shift 2
      ;;
    --draft)
      CREATE_AS_DRAFT=1
      shift
      ;;
    --asset)
      ASSETS+=("${2:?missing value for --asset}")
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 2
      ;;
  esac
done

[[ -n "${GH_TOKEN:-}" ]] || { echo "GH_TOKEN must be set" >&2; exit 3; }
[[ -n "${REPO}" ]] || { echo "Missing --repo" >&2; exit 2; }
[[ -n "${TAG}" ]] || { echo "Missing --tag" >&2; exit 2; }
[[ ${#ASSETS[@]} -gt 0 ]] || { echo "At least one --asset is required" >&2; exit 2; }

if [[ -z "${TITLE}" ]]; then
  TITLE="${TAG}"
fi

for asset in "${ASSETS[@]}"; do
  [[ -f "${asset}" ]] || { echo "Release asset not found: ${asset}" >&2; exit 4; }
done

created_release=0
was_draft=""
release_view_json="$(mktemp)"
trap 'rm -f "${release_view_json}"' EXIT

if gh release view "${TAG}" --repo "${REPO}" --json isDraft >"${release_view_json}" 2>/dev/null; then
  was_draft="$(python3 - "${release_view_json}" <<'PY'
import json
import sys
from pathlib import Path
payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
print("true" if payload.get("isDraft") else "false")
PY
)"
else
  create_args=(release create "${TAG}" --repo "${REPO}" --title "${TITLE}" --notes "")
  create_args+=(--draft)
  gh "${create_args[@]}"
  created_release=1
  was_draft="true"
fi

expected_names=()
for asset in "${ASSETS[@]}"; do
  expected_names+=("$(basename "${asset}")")
  gh release upload "${TAG}" "${asset}" --repo "${REPO}" --clobber
done

actual_names="$(
  gh release view "${TAG}" --repo "${REPO}" --json assets -q '.assets[].name'
)"

ACTUAL_NAMES="${actual_names}" python3 - "${expected_names[@]}" <<'PY'
import os
import sys

actual = {line.strip() for line in os.environ["ACTUAL_NAMES"].splitlines() if line.strip()}
expected = set(sys.argv[1:])
missing = sorted(expected - actual)
if missing:
    raise SystemExit(f"Release is missing expected assets: {', '.join(missing)}")
PY

if [[ "${CREATE_AS_DRAFT}" -eq 0 ]]; then
  gh release edit "${TAG}" --repo "${REPO}" --draft=false --title "${TITLE}"
elif [[ "${created_release}" -eq 0 && "${was_draft}" == "false" ]]; then
  echo "Release ${TAG} already published in ${REPO}; assets were updated in place."
else
  echo "Release ${TAG} left as draft in ${REPO}."
fi

echo "Published ${#ASSETS[@]} assets to ${REPO} release ${TAG}."
