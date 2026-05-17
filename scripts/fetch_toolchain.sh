#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright 2026 Ambiq <opensource@ambiq.com>
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

MANIFEST=""
PLATFORM="linux-x86_64"
OUTDIR=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --manifest) MANIFEST="${2:?}"; shift 2 ;;
    --platform) PLATFORM="${2:?}"; shift 2 ;;
    --outdir)   OUTDIR="${2:?}";   shift 2 ;;
    -h|--help)
      sed -n '2,/^$/p' "$0" | sed 's/^# \{0,1\}//'
      exit 0
      ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

[[ -n "${MANIFEST}" ]] || { echo "--manifest required" >&2; exit 2; }
[[ -n "${OUTDIR}"   ]] || { echo "--outdir required" >&2; exit 2; }
[[ -f "${MANIFEST}" ]] || { echo "manifest not found: ${MANIFEST}" >&2; exit 2; }

command -v python3 >/dev/null || { echo "python3 not on PATH" >&2; exit 3; }
command -v curl    >/dev/null || { echo "curl not on PATH" >&2; exit 3; }
command -v sha256sum >/dev/null || { echo "sha256sum not on PATH" >&2; exit 3; }

mapfile -t fields < <(python3 - "${MANIFEST}" "${PLATFORM}" <<'PY'
import json
import sys

manifest_path, platform = sys.argv[1:3]
with open(manifest_path, "r", encoding="utf-8") as handle:
    manifest = json.load(handle)
try:
    entry = manifest["platforms"][platform]
except KeyError as exc:
    raise SystemExit(f"manifest {manifest_path} has no platform {platform!r}") from exc
for value in (
    manifest["id"],
    manifest["version"],
    manifest.get("compiler_id", ""),
    entry["archive"],
    entry["url"],
    entry["sha256"],
    entry["probe"],
):
    print(value)
PY
)

TOOLCHAIN_ID="${fields[0]}"
TOOLCHAIN_VERSION="${fields[1]}"
TOOLCHAIN_COMPILER_ID="${fields[2]}"
ARCHIVE="${fields[3]}"
URL="${fields[4]}"
SHA256="${fields[5]}"
PROBE="${fields[6]}"

mkdir -p "${OUTDIR}"
OUTDIR="$(cd "${OUTDIR}" && pwd)"
download="${OUTDIR}/${ARCHIVE}"
extract_dir="${OUTDIR}/${TOOLCHAIN_ID}-${TOOLCHAIN_VERSION}"

if [[ ! -f "${download}" ]]; then
  curl -fsSL "${URL}" -o "${download}"
fi
printf '%s  %s\n' "${SHA256}" "${download}" | sha256sum -c -

rm -rf "${extract_dir}"
mkdir -p "${extract_dir}"
case "${ARCHIVE}" in
  *.tar.gz|*.tgz) tar -xzf "${download}" -C "${extract_dir}" ;;
  *.tar.xz)       tar -xJf "${download}" -C "${extract_dir}" ;;
  *.zip)          unzip -q "${download}" -d "${extract_dir}" ;;
  *) echo "unsupported archive type: ${ARCHIVE}" >&2; exit 2 ;;
esac

probe_path=""
while IFS= read -r candidate; do
  probe_path="${candidate}"
  break
done < <(find "${extract_dir}" -path "*/${PROBE}" -type f | sort)

if [[ -z "${probe_path}" ]]; then
  echo "could not find ${PROBE} under ${extract_dir}" >&2
  exit 4
fi

toolchain_root="${probe_path%/${PROBE}}"
toolchain_root="$(cd "${toolchain_root}" && pwd)"

echo "${TOOLCHAIN_ID} ${TOOLCHAIN_VERSION} staged at ${toolchain_root}"

if [[ -n "${GITHUB_OUTPUT:-}" ]]; then
  {
    echo "toolchain_id=${TOOLCHAIN_ID}"
    echo "toolchain_version=${TOOLCHAIN_VERSION}"
    echo "toolchain_compiler_id=${TOOLCHAIN_COMPILER_ID}"
    echo "toolchain_root=${toolchain_root}"
  } >> "${GITHUB_OUTPUT}"
fi
