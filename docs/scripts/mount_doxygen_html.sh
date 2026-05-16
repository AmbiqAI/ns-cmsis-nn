#!/usr/bin/env bash
set -euo pipefail

source_dir="${1:-Documentation/html}"
site_dir="${2:-site}"
api_path="${3:-api}"

if [[ ! -f "${source_dir}/index.html" ]]; then
  echo "Doxygen HTML index not found: ${source_dir}/index.html" >&2
  echo "Generate it first with ./Documentation/Doxygen/gen_doc.sh -s." >&2
  exit 1
fi

if [[ ! -d "${site_dir}" ]]; then
  echo "MkDocs site directory not found: ${site_dir}" >&2
  echo "Run mkdocs build before mounting Doxygen HTML." >&2
  exit 1
fi

target_dir="${site_dir%/}/${api_path#/}"
rm -rf "${target_dir}"
mkdir -p "${target_dir}"
cp -a "${source_dir%/}/." "${target_dir}/"

printf 'Mounted Doxygen HTML: %s -> %s\n' "${source_dir}" "${target_dir}"
