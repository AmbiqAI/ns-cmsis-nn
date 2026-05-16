#!/usr/bin/env bash
set -euo pipefail

site_dir="site"
doxygen_html_dir="Documentation/html"
api_path="api"
doxygen_version="${DOXYGEN_VERSION:-1.9.6}"
install_doxygen=0
generate_doxygen=1
mkdocs_cmd="${MKDOCS:-mkdocs}"

usage() {
  cat <<'USAGE'
Usage: build_combined_docs.sh [options]

Build the combined heliaCORE documentation artifact:
  1. Generate Doxygen HTML/XML.
  2. Build MkDocs with strict validation.
  3. Mount generated Doxygen HTML under site/api/.
  4. Verify the mounted API entrypoint exists.

Options:
  --install-doxygen          Download Doxygen 1.9.6 to /tmp if it is not on PATH.
  --doxygen-version VERSION  Doxygen version to require/install (default: 1.9.6).
  --skip-doxygen             Reuse an existing Documentation/html/ output.
  --site-dir DIR             MkDocs output directory (default: site).
  --doxygen-html-dir DIR     Generated Doxygen HTML directory (default: Documentation/html).
  --api-path PATH            Mount path inside the MkDocs site (default: api).
  -h, --help                 Show this help.

Environment:
  MKDOCS                     MkDocs executable (default: mkdocs).
  DOXYGEN_URL                Override Doxygen tarball URL.
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --install-doxygen)
      install_doxygen=1
      ;;
    --doxygen-version)
      doxygen_version="$2"
      shift
      ;;
    --skip-doxygen)
      generate_doxygen=0
      ;;
    --site-dir)
      site_dir="$2"
      shift
      ;;
    --doxygen-html-dir)
      doxygen_html_dir="$2"
      shift
      ;;
    --api-path)
      api_path="$2"
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
  shift
done

log() {
  printf '\n==> %s\n' "$*"
}

ensure_doxygen() {
  if command -v doxygen >/dev/null 2>&1; then
    actual_version="$(doxygen --version)"
    if [[ "${actual_version}" == "${doxygen_version}"* ]]; then
      log "Using Doxygen ${actual_version} from PATH"
      return
    fi
    echo "Found Doxygen ${actual_version}, but ${doxygen_version} is required." >&2
    if [[ ${install_doxygen} -eq 0 ]]; then
      echo "Re-run with --install-doxygen or put Doxygen ${doxygen_version} on PATH." >&2
      exit 1
    fi
  elif [[ ${install_doxygen} -eq 0 ]]; then
    echo "Doxygen ${doxygen_version} is required but was not found on PATH." >&2
    echo "Re-run with --install-doxygen or install Doxygen ${doxygen_version}." >&2
    exit 1
  fi

  install_root="${DOXYGEN_INSTALL_ROOT:-/tmp}"
  doxygen_dir="${install_root%/}/doxygen-${doxygen_version}"
  doxygen_bin="${doxygen_dir}/bin/doxygen"

  if [[ ! -x "${doxygen_bin}" ]]; then
    log "Downloading Doxygen ${doxygen_version} to ${install_root}"
    archive="${install_root%/}/doxygen-${doxygen_version}.linux.bin.tar.gz"
    url="${DOXYGEN_URL:-https://www.doxygen.nl/files/doxygen-${doxygen_version}.linux.bin.tar.gz}"

    if command -v curl >/dev/null 2>&1; then
      curl -fsSL "${url}" -o "${archive}"
    elif command -v wget >/dev/null 2>&1; then
      wget -O "${archive}" "${url}"
    else
      echo "Neither curl nor wget is available to download Doxygen." >&2
      exit 1
    fi

    tar -C "${install_root}" -xzf "${archive}"
  fi

  export PATH="${doxygen_dir}/bin:${PATH}"
  actual_version="$(doxygen --version)"
  if [[ "${actual_version}" != "${doxygen_version}"* ]]; then
    echo "Installed Doxygen ${actual_version}, expected ${doxygen_version}." >&2
    exit 1
  fi
  log "Using Doxygen ${actual_version} from ${doxygen_dir}"
}

if [[ ${generate_doxygen} -eq 1 ]]; then
  ensure_doxygen
  log "Generating Doxygen API reference"
  ./Documentation/Doxygen/gen_doc.sh -s
else
  log "Skipping Doxygen generation; reusing ${doxygen_html_dir}"
fi

if [[ ! -f "${doxygen_html_dir}/index.html" ]]; then
  echo "Doxygen HTML index not found: ${doxygen_html_dir}/index.html" >&2
  exit 1
fi

log "Building MkDocs site"
"${mkdocs_cmd}" build --strict

log "Mounting Doxygen HTML under ${site_dir}/${api_path}"
bash scripts/docs/mount_doxygen_html.sh "${doxygen_html_dir}" "${site_dir}" "${api_path}"

api_index="${site_dir%/}/${api_path#/}/index.html"
if [[ ! -f "${api_index}" ]]; then
  echo "Mounted API index not found: ${api_index}" >&2
  exit 1
fi

log "Combined docs built successfully"
printf 'MkDocs site: %s\n' "${site_dir}"
printf 'Doxygen API: %s\n' "${api_index}"
