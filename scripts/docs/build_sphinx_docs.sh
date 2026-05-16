#!/usr/bin/env bash
set -euo pipefail

site_dir="site"
sphinx_src="docs"
doxygen_xml_dir="Documentation/xml"
doxygen_version="${DOXYGEN_VERSION:-1.9.6}"
install_doxygen=0
generate_doxygen=1
sphinx_build_cmd="${SPHINXBUILD:-sphinx-build}"

usage() {
  cat <<'USAGE'
Usage: build_sphinx_docs.sh [options]

Build the heliaCORE documentation artifact:
  1. Generate Doxygen HTML/XML from the C headers and sources.
  2. Build Sphinx with Breathe/Exhale using the generated Doxygen XML.
  3. Verify the generated Sphinx API entrypoint exists.

Options:
  --install-doxygen          Download Doxygen 1.9.6 to /tmp if it is not on PATH.
  --doxygen-version VERSION  Doxygen version to require/install (default: 1.9.6).
  --skip-doxygen             Reuse an existing Documentation/xml/ output.
  --site-dir DIR             Sphinx HTML output directory (default: site).
  --sphinx-src DIR           Sphinx source directory (default: docs).
  --doxygen-xml-dir DIR      Generated Doxygen XML directory (default: Documentation/xml).
  -h, --help                 Show this help.

Environment:
  SPHINXBUILD                sphinx-build executable (default: sphinx-build).
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
    --sphinx-src)
      sphinx_src="$2"
      shift
      ;;
    --doxygen-xml-dir)
      doxygen_xml_dir="$2"
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
  log "Generating Doxygen API XML"
  ./Documentation/Doxygen/gen_doc.sh -s
else
  log "Skipping Doxygen generation; reusing ${doxygen_xml_dir}"
fi

if [[ ! -f "${doxygen_xml_dir}/index.xml" ]]; then
  echo "Doxygen XML index not found: ${doxygen_xml_dir}/index.xml" >&2
  exit 1
fi

mkdir -p "${sphinx_src}/api"
find "${sphinx_src}/api" -mindepth 1 ! -name .gitignore -exec rm -rf {} +

log "Building Sphinx site"
"${sphinx_build_cmd}" -b html -W --keep-going "${sphinx_src}" "${site_dir}"

api_index="${site_dir%/}/api/library_root.html"
if [[ ! -f "${api_index}" ]]; then
  echo "Generated Sphinx API index not found: ${api_index}" >&2
  exit 1
fi

log "Sphinx docs built successfully"
printf 'Sphinx site: %s\n' "${site_dir}"
printf 'Generated API: %s\n' "${api_index}"
