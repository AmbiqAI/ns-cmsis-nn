FROM python:3.10-bookworm AS ns-cmsis-nn-ci

ARG NS_CMSIS_NN_TAG=unknown
ARG FLATBUFFERS_VERSION=v25.2.10
ARG UV_VERSION=0.6.14

LABEL org.opencontainers.image.version="${NS_CMSIS_NN_TAG}" \
      com.ambiq.ns-cmsis-nn.tag="${NS_CMSIS_NN_TAG}"

ENV NS_CMSIS_NN_TAG="${NS_CMSIS_NN_TAG}"

RUN apt-get update \
      && apt-get install -y --no-install-recommends \
            binutils \
            bzip2 \
            ca-certificates \
            clang \
            clang-format \
            cmake \
            curl \
            g++ \
            gcc \
            gcc-arm-none-eabi \
            git \
            git-lfs \
            jq \
            libnewlib-arm-none-eabi \
            llvm \
            make \
            ninja-build \
            ruby-full \
            sudo \
            unzip \
            xxd \
            xz-utils \
            zip \
      && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --no-cache-dir "uv==${UV_VERSION}"

# Python dependencies the unit-test drivers import at runtime. Baking them keeps
# the FVP/host/lib-variant CI jobs free of per-run pip installs.
RUN python -m pip install --no-cache-dir numpy termcolor pyserial

RUN set -eux; \
      clang --version; \
      clang-format --version; \
      arm-none-eabi-gcc --version; \
      uv --version; \
      printf '#include <string.h>\nint main(void) { char d[4]; memset(d, 0, sizeof(d)); return d[0]; }\n' > /tmp/ns-cmsis-nn-smoke.c; \
      arm-none-eabi-gcc -mcpu=cortex-m4 -mthumb -mfpu=fpv4-sp-d16 -mfloat-abi=hard \
            -c /tmp/ns-cmsis-nn-smoke.c -o /tmp/ns-cmsis-nn-smoke.o; \
      rm -f /tmp/ns-cmsis-nn-smoke.c /tmp/ns-cmsis-nn-smoke.o

# Needed when the docker container is used with GitHub actions.
RUN git config --global --add safe.directory /github/workspace

# Pin FlatBuffers to a known-good release so the devcontainer build stays deterministic.
RUN \
    git clone --branch ${FLATBUFFERS_VERSION} --depth 1 https://github.com/google/flatbuffers.git /tmp/flatbuffers && \
    cmake -S /tmp/flatbuffers -B /tmp/flatbuffers/build -G "Unix Makefiles" -DCMAKE_BUILD_TYPE=Release -DFLATBUFFERS_BUILD_TESTS=OFF && \
    cmake --build /tmp/flatbuffers/build --parallel && \
    cmake --install /tmp/flatbuffers/build && \
    rm -rf /tmp/flatbuffers

# ---------------------------------------------------------------------------
# Arm tooling (CMSIS-Toolbox, compilers, Corstone-300 FVP) baked via vcpkg.
#
# The repo's vcpkg-configuration.json is the single source of truth for the
# pinned tool set. We install the artifacts at build time and capture the
# activation environment (PATH entries + per-tool env vars such as
# AC6_TOOLCHAIN_6_24_0 / GCC_TOOLCHAIN_14_3_1 / CLANG_TOOLCHAIN_20_1_0) into
# vcpkg-env.json. CI jobs replay that file via .github/actions/cmsis-env
# instead of downloading toolchains at job runtime.
# ---------------------------------------------------------------------------
ENV VCPKG_ROOT=/opt/vcpkg \
    VCPKG_DOWNLOADS=/opt/vcpkg/downloads \
    VCPKG_DISABLE_METRICS=1
ENV NS_CMSIS_NN_VCPKG_ENV=/opt/ns-cmsis-nn/vcpkg-env.json

# Pin the vcpkg-tool release. Bump both when adopting a newer vcpkg-tool;
# get the sha256 from the matching GitHub release's vcpkg-glibc asset.
ARG VCPKG_TOOL_VERSION=2026-07-27
ARG VCPKG_TOOL_SHA256=7e97ef6bcd58f74d079f40d086b801a0222c5d15e4ea0d8d507a538033493d04

# Pin the companion vcpkg-artifacts STANDALONE BUNDLE release asset (the
# same release tag as VCPKG_TOOL_VERSION, unless a mismatch is ever
# intentionally needed). Bump alongside VCPKG_TOOL_VERSION/VCPKG_TOOL_SHA256;
# get the sha256 from the matching GitHub release's
# vcpkg-standalone-bundle.tar.gz asset.
ARG VCPKG_STANDALONE_BUNDLE_VERSION=2026-07-27
ARG VCPKG_STANDALONE_BUNDLE_SHA256=55c0af29ee5075feeb94cec3f7ba64f17638deaace930b7ef0b5793ea0e7c379

# Bootstrap the vcpkg-artifacts standalone bundle into VCPKG_ROOT, THEN
# place the separately pinned vcpkg-glibc binary on top of it.
#
# Historical note (AmbiqAI/ns-cmsis-nn#228): dot-sourcing Microsoft's
# aka.ms/vcpkg-init.sh only downloads the vcpkg binary and defines a
# `vcpkg-shell` shell *function* in the sourcing process -- it never puts a
# `vcpkg` command on PATH. `bash -c '. vcpkg-init.sh && vcpkg version'`
# therefore failed with "vcpkg: command not found" even though the
# download itself succeeded, and the floating aka.ms script can also bump
# its pinned VCPKG_BASE_VERSION upstream without notice. We instead
# download the pinned vcpkg-tool release directly, checksum-verify it
# (matching the fetch_toolchain.sh convention used elsewhere in this repo),
# and place it on PATH ourselves so both this Dockerfile and anything that
# later shells into the image can invoke `vcpkg` reliably.
#
# Second historical note (live recovery run 31337539282 job 93305784722,
# after the fix above landed): the vcpkg-glibc BINARY alone is NOT
# sufficient for artifacts commands (`vcpkg x-update-registry --all`,
# `vcpkg activate`). Those shell out to the bundled Node-based
# vcpkg-artifacts tooling described by scripts/vcpkg-tools.json -- that
# file (along with vcpkg-artifacts.mjs, .vcpkg-root, and the scripts/ and
# triplets/ trees) only ships in the STANDALONE BUNDLE release asset, never
# in the vcpkg-glibc binary alone. Downloading only vcpkg-glibc (as this
# Dockerfile did before this second fix) built the image and ran
# `vcpkg version` successfully, but `vcpkg x-update-registry --all` then
# failed with:
#   /opt/vcpkg/scripts/vcpkg-tools.json: No such file or directory
# We download+checksum+extract the standalone bundle first (it has no
# top-level directory, so it unpacks directly into VCPKG_ROOT), THEN
# download+checksum+install the pinned vcpkg-glibc binary over it --
# mirroring the aka.ms one-liner bootstrap's own two-artifact shape, but
# fully pinned/checksummed instead of floating.
RUN set -eux; \
      mkdir -p "${VCPKG_ROOT}" /opt/ns-cmsis-nn; \
      curl -fsSL "https://github.com/microsoft/vcpkg-tool/releases/download/${VCPKG_STANDALONE_BUNDLE_VERSION}/vcpkg-standalone-bundle.tar.gz" \
            -o /tmp/vcpkg-standalone-bundle.tar.gz; \
      echo "${VCPKG_STANDALONE_BUNDLE_SHA256}  /tmp/vcpkg-standalone-bundle.tar.gz" | sha256sum -c -; \
      tar -xzf /tmp/vcpkg-standalone-bundle.tar.gz -C "${VCPKG_ROOT}"; \
      rm -f /tmp/vcpkg-standalone-bundle.tar.gz; \
      curl -fsSL "https://github.com/microsoft/vcpkg-tool/releases/download/${VCPKG_TOOL_VERSION}/vcpkg-glibc" \
            -o "${VCPKG_ROOT}/vcpkg"; \
      echo "${VCPKG_TOOL_SHA256}  ${VCPKG_ROOT}/vcpkg" | sha256sum -c -; \
      chmod +x "${VCPKG_ROOT}/vcpkg"; \
      echo "${VCPKG_TOOL_VERSION}" > "${VCPKG_ROOT}/vcpkg-version.txt"

ENV PATH="${VCPKG_ROOT}:${PATH}"

# Verify `vcpkg` reliably resolves via PATH (not just via the full path used
# below), so a regression here fails the image build instead of silently
# skipping downstream release assets.
RUN vcpkg version

# Build-time contract: the standalone bundle's companion files that
# artifacts commands (x-update-registry, activate) depend on must be
# present BEFORE we ever invoke those commands below. Fail the image build
# here, with a specific message, instead of surfacing an oblique "No such
# file or directory" from deep inside vcpkg's Node-based artifacts tooling.
RUN set -eux; \
      for f in .vcpkg-root scripts/vcpkg-tools.json vcpkg-artifacts.mjs; do \
            if [ ! -f "${VCPKG_ROOT}/${f}" ]; then \
                  echo "missing required vcpkg-artifacts companion file: ${VCPKG_ROOT}/${f} (standalone bundle extraction incomplete or missing)" >&2; \
                  exit 1; \
            fi; \
      done; \
      [ -d "${VCPKG_ROOT}/triplets" ] || { echo "missing required vcpkg-artifacts companion directory: ${VCPKG_ROOT}/triplets" >&2; exit 1; }

# Resolve and install the pinned tool set, then bake the activation environment.
#
# Third historical note: an earlier revision of this step ran
# `vcpkg x-update-registry --all` before `vcpkg activate`, and separately
# deleted VCPKG_DOWNLOADS immediately afterward. Both were mistakes,
# discovered and fixed together:
#
#  - `vcpkg activate` already performs its own registry refresh internally
#    (it updates both the default microsoft/vcpkg-ce-catalog registry and
#    the repo's custom `arm` registry from vcpkg-configuration.json as part
#    of resolving artifacts). The separate `x-update-registry --all` call
#    was therefore redundant -- it exercised the exact same
#    vcpkg-ce-catalog fetch/extract path a second time for no benefit, and
#    during local isolated-Docker validation (linux/amd64, 2026-08) it was
#    observed to occasionally fail with an internal vcpkg-artifacts
#    `z-extract.cpp: Value was null` error and, separately, with plain
#    network timeouts while downloading the catalog's bundled Node
#    runtime -- both consistent with a flaky upstream fetch, not a
#    permanent post-sunset breakage: `vcpkg activate` alone (this block)
#    was rerun successfully multiple times in a row with no code changes.
#    Dropping the redundant call removes one of the two places that fetch
#    could flake, and `contracts/vcpkg_bundle_companion_files` (see
#    scripts/check_vcpkg_bundle_companion_files.sh) now asserts this
#    Dockerfile never reintroduces a bare/explicit default-registry update
#    ahead of `vcpkg activate`.
#  - VCPKG_DOWNLOADS is exactly where `vcpkg activate` installs the
#    resolved tool binaries (e.g.
#    /opt/vcpkg/downloads/artifacts/<hash>/tools.open.cmsis.pack.cmsis.toolbox/...),
#    and per the header comment above, CI is meant to replay vcpkg-env.json
#    via .github/actions/cmsis-env using those exact on-disk paths instead
#    of re-downloading toolchains at job runtime. The old `rm -rf
#    "${VCPKG_DOWNLOADS}"` here deleted those toolchains from the image
#    immediately after installing them, before the smoke check below (or
#    any later consumer of vcpkg-env.json) could ever use them. This bug
#    was never previously exposed because `x-update-registry --all` had
#    always failed before execution reached this far in every prior
#    live/local run. It is intentionally not reinstated in any form (not
#    even moved later): the installed tool tree under VCPKG_DOWNLOADS is
#    meant to persist in the built image.
COPY vcpkg-configuration.json /opt/ns-cmsis-nn/vcpkg-configuration.json
RUN set -eux; \
      cd /opt/ns-cmsis-nn; \
      "${VCPKG_ROOT}/vcpkg" activate \
            --downloads-root="${VCPKG_DOWNLOADS}" \
            --json="${NS_CMSIS_NN_VCPKG_ENV}"; \
      jq -e '.tools' "${NS_CMSIS_NN_VCPKG_ENV}" >/dev/null

# Build-time smoke check: prove the baked environment resolves every tool.
#
# armclang is special-cased: hosted GitHub runners (and this Dockerfile's own
# build, which never receives ARMLMD_LICENSE_FILE as a build secret) have no
# Arm Compiler for Embedded license, so `armclang --version` always fails its
# license checkout here -- this is the exact same documented, expected,
# permanent condition already handled as an optional/best-effort asset
# throughout .github/workflows/release.yml (AmbiqAI/ns-cmsis-nn#228; see
# docs/guides/releases.md). We still assert vcpkg resolved and installed the
# armclang binary itself (via `command -v`), so a real regression -- e.g. the
# tool failing to install, or failing for any reason OTHER than the known
# license-checkout error -- still fails this build. We do not attempt to
# provide a license here.
RUN set -eux; \
      export PATH="$(jq -r '.paths.PATH[]?' "${NS_CMSIS_NN_VCPKG_ENV}" | tr '\n' ':')${PATH}"; \
      cbuild --version; \
      cpackget --version; \
      csolution --version; \
      command -v armclang; \
      if ! armclang_out="$(armclang --version 2>&1)"; then \
            if ! printf '%s' "${armclang_out}" | grep -q "ARMLMD_LICENSE_FILE is not set"; then \
                  echo "armclang --version failed for an unexpected reason (not the known no-license condition):" >&2; \
                  echo "${armclang_out}" >&2; \
                  exit 1; \
            fi; \
            echo "armclang --version: no Arm Compiler license configured for this build (expected; AmbiqAI/ns-cmsis-nn#228), skipping." >&2; \
      fi; \
      arm-none-eabi-gcc --version; \
      command -v FVP_Corstone_SSE-300
