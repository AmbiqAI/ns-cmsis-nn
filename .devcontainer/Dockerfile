FROM python:3.10-bookworm@sha256:2cffd68eda34762d010cba5de2e4462fcdab15ac2a53722b7cb17e8c76255cb3 AS ns-cmsis-nn-ci

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

# clang-format is pip-installed so the dev container matches the CI gate and the
# pre-commit pin (16.0.6); the apt package on this base image is 14.0.6.
RUN python -m pip install --no-cache-dir clang-format==16.0.6

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

# Immutable CI tooling is installed directly from vendor archives recorded in
# the repository-owned manifest. No package solver or floating registry is
# involved; every downloaded byte is SHA-256 verified before extraction.
ENV NS_CMSIS_NN_TOOLS=/opt/ns-cmsis-nn/tools \
    NS_CMSIS_NN_TOOL_ENV=/opt/ns-cmsis-nn/tool-env.json
COPY ci/tools/manifest.json /opt/ns-cmsis-nn/tool-manifest.json
COPY scripts/install_ci_tools.sh /opt/ns-cmsis-nn/install_ci_tools.sh
COPY scripts/ci/validate_ci_tool_manifest.jq /opt/ns-cmsis-nn/ci/validate_ci_tool_manifest.jq
RUN bash /opt/ns-cmsis-nn/install_ci_tools.sh \
      /opt/ns-cmsis-nn/tool-manifest.json \
      "${NS_CMSIS_NN_TOOLS}" \
      "${NS_CMSIS_NN_TOOL_ENV}"

ENV PATH="/opt/ns-cmsis-nn/tools/cmsis-toolbox/bin:/opt/ns-cmsis-nn/tools/cmake/bin:/opt/ns-cmsis-nn/tools/ninja:/opt/ns-cmsis-nn/tools/arm-gnu/bin:/opt/ns-cmsis-nn/tools/llvm-embedded/bin:/opt/ns-cmsis-nn/tools/armclang/bin:/opt/ns-cmsis-nn/tools/corstone-300-fvp/bin:${PATH}" \
    GCC_TOOLCHAIN_14_3_1="/opt/ns-cmsis-nn/tools/arm-gnu/bin" \
    CLANG_TOOLCHAIN_19_1_5="/opt/ns-cmsis-nn/tools/llvm-embedded/bin" \
    AC6_TOOLCHAIN_6_23_32="/opt/ns-cmsis-nn/tools/armclang/bin" \
    CLANG_TOOLCHAIN="/opt/ns-cmsis-nn/tools/llvm-embedded/bin" \
    AC6_TOOLCHAIN="/opt/ns-cmsis-nn/tools/armclang/bin" \
    AVH_FVP_PLUGINS="/opt/ns-cmsis-nn/tools/corstone-300-fvp/plugins"

# Build-time smoke check: prove the baked environment resolves every required
# tool. Armclang's executable is checked without requiring a license checkout.
RUN set -eux; \
      cbuild --version; \
      cpackget --version; \
      csolution --version; \
      command -v armclang; \
      arm-none-eabi-gcc --version; \
      clang --version; \
      cmake --version; \
      ninja --version; \
      command -v FVP_Corstone_SSE-300
