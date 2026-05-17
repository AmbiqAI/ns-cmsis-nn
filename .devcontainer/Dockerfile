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
            ca-certificates \
            clang \
            clang-format \
            cmake \
            curl \
            gcc-arm-none-eabi \
            git \
            git-lfs \
            jq \
            libnewlib-arm-none-eabi \
            make \
            ninja-build \
            ruby-full \
            sudo \
            unzip \
            xxd \
            zip \
      && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --no-cache-dir "uv==${UV_VERSION}"

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
