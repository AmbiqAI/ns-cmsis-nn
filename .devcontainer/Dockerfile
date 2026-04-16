FROM python:3.10-bookworm as ns-cmsis-nn-ci

ARG NS_CMSIS_NN_TAG=unknown
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
        make \
        ninja-build \
        ruby-full \
        sudo \
        unzip \
        xxd \
        zip \
    && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --no-cache-dir "uv==${UV_VERSION}"

COPY docker/vendor/armclang /opt/armclang

ENV ARM_COMPILER_DIR=/opt/armclang
ENV PATH="/opt/armclang/bin:${PATH}"

# Needed when the docker container is used with GitHub actions.
RUN git config --global --add safe.directory /github/workspace

RUN test -x /opt/armclang/bin/armclang \
    && test -d /opt/armclang/lib \
    && arm-none-eabi-gcc --version \
    && uv --version

# Install flatbuffers (flatc)
RUN \
    git clone https://github.com/google/flatbuffers.git /tmp/flatbuffers && \
    cd /tmp/flatbuffers && \
    cmake -G "Unix Makefiles" && \
    make && \
    make install && \
    rm -rf flatbuffers
