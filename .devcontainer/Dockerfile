FROM python:3.10-bookworm as ns-cmsis-nn-ci

ARG NS_CMSIS_NN_TAG=unknown
ARG HELIA_CORE_TESTER_TAG=unknown

LABEL org.opencontainers.image.version="${NS_CMSIS_NN_TAG}" \
      com.ambiq.ns-cmsis-nn.tag="${NS_CMSIS_NN_TAG}" \
      com.ambiq.helia-core-tester.tag="${HELIA_CORE_TESTER_TAG}"

ENV NS_CMSIS_NN_TAG="${NS_CMSIS_NN_TAG}" \
    HELIA_CORE_TESTER_TAG="${HELIA_CORE_TESTER_TAG}"

RUN apt-get update

RUN apt-get install -y zip xxd sudo ruby-full cmake pip curl git git-lfs

RUN apt install -y lsb-release wget software-properties-common gnupg
RUN wget https://apt.llvm.org/llvm.sh
RUN chmod +x llvm.sh
RUN ./llvm.sh 16
RUN ln -s /usr/bin/clang-16 /usr/bin/clang
RUN ln -s /usr/bin/clang++-16 /usr/bin/clang++

RUN apt-get install clang-format-16
RUN ln -s /usr/bin/clang-format-16 /usr/bin/clang-format

# Needed when the docker container is used with GitHub actions.
RUN git config --global --add safe.directory /github/workspace

# Install flatbuffers (flatc)
RUN \
    git clone https://github.com/google/flatbuffers.git /tmp/flatbuffers && \
    cd /tmp/flatbuffers && \
    cmake -G "Unix Makefiles" && \
    make && \
    make install && \
    rm -rf flatbuffers
