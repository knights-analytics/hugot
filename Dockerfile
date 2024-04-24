ARG GO_VERSION=1.22.2
ARG RUST_VERSION=1.77
ARG ONNXRUNTIME_VERSION=1.17.3
ARG CUDA_VERSION=12.4

#--- rust build of tokenizer ---

FROM rust:$RUST_VERSION AS tokenizer

RUN git clone https://github.com/knights-analytics/tokenizers -b main && \
    cd tokenizers && \
    cargo build --release

#--- build and test layer ---

FROM public.ecr.aws/amazonlinux/amazonlinux:2023 AS hugot-build
ARG GO_VERSION
ARG ONNXRUNTIME_VERSION

RUN dnf -y install gcc jq bash tar xz gzip glibc-static libstdc++ wget zip git && \
    ln -s /usr/lib64/libstdc++.so.6 /usr/lib64/libstdc++.so && \
    dnf install -y 'dnf-command(config-manager)' && \
    dnf config-manager --add-repo https://download.fedoraproject.org/pub/fedora/linux/releases/39/Everything/x86_64/os/ && \
    # from fedora
    dnf config-manager --add-repo https://developer.download.nvidia.com/compute/cuda/repos/fedora39/x86_64/cuda-fedora39.repo && \
    dnf install -y cuda-cudart-12-4 libcublas-12-4 libcurand-12-4 libcufft-12-4 && \
    # from rhel
    dnf config-manager --add-repo https://developer.download.nvidia.com/compute/cuda/repos/rhel9/x86_64/cuda-rhel9.repo && \
    dnf install -y libcudnn8 && \
    dnf clean all

# go
RUN curl -LO https://golang.org/dl/go${GO_VERSION}.linux-amd64.tar.gz && \
    tar -C /usr/local -xzf go${GO_VERSION}.linux-amd64.tar.gz && \
    rm go${GO_VERSION}.linux-amd64.tar.gz
ENV PATH="$PATH:/usr/local/go/bin"

# tokenizer
COPY --from=tokenizer /tokenizers/target/release/libtokenizers.a /usr/lib/libtokenizers.a

# onnxruntime cpu and gpu
RUN curl -LO https://github.com/microsoft/onnxruntime/releases/download/v${ONNXRUNTIME_VERSION}/onnxruntime-linux-x64-${ONNXRUNTIME_VERSION}.tgz && \
   tar -xzf onnxruntime-linux-x64-${ONNXRUNTIME_VERSION}.tgz && \
   mv ./onnxruntime-linux-x64-${ONNXRUNTIME_VERSION}/lib/libonnxruntime.so.${ONNXRUNTIME_VERSION} /usr/lib64/onnxruntime.so && \
   curl -LO https://github.com/microsoft/onnxruntime/releases/download/v${ONNXRUNTIME_VERSION}/onnxruntime-linux-x64-gpu-cuda12-${ONNXRUNTIME_VERSION}.tgz && \
   tar -xzf onnxruntime-linux-x64-gpu-cuda12-${ONNXRUNTIME_VERSION}.tgz && \
   mv ./onnxruntime-linux-x64-gpu-${ONNXRUNTIME_VERSION}/lib /usr/lib64/onnxruntime-gpu

# build gotestsum and test2json
RUN GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go build -o test2json -ldflags="-s -w" cmd/test2json && mv test2json /usr/local/bin/test2json && \
    curl -LO https://github.com/gotestyourself/gotestsum/releases/download/v1.11.0/gotestsum_1.11.0_linux_amd64.tar.gz && \
    tar -xzf gotestsum_1.11.0_linux_amd64.tar.gz --directory /usr/local/bin

# build cli binary
COPY . /build
WORKDIR /build
RUN cd ./cmd && CGO_ENABLED=1 GOOS=linux GOARCH=amd64 go build -a -o ./target main.go

# NON-PRIVILEDGED USER
# create non-priviledged testuser with id: 1000
RUN dnf install --disablerepo=* --enablerepo=amazonlinux --allowerasing -y dirmngr sudo which && dnf clean all
RUN useradd -u 1000 -m testuser && chown -R testuser:testuser /build && usermod -a -G wheel testuser
RUN echo "testuser ALL=(ALL) NOPASSWD: ALL" >> /etc/sudoers.d/testuser

# ENTRYPOINT
COPY ./scripts/entrypoint.sh /entrypoint.sh
# convert windows line endings if present
RUN sed -i 's/\r//g' /entrypoint.sh
RUN chmod +x /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]

# artifacts layer
FROM scratch AS artifacts

COPY --from=hugot-build /usr/lib64/onnxruntime.so onnxruntime-linux-x64.so
COPY --from=hugot-build /usr/lib64/onnxruntime-gpu onnxruntime-linux-x64-gpu
COPY --from=hugot-build /usr/lib/libtokenizers.a libtokenizers.a
COPY --from=hugot-build /build/cmd/target /hugot-cli-linux-x64
