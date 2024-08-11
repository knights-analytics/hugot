#--- dockerfile to test hugot  ---

ARG GO_VERSION=1.22.6
ARG ONNXRUNTIME_VERSION=1.18.0
ARG BUILD_PLATFORM=linux/amd64

#--- build and test layer ---

FROM --platform=$BUILD_PLATFORM public.ecr.aws/amazonlinux/amazonlinux:2023 AS hugot-build
ARG GO_VERSION
ARG ONNXRUNTIME_VERSION

COPY ./go.mod .

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

RUN tokenizer_version=$(grep 'github.com/daulet/tokenizers' go.mod | awk '{print $2}') && \
    tokenizer_version=$(echo $tokenizer_version | awk -F'-' '{print $NF}') && \
    echo "tokenizer_version: $tokenizer_version" && \
    curl -LO https://github.com/daulet/tokenizers/releases/download/${tokenizer_version}/libtokenizers.linux-amd64.tar.gz && \
    tar -C /usr/lib -xzf libtokenizers.linux-amd64.tar.gz && \
    rm libtokenizers.linux-amd64.tar.gz

# go
RUN curl -LO https://golang.org/dl/go${GO_VERSION}.linux-amd64.tar.gz && \
    tar -C /usr/local -xzf go${GO_VERSION}.linux-amd64.tar.gz && \
    rm go${GO_VERSION}.linux-amd64.tar.gz
ENV PATH="$PATH:/usr/local/go/bin"

# onnxruntime cpu and gpu
RUN curl -LO https://github.com/microsoft/onnxruntime/releases/download/v${ONNXRUNTIME_VERSION}/onnxruntime-linux-x64-${ONNXRUNTIME_VERSION}.tgz && \
   tar -xzf onnxruntime-linux-x64-${ONNXRUNTIME_VERSION}.tgz && \
   mv ./onnxruntime-linux-x64-${ONNXRUNTIME_VERSION}/lib/libonnxruntime.so.${ONNXRUNTIME_VERSION} /usr/lib64/onnxruntime.so && \
   curl -LO https://github.com/microsoft/onnxruntime/releases/download/v${ONNXRUNTIME_VERSION}/onnxruntime-linux-x64-gpu-cuda12-${ONNXRUNTIME_VERSION}.tgz && \
   tar -xzf onnxruntime-linux-x64-gpu-cuda12-${ONNXRUNTIME_VERSION}.tgz && \
   mv ./onnxruntime-linux-x64-gpu-${ONNXRUNTIME_VERSION}/lib /usr/lib64/onnxruntime-gpu

# build gotestsum and test2json
RUN GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go build -o test2json -ldflags="-s -w" cmd/test2json && mv test2json /usr/local/bin/test2json && \
    curl -LO https://github.com/gotestyourself/gotestsum/releases/download/v1.12.0/gotestsum_1.12.0_linux_amd64.tar.gz && \
    tar -xzf gotestsum_1.12.0_linux_amd64.tar.gz --directory /usr/local/bin

# build cli binary
COPY . /build
WORKDIR /build
RUN cd ./cmd && CGO_ENABLED=1 CGO_LDFLAGS="-L/usr/lib/" GOOS=linux GOARCH=amd64 go build -a -o ./target main.go

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
FROM --platform=$BUILD_PLATFORM scratch AS artifacts

COPY --from=hugot-build /usr/lib64/onnxruntime.so onnxruntime-linux-x64.so
COPY --from=hugot-build /usr/lib64/onnxruntime-gpu onnxruntime-linux-x64-gpu
COPY --from=hugot-build /usr/lib/libtokenizers.a libtokenizers.a
COPY --from=hugot-build /build/cmd/target /hugot-cli-linux-x64
