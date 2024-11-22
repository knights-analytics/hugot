#--- dockerfile to test hugot  ---

ARG GO_VERSION=1.23.3
ARG ONNXRUNTIME_VERSION=1.20.0
ARG BUILD_PLATFORM=linux/amd64

#--- runtime layer with all hugot dependencies for cpu and gpu ---

FROM --platform=$BUILD_PLATFORM public.ecr.aws/amazonlinux/amazonlinux:2023 AS hugot-runtime
ARG GO_VERSION
ARG ONNXRUNTIME_VERSION

ENV PATH="$PATH:/usr/local/go/bin"

COPY ./scripts/download-onnxruntime.sh /download-onnxruntime.sh
RUN --mount=src=./go.mod,dst=/go.mod \
    sed -i 's/\r//g' /download-onnxruntime.sh && chmod +x /download-onnxruntime.sh && \
    dnf -y install gcc jq bash tar xz gzip glibc-static libstdc++ wget zip git && \
    ln -s /usr/lib64/libstdc++.so.6 /usr/lib64/libstdc++.so && \
    dnf install -y 'dnf-command(config-manager)' && \
    dnf config-manager --add-repo https://download.fedoraproject.org/pub/fedora/linux/releases/39/Everything/x86_64/os/ && \
    # from fedora
    dnf config-manager --add-repo https://developer.download.nvidia.com/compute/cuda/repos/fedora39/x86_64/cuda-fedora39.repo && \
    dnf install -y cuda-cudart-12-6 libcublas-12-6 libcurand-12-6 libcufft-12-6 && \
    # from rhel
    dnf config-manager --add-repo https://developer.download.nvidia.com/compute/cuda/repos/rhel9/x86_64/cuda-rhel9.repo && \
    dnf install -y libcudnn9-cuda-12 && \
    dnf clean all && \
    # NON-PRIVILEDGED USER
    # create non-priviledged testuser with id: 1000
    dnf install --disablerepo=* --enablerepo=amazonlinux --allowerasing -y dirmngr sudo which && dnf clean all && \
    useradd -u 1000 -m testuser && usermod -a -G wheel testuser && \
    echo "testuser ALL=(ALL) NOPASSWD: ALL" >> /etc/sudoers.d/testuser && \
    # tokenizers
    tokenizer_version=$(grep 'github.com/daulet/tokenizers' /go.mod | awk '{print $2}') && \
    tokenizer_version=$(echo $tokenizer_version | awk -F'-' '{print $NF}') && \
    echo "tokenizer_version: $tokenizer_version" && \
    curl -LO https://github.com/daulet/tokenizers/releases/download/${tokenizer_version}/libtokenizers.linux-amd64.tar.gz && \
    tar -C /usr/lib -xzf libtokenizers.linux-amd64.tar.gz && \
    rm libtokenizers.linux-amd64.tar.gz && \
    # go
    curl -LO https://golang.org/dl/go${GO_VERSION}.linux-amd64.tar.gz && \
    tar -C /usr/local -xzf go${GO_VERSION}.linux-amd64.tar.gz && \
    rm go${GO_VERSION}.linux-amd64.tar.gz && \
    # onnxruntime cpu and gpu \
    /download-onnxruntime.sh ${ONNXRUNTIME_VERSION}

#--- test layer ---

FROM --platform=$BUILD_PLATFORM hugot-runtime AS hugot-build

COPY . /build

RUN cd /build && \
    chown -R testuser:testuser /build && \
    # cli binary
    cd /build/cmd && CGO_ENABLED=1 CGO_LDFLAGS="-L/usr/lib/" GOOS=linux GOARCH=amd64 go build -a -o /cli main.go && \
    cd / && \
    curl -LO https://github.com/gotestyourself/gotestsum/releases/download/v1.12.0/gotestsum_1.12.0_linux_amd64.tar.gz && \
    tar -xzf gotestsum_1.12.0_linux_amd64.tar.gz --directory /usr/local/bin && \
    # entrypoint
    cp /build/scripts/entrypoint.sh /entrypoint.sh && sed -i 's/\r//g' /entrypoint.sh && chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]

#--- artifacts layer ---
FROM --platform=$BUILD_PLATFORM scratch AS artifacts

COPY --from=hugot-build /usr/lib64/onnxruntime.so onnxruntime-linux-x64.so
COPY --from=hugot-build /usr/lib64/onnxruntime-gpu onnxruntime-linux-x64-gpu
COPY --from=hugot-build /usr/lib/libtokenizers.a libtokenizers.a
COPY --from=hugot-build /cli /hugot-cli-linux-x64
