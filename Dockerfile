#--- dockerfile to test hugot  ---

ARG GO_VERSION=1.26.0
ARG ONNXRUNTIME_VERSION=1.24.1
ARG ONNXRUNTIME_GENAI_VERSION=0.12.0
ARG GOPJRT_VERSION=0.83.4
ARG BUILD_PLATFORM=linux/amd64

#--- runtime layer with all hugot dependencies for cpu 
#--- the image generated does not contain the hugot code, only the dependencies needed by hugot and the compiled cli binary

FROM --platform=$BUILD_PLATFORM public.ecr.aws/amazonlinux/amazonlinux:2023 AS hugot-runtime
ARG GO_VERSION
ARG ONNXRUNTIME_VERSION
ARG ONNXRUNTIME_GENAI_VERSION
ARG GOPJRT_VERSION

ENV PATH="$PATH:/usr/local/go/bin" \
    GOPJRT_NOSUDO=1

COPY ./scripts/download-onnxruntime.sh /download-onnxruntime.sh
COPY ./scripts/download-onnxruntime-genai.sh /download-onnxruntime-genai.sh
RUN --mount=src=./go.mod,dst=/go.mod \
    dnf --allowerasing -y install gcc jq bash tar xz gzip glibc-static libstdc++ wget zip git dirmngr sudo which && \
    ln -s /usr/lib64/libstdc++.so.6 /usr/lib64/libstdc++.so && \
    dnf clean all && \
    # go
    curl -LO https://golang.org/dl/go${GO_VERSION}.linux-amd64.tar.gz && \
    tar -C /usr/local -xzf go${GO_VERSION}.linux-amd64.tar.gz && \
    rm go${GO_VERSION}.linux-amd64.tar.gz && \
    # tokenizers
    tokenizer_version=$(grep 'github.com/daulet/tokenizers' /go.mod | awk '{print $2}') && \
    tokenizer_version=$(echo $tokenizer_version | awk -F'-' '{print $NF}') && \
    echo "tokenizer_version: $tokenizer_version" && \
    curl -LO https://github.com/daulet/tokenizers/releases/download/${tokenizer_version}/libtokenizers.linux-amd64.tar.gz && \
    tar -C /usr/lib -xzf libtokenizers.linux-amd64.tar.gz && \
    rm libtokenizers.linux-amd64.tar.gz && \
    # onnxruntime cpu
    sed -i 's/\r//g' /download-onnxruntime.sh && chmod +x /download-onnxruntime.sh && \
    /download-onnxruntime.sh ${ONNXRUNTIME_VERSION} && \
    # onnxruntime genai
    sed -i 's/\r//g' /download-onnxruntime-genai.sh && chmod +x /download-onnxruntime-genai.sh && \
    /download-onnxruntime-genai.sh ${ONNXRUNTIME_GENAI_VERSION} && \
    # XLA/goMLX
    GOPROXY=direct go run github.com/gomlx/go-xla/cmd/pjrt_installer@latest -plugin=linux -version=v${GOPJRT_VERSION} -path=/usr/local/lib/go-xla && \
    # NON-PRIVILEGED USER
    # create non-privileged testuser with id: 1000
    useradd -u 1000 -m testuser && usermod -a -G wheel testuser && \
    echo "testuser ALL=(ALL) NOPASSWD: ALL" >> /etc/sudoers.d/testuser

COPY . /build
RUN cd /build && \
    chown -R testuser:testuser /build && \
    # cli binary
    cd /build/cmd && CGO_ENABLED=1 CGO_LDFLAGS="-L/usr/lib/" GOOS=linux GOARCH=amd64 go build -tags "ALL" -a -o /cli main.go && \
    cd / && rm -rf /build
