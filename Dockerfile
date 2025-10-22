#--- dockerfile to test hugot  ---

ARG GO_VERSION=1.25.3
ARG ONNXRUNTIME_VERSION=1.22.0

#--- runtime layer with all hugot dependencies for cpu 
#--- the image generated does not contain the hugot code, only the dependencies needed by hugot and the compiled cli binary

FROM --platform=$BUILDPLATFORM public.ecr.aws/amazonlinux/amazonlinux:2023 AS hugot-runtime
ARG GO_VERSION
ARG TARGETOS
ARG TARGETARCH
ARG ONNXRUNTIME_VERSION

ENV PATH="$PATH:/usr/local/go/bin" \
    GOPJRT_NOSUDO=1

COPY ./scripts/download-onnxruntime.sh /download-onnxruntime.sh
COPY ./scripts/install-gomlx-gopjrt.sh /install-gomlx-gopjrt.sh
COPY ./scripts/install-tokenizers.sh /install-tokenizers.sh

RUN --mount=src=./go.mod,dst=/go.mod \
    dnf --allowerasing -y install gcc jq bash tar xz gzip glibc-static libstdc++ wget zip git dirmngr sudo which && \
    ln -s /usr/lib64/libstdc++.so.6 /usr/lib64/libstdc++.so && \
    dnf clean all && \
    # go
    curl -LO https://golang.org/dl/go${GO_VERSION}.${TARGETOS}-${TARGETARCH}.tar.gz && \
    tar -C /usr/local -xzf go${GO_VERSION}.${TARGETOS}-${TARGETARCH}.tar.gz && \
    rm go${GO_VERSION}.${TARGETOS}-${TARGETARCH}.tar.gz && \
    # tokenizers
    sed -i 's/\r//g' /install-tokenizers.sh && chmod +x /install-tokenizers.sh && \
    /install-tokenizers.sh ${TARGETOS} ${TARGETARCH} && \
    # onnxruntime cpu
    sed -i 's/\r//g' /download-onnxruntime.sh && chmod +x /download-onnxruntime.sh && \
    /download-onnxruntime.sh ${TARGETOS} ${TARGETARCH} ${ONNXRUNTIME_VERSION} && \
    # XLA/goMLX
    sed -i 's/\r//g' /install-gomlx-gopjrt.sh && chmod +x /install-gomlx-gopjrt.sh && \
    /install-gomlx-gopjrt.sh && \
    # NON-PRIVILEGED USER
    # create non-privileged testuser with id: 1000
    useradd -u 1000 -m testuser && usermod -a -G wheel testuser && \
    echo "testuser ALL=(ALL) NOPASSWD: ALL" >> /etc/sudoers.d/testuser

COPY . /build

RUN cd /build && \
    chown -R testuser:testuser /build && \
    # cli binary
    cd /build/cmd && CGO_ENABLED=1 CGO_LDFLAGS="-L/usr/lib/" GOOS=${TARGETOS} GOARCH=${TARGETARCH} go build -tags "ALL" -a -o /cli main.go && \
    cd / && rm -rf /build
