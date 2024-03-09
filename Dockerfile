ARG GO_VERSION=1.22.0
ARG RUST_VERSION=1.76
ARG ONNXRUNTIME_VERSION=1.17.1

#--- rust build of tokenizer ---

FROM rust:$RUST_VERSION AS tokenizer

RUN git clone https://github.com/knights-analytics/tokenizers -b main && \
    cd tokenizers && \
    cargo build --release

#--- build layer ---

FROM public.ecr.aws/amazonlinux/amazonlinux:2023 AS building
ARG GO_VERSION
ARG ONNXRUNTIME_VERSION

RUN dnf -y install gcc jq bash tar xz gzip glibc-static libstdc++ wget zip git && \
    ln -s /usr/lib64/libstdc++.so.6 /usr/lib64/libstdc++.so && \
    dnf install -y 'dnf-command(config-manager)' && \
    dnf config-manager \
    --add-repo https://download.fedoraproject.org/pub/fedora/linux/releases/39/Everything/x86_64/os/ && \
    dnf clean all

RUN curl -LO https://golang.org/dl/go${GO_VERSION}.linux-amd64.tar.gz && \
    tar -C /usr/local -xzf go${GO_VERSION}.linux-amd64.tar.gz && \
    rm go${GO_VERSION}.linux-amd64.tar.gz
ENV PATH="$PATH:/usr/local/go/bin"

COPY --from=tokenizer /tokenizers/target/release/libtokenizers.a /usr/lib/libtokenizers.a
RUN curl -LO https://github.com/microsoft/onnxruntime/releases/download/v${ONNXRUNTIME_VERSION}/onnxruntime-linux-x64-${ONNXRUNTIME_VERSION}.tgz && \
   tar -xzf onnxruntime-linux-x64-${ONNXRUNTIME_VERSION}.tgz && \
   mv ./onnxruntime-linux-x64-${ONNXRUNTIME_VERSION}/lib/libonnxruntime.so.${ONNXRUNTIME_VERSION} /usr/lib64/onnxruntime.so

RUN GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go build -o test2json -ldflags="-s -w" cmd/test2json && mv test2json /usr/local/bin/test2json && \
    curl -LO https://github.com/gotestyourself/gotestsum/releases/download/v1.11.0/gotestsum_1.11.0_linux_amd64.tar.gz && \
    tar -xzf gotestsum_1.11.0_linux_amd64.tar.gz --directory /usr/local/bin

COPY . /build
WORKDIR /build
RUN go mod download && CGO_ENABLED=1 GOOS=linux GOARCH=amd64 && \
    mkdir /unittest && go test -c . -o /unittest/pipelines.test && \
    go clean -r -cache -testcache -modcache

# cli build
RUN cd ./cmd && CGO_ENABLED=1 GOOS=linux GOARCH=amd64 go build -a -o ./target main.go

COPY ./models /models

# NON-PRIVILEDGED USER
# create non-priviledged testuser with id: 1000
RUN dnf install --disablerepo=* --enablerepo=amazonlinux --allowerasing -y dirmngr && dnf clean all
RUN useradd -u 1000 -m testuser && chown -R testuser:testuser /unittest

#--- test layer

FROM public.ecr.aws/amazonlinux/amazonlinux:2023 AS testing

RUN dnf install --disablerepo=* --enablerepo=amazonlinux --allowerasing -y dirmngr && dnf clean all

COPY --from=building /usr/lib64/onnxruntime.so /usr/lib64/onnxruntime.so
COPY --from=building /usr/lib/libtokenizers.a /usr/lib/libtokenizers.a
COPY --from=building /unittest /unittest
COPY --from=building /usr/local/bin/test2json /usr/local/bin/test2json
COPY --from=building /usr/local/bin/gotestsum /usr/local/bin/gotestsum
COPY --from=building /models /models
COPY --from=building /build/cmd/target /usr/local/bin/hugot

ENV GOVERSION=$GO_VERSION

# NON-PRIVILEDGED USER
# create non-priviledged testuser with id: 1000
RUN useradd -u 1000 -m testuser && chown -R testuser:testuser /unittest

# ENTRYPOINT
COPY ./scripts/entrypoint.sh /entrypoint.sh
# convert windows line endings if present
RUN sed -i 's/\r//g' /entrypoint.sh
RUN chmod +x /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]

#--- artifacts layer

FROM scratch AS artifacts

COPY --from=building /usr/lib64/onnxruntime.so onnxruntime.so
COPY --from=building /usr/lib/libtokenizers.a libtokenizers.a
COPY --from=building /build/cmd/target /hugot-cli-linux-amd64