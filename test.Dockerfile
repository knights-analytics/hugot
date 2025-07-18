ARG BUILD_PLATFORM=linux/amd64

FROM --platform=$BUILD_PLATFORM hugot:latest AS hugot-test

COPY . /build

RUN cd /build && \
    chown -R testuser:testuser /build && \
    curl -LO https://github.com/gotestyourself/gotestsum/releases/download/v1.12.3/gotestsum_1.12.3_linux_amd64.tar.gz && \
    tar -xzf gotestsum_1.12.3_linux_amd64.tar.gz --directory /usr/local/bin && \
    # entrypoint
    cp /build/scripts/entrypoint.sh /entrypoint.sh && sed -i 's/\r//g' /entrypoint.sh && chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]

#--- artifacts layer ---
FROM --platform=$BUILD_PLATFORM scratch AS artifacts

COPY --from=hugot-test /usr/lib64/onnxruntime.so onnxruntime-linux-x64.so
COPY --from=hugot-test /usr/lib/libtokenizers.a libtokenizers.a
COPY --from=hugot-test /cli /hugot-cli-linux-x64