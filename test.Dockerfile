ARG BUILDPLATFORM

FROM ghcr.io/knights-analytics/hugot/models:latest AS models

FROM --platform=$BUILDPLATFORM hugot:latest AS hugot-test

COPY . /build
COPY --from=models /models /build/models

RUN cd /build && \
    chown -R testuser:testuser /build && \
    curl -LO https://github.com/gotestyourself/gotestsum/releases/download/v1.13.0/gotestsum_1.13.0_linux_amd64.tar.gz && \
    tar -xzf gotestsum_1.13.0_linux_amd64.tar.gz --directory /usr/local/bin && \
    cp /build/scripts/entrypoint.sh /entrypoint.sh && sed -i 's/\r//g' /entrypoint.sh && chmod +x /entrypoint.sh && \
    cp /build/scripts/run-unit-tests-container.sh /run-unit-tests-container.sh && sed -i 's/\r//g' /run-unit-tests-container.sh && chmod +x /run-unit-tests-container.sh

ENTRYPOINT ["/entrypoint.sh"]

#--- artifacts layer ---
FROM --platform=$BUILDPLATFORM scratch AS artifacts

COPY --from=hugot-test /usr/lib/libonnxruntime.so libonnxruntime-linux-x64.so
COPY --from=hugot-test /usr/lib/libonnxruntime-genai.so libonnxruntime-genai-linux-x64.so
COPY --from=hugot-test /usr/lib/libtokenizers.a libtokenizers.a
