#--- dockerfile to test hugot  ---

ARG GO_VERSION
ARG ONNXRUNTIME_VERSION
ARG ONNXRUNTIME_GENAI_VERSION
ARG GOPJRT_VERSION
ARG JAX_CUDA_VERSION
ARG BUILDPLATFORM
ARG LLAMA_CPP_VERSION

FROM nvidia/cuda:13.5.0-cudnn-runtime-amzn2023 AS cuda

FROM nvidia/cuda:13.5.0-devel-amzn2023 AS llama-build
ARG LLAMA_CPP_VERSION
RUN dnf -y install git cmake gcc gcc-c++ make && dnf clean all && \
    git clone --depth 1 --branch "${LLAMA_CPP_VERSION}" https://github.com/ggml-org/llama.cpp.git /src/llama.cpp && \
    cmake -S /src/llama.cpp -B /src/llama.cpp/build \
      -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON -DGGML_CUDA=ON \
      -DCMAKE_CUDA_ARCHITECTURES="80;86;89;90;100;120" \
      -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_SERVER=OFF && \
    cmake --build /src/llama.cpp/build --parallel && \
    cmake --install /src/llama.cpp/build --prefix /opt/llama

#--- runtime layer with all hugot dependencies for cpu and gpu ---

FROM --platform=$BUILDPLATFORM public.ecr.aws/amazonlinux/amazonlinux:2023 AS hugot-runtime
ARG GO_VERSION
ARG ONNXRUNTIME_VERSION
ARG ONNXRUNTIME_GENAI_VERSION
ARG GOPJRT_VERSION
ARG JAX_CUDA_VERSION

ENV PATH="$PATH:/usr/local/go/bin" \
    GOPJRT_NOSUDO=1 \
    GOEXPERIMENT=simd

COPY ./scripts/download-onnxruntime.sh /download-onnxruntime.sh
COPY ./scripts/download-onnxruntime-genai.sh /download-onnxruntime-genai.sh
RUN --mount=src=./go.mod,dst=/go.mod \
    --mount=src=/,dst=/cudaassets,from=cuda \
    dnf --allowerasing -y install gcc jq bash tar xz gzip glibc-static libstdc++ wget zip git dirmngr sudo which && \
    ln -s /usr/lib64/libstdc++.so.6 /usr/lib64/libstdc++.so && \
    curl -LO https://download.fedoraproject.org/pub/fedora/linux/releases/43/Everything/x86_64/os/Packages/g/glibc-2.42-4.fc43.x86_64.rpm && \
    rpm -Uvh --nodeps --force glibc-2.42-4.fc43.x86_64.rpm && \
    rm glibc-2.42-4.fc43.x86_64.rpm && \
    # CUDA \
    mkdir -p /usr/local/cuda/lib64 /usr/lib64 && \
    cp -a \
      /cudaassets/usr/local/cuda-13.5/targets/x86_64-linux/lib/libcudart.so.* \
      /cudaassets/usr/local/cuda-13.5/targets/x86_64-linux/lib/libcublas.so.* \
      /cudaassets/usr/local/cuda-13.5/targets/x86_64-linux/lib/libcublasLt.so.* \
      /cudaassets/usr/local/cuda-13.5/targets/x86_64-linux/lib/libnvblas.so.* \
      /cudaassets/usr/local/cuda-13.5/targets/x86_64-linux/lib/libcurand.so.* \
      /cudaassets/usr/local/cuda-13.5/targets/x86_64-linux/lib/libcufft.so.* \
      /cudaassets/usr/local/cuda-13.5/targets/x86_64-linux/lib/libcufftw.so.* \
      /usr/local/cuda/lib64/ && \
    cp -a /cudaassets/usr/lib64/libcudnn*.so* /usr/lib64/ && \
    # Register the copied CUDA libraries with the dynamic linker. \
    echo '/usr/local/cuda/lib64' > /etc/ld.so.conf.d/cuda.conf && \
    ldconfig && \
    # go \
    curl -LO https://golang.org/dl/go${GO_VERSION}.linux-amd64.tar.gz && \
    tar -C /usr/local -xzf go${GO_VERSION}.linux-amd64.tar.gz && \
    rm go${GO_VERSION}.linux-amd64.tar.gz && \
    # onnxruntime cpu and gpu \
    sed -i 's/\r//g' /download-onnxruntime.sh && chmod +x /download-onnxruntime.sh && \
    /download-onnxruntime.sh --onnxruntime-version=${ONNXRUNTIME_VERSION} --cuda && \
    sed -i 's/\r//g' /download-onnxruntime-genai.sh && chmod +x /download-onnxruntime-genai.sh && \
    /download-onnxruntime-genai.sh --onnxruntime-genai-version=${ONNXRUNTIME_GENAI_VERSION} --cuda && \
    # XLA/goMLX \
    GOPROXY=direct go run github.com/gomlx/go-xla/cmd/pjrt_installer@latest -plugin=linux -version=v${GOPJRT_VERSION} -path=/usr/local/lib/go-xla && \
    GOPROXY=direct go run github.com/gomlx/go-xla/cmd/pjrt_installer@latest -plugin=cuda13 -version=${JAX_CUDA_VERSION} -path=/usr/local/lib/go-xla && \
    # NON-PRIVILEGED USER \
    # create non-privileged testuser with id: 1000
    useradd -u 1000 -m testuser && usermod -a -G wheel testuser && \
    echo "testuser ALL=(ALL) NOPASSWD: ALL" >> /etc/sudoers.d/testuser

COPY --from=llama-build /opt/llama /usr/local
RUN ldconfig
