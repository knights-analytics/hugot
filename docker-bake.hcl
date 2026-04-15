# docker-bake.hcl

# ---- Versions ----
variable "GO_VERSION"                { default = "1.26.2" }
variable "GOTESTSUM_VERSION"         { default = "1.13.0" }
variable "GOPJRT_VERSION"            { default = "0.98.0" }
variable "ONNXRUNTIME_VERSION"       { default = "1.24.4" }
variable "ONNXRUNTIME_GENAI_VERSION" { default = "0.12.2" }
variable "JAX_CUDA_VERSION"          { default = "0.9.1" }

target "base" {
  context   = "."
  platforms = ["linux/amd64"]
  args = {
    GO_VERSION                = GO_VERSION
    GOPJRT_VERSION            = GOPJRT_VERSION
    ONNXRUNTIME_VERSION       = ONNXRUNTIME_VERSION
    ONNXRUNTIME_GENAI_VERSION = ONNXRUNTIME_GENAI_VERSION
    JAX_CUDA_VERSION          = JAX_CUDA_VERSION
  }
}

target "hugot" {
  inherits   = ["base"]
  dockerfile = "Dockerfile"
  tags = [
    "hugot:latest"
  ]
  output = ["type=docker"]
}

target "hugot-cuda" {
  inherits   = ["base"]
  dockerfile = "cuda.Dockerfile"
  tags = [
    "hugot-cuda:latest",
  ]
  output = ["type=docker"]
}

target "hugot-test" {
  inherits   = ["base"]
  dockerfile = "test.Dockerfile"
  target     = "hugot-test"
  tags = [
    "hugot-test:latest"
  ]
  output = ["type=docker"]
}

target "hugot-ghcr" {
  inherits   = ["base"]
  dockerfile = "Dockerfile"
  tags = [
    "ghcr.io/knights-analytics/hugot:latest"
  ]
  output = ["type=registry"]
}
