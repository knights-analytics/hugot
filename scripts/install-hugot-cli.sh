#!/bin/bash

# The default version to download (leave empty for 'latest' version).
ver=""

# The default GitHub path to the specified version.
ver_path="latest/download"

# Default download destinations.
onnx_runtime_path=$HOME/lib/hugot/
hugot_path=$HOME/.local/bin/


# Help:
function show_help() {
    echo "Usage: "$(basename "${0}")" [OPTIONS]"
    echo "Download ONNX shared runtime and hugot CLI tool."
    echo
    echo "  -h, --help                 Show this help."
    echo "  -n, --onnx-runtime-path    Where to install onnxruntime.so, defaults to:"
    echo "                             ${onnx_runtime_path}"
    echo "  -u, --hugot-path           Where to install hugot CLI tool, defaults to:"
    echo "                             ${hugot_path}"
    echo "  -v, --version              Version to download. Defaults to latest. See:"
    echo "                             https://github.com/knights-analytics/hugot/releases"
}

# Parse the arguments, override the defaults.
VALID_ARGS=$(getopt -o 'hv:n:u:' --long 'help,version:,onnx-runtime-path:,hugot-path' -- "${@}")
if [[ $? -ne 0 ]]; then
    exit $?
fi
eval set -- "${VALID_ARGS}"
while [ : ]; do
    case "${1}" in
        -h|--help)
            show_help
            exit 0
            ;;
        -v|--version)
            ver="${2}"
            shift 2
            ;;
        -n|--onnx-runtime-path)
            onnx_runtime_path="${2}"
            shift 2
            ;;
        -u|--hugot-path)
            hugot_path="${2}"
            shift 2
            ;;
        --)
            shift
            break
            ;;
    esac
done

if [[ "${ver}" = "" ]]; then
    echo "Installing hugot CLI, latest version"
else
    echo "Installing hugot CLI, version ${ver}"
    if [[ "${ver}" =~ ^"v" ]]; then
        ver_path="download/${ver}"
    else
        ver_path="download/v${ver}"
    fi
fi

# Make the destination dirs.
mkdir -p "${onnx_runtime_path}" 2>/dev/null
mkdir -p "${hugot_path}" 2>/dev/null

# Locations to download binaries.
onnx_filename="${onnx_runtime_path}/onnxruntime.so"
hugot_filename="${hugot_path}/hugot"

# Tidy path names.
command -v readlink >/dev/null && {
    onnx_filename=$(readlink -m "${onnx_filename}")
    hugot_filename=$(readlink -m "${hugot_filename}")
}

echo "  ${onnx_filename}"
echo "  ${hugot_filename}"

# Fetch the binaries from GitHub.
# Note any failures so an appropriate exit code can be given at the end.
failures=""
{
    curl -S -s -\# -L https://github.com/knights-analytics/hugot/releases/"${ver_path}"/onnxruntime-linux-x64.so \
         -o "${onnx_filename}"
} && {
    echo "onnxruntime.so shared library installed at ${onnx_filename}"
} || {
    echo "onnxruntime.so could NOT be installed at ${onnx_filename}"
    ((failures++))
}

{
    curl -L https://github.com/knights-analytics/hugot/releases/"${ver_path}"/hugot-cli-linux-x64 \
         -o "${hugot_filename}"
} && {
    # Make hugot executable.
    chmod +x "${hugot_filename}" 2>/dev/null

    echo "hugot installed at ${hugot_filename}"
} || {
    echo "hugot could NOT be installed at ${hugot_filename}"
    ((failures++))
}

if [[ "${failures}" -gt 0 ]]; then
    exit 1
fi
