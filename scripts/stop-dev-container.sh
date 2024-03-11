#!/bin/bash

set -e

# Directory of *this* script
this_dir="$( cd "$( dirname "$0" )" && pwd )"
export src_dir="$(realpath "${this_dir}/..")"

docker compose -f ./compose-dev.yaml down -v