#!/bin/bash

set -e

cd /build && \
mkdir -p /test/unit && \
gotestsum --junitfile=/test/unit/unit.xml --jsonfile=/test/unit/unit.json -- -coverprofile=/test/unit/cover.out -race -covermode=atomic ./...

echo Done.
