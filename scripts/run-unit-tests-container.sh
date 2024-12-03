#!/bin/bash

set -e

cd /build && \
mkdir -p /test/unit && \
go run ./testData/downloadModels.go && \
gotestsum --junitfile=/test/unit/unit.xml --jsonfile=/test/unit/unit.json -- -tags=ALL -coverprofile=/test/unit/cover.out.pre -race -covermode=atomic ./...
grep -v "downloadModels.go" /test/unit/cover.out.pre > /test/unit/cover.out
rm  /test/unit/cover.out.pre

echo Done.
