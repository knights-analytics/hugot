#!/bin/bash

set -e

cd /build && \
mkdir -p /test/unit && \
go run ./testData/downloadModels.go

echo HUGOT_BUILD_TAG is "$HUGOT_BUILD_TAG"

if [[ -n $HUGOT_BUILD_TAG ]]; then 
    echo "running with -tags=ALL"
    gotestsum --junitfile=/test/unit/unit.xml --jsonfile=/test/unit/unit.json -- -tags=ALL -coverprofile=/test/unit/cover.out.pre -coverpkg ./... -timeout 30m -race -covermode=atomic ./...
else 
    echo "running without build tags"
    gotestsum --junitfile=/test/unit/unit.xml --jsonfile=/test/unit/unit.json -- -coverprofile=/test/unit/cover.out.pre  ./... -timeout 30m -race -covermode=atomic ./...
fi

grep -v "downloadModels.go" /test/unit/cover.out.pre > /test/unit/cover.out && rm  /test/unit/cover.out.pre

echo Done.
