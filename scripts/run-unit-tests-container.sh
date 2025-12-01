#!/bin/bash

set -e

folder=/test/unit

cd /build && \
mkdir -p $folder && \
go run ./testcases/downloadModels.go

echo "Running ORT tests..."

gotestsum --format testname --junitfile=$folder/unit-ort.xml --jsonfile=$folder/unit-ort.json -- -coverprofile=$folder/cover-ort.out -coverpkg ./... -tags=ORT -timeout 60m -race

echo "ORT tests completed."

echo "Running XLA tests..."

gotestsum --format testname --junitfile=$folder/unit-xla.xml --jsonfile=$folder/unit-xla.json -- -coverprofile=$folder/cover-xla.out -coverpkg ./... -tags=XLA -timeout 60m

echo "XLA tests completed."

# echo "Running training tests..."

gotestsum --format testname --junitfile=$folder/unit-training.xml --jsonfile=$folder/unit-training.json -- -coverprofile=$folder/cover-training.out -coverpkg ./... -tags=ORT,XLA,TRAINING -timeout 60m

echo "Training tests completed."

# echo "Running simplego tests..."

# gotestsum --format testname --junitfile=$folder/unit-go.xml --jsonfile=$folder/unit-go.json -- -tags=GO -timeout 60m

# echo "simplego tests completed."

echo "merging coverage files"
head -n 1 $folder/cover-ort.out > $folder/cover.out
tail -n +2 $folder/cover-ort.out >> $folder/cover.out
tail -n +2 $folder/cover-xla.out >> $folder/cover.out
tail -n +2 $folder/cover-training.out >> $folder/cover.out
# tail -n +2 $folder/cover-go.out >> $folder/cover.out

head -n 1 $folder/cover.out > $folder/cover.dedup.out
tail -n +2 $folder/cover.out | sort | uniq >> $folder/cover.dedup.out
grep -v "downloadModels.go" $folder/cover.dedup.out > $folder/cover.final.out
cat $folder/cover.final.out

echo Done.
