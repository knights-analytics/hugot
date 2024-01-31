#!/bin/bash

set -e

gotestsum --raw-command --junitfile "/test/unit/pipelines.xml" --jsonfile "/test/unit/pipelines.json" -- test2json -t -p pipelines /unittest/pipelines.test -test.v=test2json