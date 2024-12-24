SHELL := bash

.PHONY: run-tests clean

all:

run-tests:
	scripts/run-unit-tests.sh $(BUILD_TAG)

clean:
	rm -r ./testTarget || true
	rm -r ./artifacts || true

start-dev-container:
	scripts/start-dev-container.sh

stop-dev-container:
	scripts/stop-dev-container.sh