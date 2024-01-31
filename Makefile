SHELL := bash

.PHONY: run-tests clean

all:

run-tests:
	scripts/run-unit-tests.sh

clean:
	rm -r ./testTarget || true

start-dev-container:
	scripts/start-dev-container.sh

stop-dev-container:
	scripts/stop-dev-container.sh