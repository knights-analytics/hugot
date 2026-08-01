# PPLX Q4 INT8 parity verifier

Pinned Python ONNX Runtime reference for the PPLX Q4 native INT8 qualification harness.

```bash
uv sync
uv run python verify_pplx_int8_parity.py \
  --manifest ../../testdata/pplx/manifest.json \
  --model-dir ../../.local-pplx/onnx \
  --fixture ../../testdata/pplx/golden_fixture.json
```

The verifier independently tokenizes every raw fixture string with the pinned
`tokenizer.json`, checks the resulting token IDs and attention masks against the
fixture, then uses those fresh arrays for Python ONNX Runtime inference. It also
checks exact signed INT8 output bytes and SHA-256 values.

Model artifacts are not committed. Provide a local model directory whose files
match the hashes in `testdata/pplx/manifest.json`.
