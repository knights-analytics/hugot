"""Verify PPLX Q4 native INT8 parity against pinned qualification data.

The verifier checks artifact hashes and manifest identity, independently tokenizes
every fixture input from raw text, runs Python ONNX Runtime with the fresh token
arrays, and compares the resulting signed INT8 bytes with the pinned fixture.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer


def sha256_file(path: Path) -> str:
    """Return the SHA-256 hex digest of a file read in 1 MiB chunks."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_manifest_digest(manifest: dict[str, Any]) -> str:
    """Return the SHA-256 of canonical JSON with manifest_digest omitted.

    The digest covers every manifest field except ``manifest_digest`` itself so the
    recorded digest can be self-describing. Canonicalization uses UTF-8 JSON with
    sorted object keys and compact separators ``(',', ':')``.
    """
    payload = dict(manifest)
    payload.pop("manifest_digest", None)
    canonical = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def verify_manifest_digest(manifest: dict[str, Any]) -> None:
    """Verify the recorded manifest_digest matches the canonical digest algorithm."""
    recorded = manifest.get("manifest_digest")
    if not isinstance(recorded, str) or recorded == "":
        raise ValueError("manifest_digest is required")
    actual = canonical_manifest_digest(manifest)
    if actual != recorded:
        raise ValueError(f"manifest_digest {recorded} != canonical digest {actual}")


def load_json(path: Path) -> dict[str, Any]:
    """Load a JSON object from path."""
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise TypeError(f"{path} must contain a JSON object")
    return payload


def verify_manifest_files(manifest: dict[str, Any], model_dir: Path) -> None:
    """Verify every pinned file exists with the expected size and SHA-256."""
    expected_files: list[dict[str, Any]] = [
        manifest["graph"],
        *manifest["external_data"],
        *manifest["tokenizer_files"],
    ]
    for entry in expected_files:
        path = model_dir / entry["path"]
        if not path.is_file():
            raise FileNotFoundError(f"missing artifact {path}")
        size = path.stat().st_size
        if size != int(entry["size"]):
            raise ValueError(f"{path}: size {size} != expected {entry['size']}")
        digest = sha256_file(path)
        if digest != entry["sha256"]:
            raise ValueError(f"{path}: sha256 {digest} != expected {entry['sha256']}")


def load_tokenizer(manifest: dict[str, Any], model_dir: Path) -> Tokenizer:
    """Load the verified tokenizer JSON used by Hugot's ORT tokenizer path.

    The tokenizer is loaded only after ``verify_manifest_files`` has checked every
    tokenizer artifact. Padding and truncation are disabled explicitly because
    Hugot encodes each input independently, does not batch-pad tokenizer output,
    and applies any maximum-token limit after encoding.

    Args:
        manifest: Pinned artifact manifest containing the tokenizer file entries.
        model_dir: Directory containing the verified model and tokenizer artifacts.

    Returns:
        A tokenizer configured for independent, unpadded, untruncated encodings.

    Raises:
        ValueError: If the manifest does not contain ``tokenizer.json``.
    """
    tokenizer_entry = next(
        (
            entry
            for entry in manifest["tokenizer_files"]
            if entry["path"] == "tokenizer.json"
        ),
        None,
    )
    if tokenizer_entry is None:
        raise ValueError("manifest tokenizer_files must include tokenizer.json")

    tokenizer = Tokenizer.from_file(str(model_dir / tokenizer_entry["path"]))
    tokenizer.no_padding()
    tokenizer.no_truncation()
    return tokenizer


def load_hugot_max_allowed_tokens(manifest: dict[str, Any], model_dir: Path) -> int:
    """Read the post-encoding token limit applied by Hugot's model loader.

    Hugot reads ``max_position_embeddings`` from ``config.json`` and truncates
    encoded fields from the right when the value is positive. Returning zero
    preserves Hugot's unlimited behavior when the configuration omits the field.

    Args:
        manifest: Pinned artifact manifest containing the verified config entry.
        model_dir: Directory containing the verified model artifacts.

    Returns:
        The positive Hugot token limit, or zero when no limit is configured.

    Raises:
        ValueError: If the manifest does not contain ``config.json``.
    """
    config_entry = next(
        (
            entry
            for entry in manifest["tokenizer_files"]
            if entry["path"] == "config.json"
        ),
        None,
    )
    if config_entry is None:
        raise ValueError("manifest tokenizer_files must include config.json")

    config = load_json(model_dir / config_entry["path"])
    max_position_embeddings = config.get("max_position_embeddings")
    if isinstance(max_position_embeddings, int) and max_position_embeddings > 0:
        return max_position_embeddings
    return 0


def tokenize_fixture_case(
    tokenizer: Tokenizer,
    case: dict[str, Any],
    max_allowed_tokens: int,
) -> list[dict[str, list[int]]]:
    """Tokenize and validate every raw text in one fixture case.

    Each raw string is encoded separately with special-token insertion enabled.
    The resulting token IDs and attention mask are compared with the pinned
    fixture before any ONNX input is constructed. If Hugot's model configuration
    imposes a positive maximum, all returned encoding fields are truncated from
    the right after tokenization, matching Hugot's post-encoding behavior.

    Args:
        tokenizer: Verified Hugging Face tokenizer configured without padding or truncation.
        case: Fixture case containing raw texts and expected per-input token arrays.
        max_allowed_tokens: Hugot's post-encoding token limit, or zero for unlimited.

    Returns:
        Fresh token IDs and attention masks for each raw input, in fixture order.

    Raises:
        ValueError: If fixture structure, tokenization results, or sequence lengths differ.
    """
    case_name = str(case["name"])
    texts = case["texts"]
    expected_inputs = case["inputs"]
    if len(texts) != len(expected_inputs):
        raise ValueError(
            f"{case_name}: raw text count {len(texts)} != fixture input count {len(expected_inputs)}"
        )

    tokenized_inputs: list[dict[str, list[int]]] = []
    for index, text in enumerate(texts):
        encoding = tokenizer.encode(text, add_special_tokens=True)
        token_ids = list(encoding.ids)
        attention_mask = list(encoding.attention_mask)
        if len(token_ids) != len(attention_mask):
            raise ValueError(
                f"{case_name} input {index}: Python token ID length {len(token_ids)} "
                f"!= attention-mask length {len(attention_mask)}"
            )

        if max_allowed_tokens > 0 and len(token_ids) > max_allowed_tokens:
            token_ids = token_ids[:max_allowed_tokens]
            attention_mask = attention_mask[:max_allowed_tokens]

        expected = expected_inputs[index]
        expected_token_ids = [int(value) for value in expected["token_ids"]]
        expected_attention_mask = [int(value) for value in expected["attention_mask"]]
        if token_ids != expected_token_ids:
            raise ValueError(
                f"{case_name} input {index}: token_ids mismatch; "
                f"expected {expected_token_ids}, actual {token_ids}"
            )
        if attention_mask != expected_attention_mask:
            raise ValueError(
                f"{case_name} input {index}: attention_mask mismatch; "
                f"expected {expected_attention_mask}, actual {attention_mask}"
            )

        expected_sequence_length = int(expected["sequence_length"])
        if len(token_ids) != expected_sequence_length:
            raise ValueError(
                f"{case_name} input {index}: sequence_length mismatch; "
                f"expected {expected_sequence_length}, actual {len(token_ids)}"
            )
        tokenized_inputs.append(
            {
                "token_ids": token_ids,
                "attention_mask": attention_mask,
            }
        )

    actual_padded_sequence_length = max(
        (len(item["token_ids"]) for item in tokenized_inputs),
        default=0,
    )
    expected_padded_sequence_length = int(case["padded_sequence_length"])
    if actual_padded_sequence_length != expected_padded_sequence_length:
        raise ValueError(
            f"{case_name}: padded_sequence_length mismatch; "
            f"expected {expected_padded_sequence_length}, actual {actual_padded_sequence_length}"
        )
    return tokenized_inputs


def run_case(
    session: ort.InferenceSession,
    case: dict[str, Any],
    tokenized_inputs: list[dict[str, list[int]]],
    dimension: int,
) -> bytes:
    """Run one case with freshly tokenized inputs and return signed INT8 bytes.

    Args:
        session: Python ONNX Runtime session for the pinned graph.
        case: Fixture case containing batch metadata and expected output metadata.
        tokenized_inputs: Fresh per-input token IDs and attention masks validated
            against the fixture.
        dimension: Expected output vector width from the manifest.

    Returns:
        Contiguous row-major bytes containing the signed INT8 output tensor.

    Raises:
        ValueError: If batch or input metadata is inconsistent.
        TypeError: If the selected ONNX output is not INT8.
    """
    batch = len(tokenized_inputs)
    if batch != int(case["batch_size"]):
        raise ValueError(
            f"{case['name']}: batch_size mismatch; expected {case['batch_size']}, actual {batch}"
        )
    max_len = max(
        (len(item["token_ids"]) for item in tokenized_inputs),
        default=0,
    )
    if max_len == 0:
        raise ValueError(f"{case['name']}: cannot run an empty token batch")
    input_ids = np.zeros((batch, max_len), dtype=np.int64)
    attention = np.zeros((batch, max_len), dtype=np.int64)
    for index, row in enumerate(tokenized_inputs):
        token_ids = row["token_ids"]
        mask = row["attention_mask"]
        input_ids[index, : len(token_ids)] = token_ids
        attention[index, : len(mask)] = mask

    outputs = session.run(
        ["pooler_output_int8"],
        {
            "input_ids": input_ids,
            "attention_mask": attention,
        },
    )
    array = outputs[0]
    if array.dtype != np.int8:
        raise TypeError(f"{case['name']}: expected int8 output, got {array.dtype}")
    if array.shape != (batch, dimension):
        raise ValueError(
            f"{case['name']}: expected shape {(batch, dimension)}, got {array.shape}"
        )
    return array.tobytes(order="C")


def main() -> int:
    """Run the PPLX Q4 native INT8 qualification verifier."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest", required=True, type=Path, help="Pinned artifact manifest JSON"
    )
    parser.add_argument(
        "--model-dir", required=True, type=Path, help="Verified model directory"
    )
    parser.add_argument(
        "--fixture", required=True, type=Path, help="Golden fixture JSON"
    )
    args = parser.parse_args()

    manifest = load_json(args.manifest)
    fixture = load_json(args.fixture)
    model_dir = args.model_dir.resolve()

    verify_manifest_digest(manifest)
    verify_manifest_files(manifest, model_dir)
    tokenizer = load_tokenizer(manifest, model_dir)
    max_allowed_tokens = load_hugot_max_allowed_tokens(manifest, model_dir)

    expected_ort = str(manifest["ort_version"])
    if ort.__version__ != expected_ort:
        raise RuntimeError(f"onnxruntime {ort.__version__} != manifest {expected_ort}")

    if fixture.get("selected_output") != "pooler_output_int8":
        raise ValueError("fixture selected_output must be pooler_output_int8")
    if fixture.get("output_dtype") != "INT8":
        raise ValueError("fixture output_dtype must be INT8")

    graph_path = model_dir / manifest["graph"]["path"]
    session = ort.InferenceSession(str(graph_path), providers=["CPUExecutionProvider"])
    output_names = [output.name for output in session.get_outputs()]
    if "pooler_output_int8" not in output_names:
        raise RuntimeError(f"model outputs missing pooler_output_int8: {output_names}")

    dimension = int(manifest["dimension"])
    failures: list[str] = []
    for case in fixture["cases"]:
        tokenized_inputs = tokenize_fixture_case(tokenizer, case, max_allowed_tokens)
        actual = run_case(session, case, tokenized_inputs, dimension)
        expected = base64.b64decode(case["expected_bytes_b64"])
        actual_digest = hashlib.sha256(actual).hexdigest()
        if actual != expected:
            failures.append(f"{case['name']}: byte mismatch")
        if actual_digest != case["expected_sha256"]:
            failures.append(
                f"{case['name']}: sha256 {actual_digest} != {case['expected_sha256']}"
            )
        print(f"ok {case['name']} bytes={len(actual)} sha256={actual_digest}")

    if failures:
        for failure in failures:
            print(failure, file=sys.stderr)
        return 1
    print("PPLX INT8 Python parity verification passed")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
