#!/usr/bin/env python3
"""
Export T5/FLAN-T5/doc2query models to ONNX format for use with Hugot's Seq2SeqPipeline.

This script exports T5-based models using Hugging Face's Optimum library, which creates
three separate ONNX files for encoder-decoder models:
  - encoder_model.onnx
  - decoder_model.onnx (initial decoder, no past_key_values)
  - decoder_with_past_model.onnx (decoder with past_key_values for efficient generation)

Usage:
    python export_t5_onnx.py doc2query/msmarco-t5-small-v1 --output ./models/doc2query-t5-small
    python export_t5_onnx.py lmqg/flan-t5-small-squad-qg --output ./models/flan-t5-small-qg

Requirements:
    pip install optimum[exporters] onnx onnxruntime transformers

Examples:
    # Export doc2query-small (recommended for query generation)
    python export_t5_onnx.py doc2query/msmarco-t5-small-v1 -o ./models/doc2query-small

    # Export LMQG FLAN-T5 question generation model
    python export_t5_onnx.py lmqg/flan-t5-small-squad-qg -o ./models/flan-t5-small-qg

    # Export T5-small for general seq2seq
    python export_t5_onnx.py t5-small -o ./models/t5-small
"""

import argparse
import json
import os
import shutil
import sys
from pathlib import Path


def check_dependencies():
    """Check that required packages are installed."""
    missing = []

    try:
        import torch
    except ImportError:
        missing.append("torch")

    try:
        import transformers
    except ImportError:
        missing.append("transformers")

    try:
        import optimum
    except ImportError:
        missing.append("optimum[exporters]")

    try:
        import onnx
    except ImportError:
        missing.append("onnx")

    try:
        import onnxruntime
    except ImportError:
        missing.append("onnxruntime")

    if missing:
        print(f"Error: Missing required packages: {', '.join(missing)}")
        print(f"Install with: pip install {' '.join(missing)}")
        sys.exit(1)


def export_model(model_name: str, output_dir: str) -> None:
    """
    Export a T5 model to ONNX format using Optimum.

    Args:
        model_name: HuggingFace model name (e.g., 'doc2query/msmarco-t5-small-v1')
        output_dir: Directory to save the exported model
    """
    from optimum.exporters.onnx import main_export
    from transformers import AutoTokenizer, AutoConfig

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Exporting {model_name} to {output_dir}")

    # Load model config first to get model info
    print("\n1. Loading model configuration...")
    config = AutoConfig.from_pretrained(model_name)

    # Export to ONNX using Optimum
    print("\n2. Exporting to ONNX format...")
    print("   This may take a few minutes...")

    # Use optimum's main_export function
    main_export(
        model_name_or_path=model_name,
        output=output_path,
        task="text2text-generation-with-past",  # This creates encoder + decoder + decoder_with_past
        opset=14,
        device="cpu",
    )

    # Rename files to match expected naming convention for Hugot
    print("\n3. Renaming ONNX files to match Hugot conventions...")
    rename_map = {
        "encoder_model.onnx": "encoder.onnx",
        "decoder_model.onnx": "decoder-init.onnx",  # Initial decoder (no past)
        "decoder_with_past_model.onnx": "decoder.onnx",  # Decoder with past_key_values
    }

    for old_name, new_name in rename_map.items():
        old_path = output_path / old_name
        new_path = output_path / new_name
        if old_path.exists():
            shutil.move(str(old_path), str(new_path))
            print(f"   Renamed {old_name} -> {new_name}")

    # Copy tokenizer files - use AutoTokenizer for broader compatibility
    print("\n4. Ensuring tokenizer files are present...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.save_pretrained(str(output_path))

    # Create/update config.json with seq2seq-specific settings
    print("\n5. Updating config.json with seq2seq settings...")
    config_dict = config.to_dict()

    # Ensure required fields are present
    if "decoder_start_token_id" not in config_dict:
        config_dict["decoder_start_token_id"] = config_dict.get("pad_token_id", 0)
    if "eos_token_id" not in config_dict:
        config_dict["eos_token_id"] = 1  # Default for T5
    if "pad_token_id" not in config_dict:
        config_dict["pad_token_id"] = 0  # Default for T5
    if "num_decoder_layers" not in config_dict:
        config_dict["num_decoder_layers"] = config_dict.get("num_layers", 6)

    config_path = output_path / "config.json"
    with open(config_path, "w") as f:
        json.dump(config_dict, f, indent=2)

    # List exported files
    print("\n6. Export complete!")
    print(f"\nExported files in {output_dir}:")
    for f in sorted(output_path.iterdir()):
        if f.is_file():
            size_mb = f.stat().st_size / (1024 * 1024)
            print(f"   {f.name} ({size_mb:.1f} MB)")

    # Print usage instructions
    print("\n" + "=" * 60)
    print("Usage in Hugot (Go):")
    print("=" * 60)
    print(f"""
config := backends.PipelineConfig[*pipelines.Seq2SeqPipeline]{{
    ModelPath: "{output_dir}",
    Name:      "doc2query",
    Options: []backends.PipelineOption[*pipelines.Seq2SeqPipeline]{{
        pipelines.WithSeq2SeqMaxTokens(64),
        pipelines.WithNumReturnSequences(5),
        pipelines.WithSampling(0.95, 0.7),
    }},
}}

pipeline, err := pipelines.NewSeq2SeqPipeline(config, opts)
output, err := pipeline.Run([]string{{"Your document text here..."}})
// output.GeneratedTexts[0] contains the generated queries
""")


def test_exported_model(output_dir: str, test_input: str = None) -> None:
    """Test the exported model with a sample input."""
    from transformers import AutoTokenizer
    import onnxruntime as ort
    import numpy as np

    print("\n" + "=" * 60)
    print("Testing exported model...")
    print("=" * 60)

    output_path = Path(output_dir)

    # Check for required files
    encoder_file = output_path / "encoder.onnx"
    decoder_init_file = output_path / "decoder-init.onnx"
    decoder_file = output_path / "decoder.onnx"

    if not encoder_file.exists():
        print(f"Warning: encoder.onnx not found at {encoder_file}")
        return
    if not decoder_init_file.exists():
        print(f"Warning: decoder-init.onnx not found at {decoder_init_file}")
        return

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(output_path))

    # Load ONNX models
    print("\nLoading ONNX models...")
    encoder_session = ort.InferenceSession(str(encoder_file))
    decoder_init_session = ort.InferenceSession(str(decoder_init_file))

    # Test input
    if test_input is None:
        test_input = "Python is an interpreted, high-level programming language."
    print(f"\nInput: {test_input}")

    # Tokenize
    inputs = tokenizer(test_input, return_tensors="np", padding=True)
    input_ids = inputs["input_ids"].astype(np.int64)
    attention_mask = inputs["attention_mask"].astype(np.int64)

    # Run encoder
    print("\nRunning encoder...")
    encoder_outputs = encoder_session.run(
        None,
        {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        },
    )
    encoder_hidden_states = encoder_outputs[0]
    print(f"Encoder output shape: {encoder_hidden_states.shape}")

    # Run decoder-init for first token
    print("\nRunning decoder-init...")
    decoder_input_ids = np.array([[0]], dtype=np.int64)  # Start token

    # Get decoder-init input names to understand expected inputs
    decoder_init_inputs = {inp.name for inp in decoder_init_session.get_inputs()}
    print(f"Decoder-init expects inputs: {decoder_init_inputs}")

    feed_dict = {"input_ids": decoder_input_ids}
    if "encoder_hidden_states" in decoder_init_inputs:
        feed_dict["encoder_hidden_states"] = encoder_hidden_states
    if "encoder_attention_mask" in decoder_init_inputs:
        feed_dict["encoder_attention_mask"] = attention_mask

    decoder_outputs = decoder_init_session.run(None, feed_dict)
    logits = decoder_outputs[0]
    print(f"Decoder logits shape: {logits.shape}")

    # Get first predicted token
    next_token = np.argmax(logits[0, -1, :])
    decoded = tokenizer.decode([next_token], skip_special_tokens=True)
    print(f"\nFirst predicted token: {next_token} -> '{decoded}'")

    # Simple greedy generation (just a few tokens for testing)
    print("\nGenerating (greedy, max 20 tokens)...")
    generated_ids = [0]  # Start with decoder start token

    for _ in range(20):
        decoder_input_ids = np.array([generated_ids], dtype=np.int64)
        feed_dict = {"input_ids": decoder_input_ids}
        if "encoder_hidden_states" in decoder_init_inputs:
            feed_dict["encoder_hidden_states"] = encoder_hidden_states
        if "encoder_attention_mask" in decoder_init_inputs:
            feed_dict["encoder_attention_mask"] = attention_mask

        decoder_outputs = decoder_init_session.run(None, feed_dict)
        logits = decoder_outputs[0]
        next_token = int(np.argmax(logits[0, -1, :]))

        if next_token == 1:  # EOS token for T5
            break
        generated_ids.append(next_token)

    result = tokenizer.decode(generated_ids[1:], skip_special_tokens=True)  # Skip start token
    print(f"\nGenerated output: {result}")


def main():
    parser = argparse.ArgumentParser(
        description="Export T5/FLAN-T5/doc2query models to ONNX format for Hugot",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s doc2query/msmarco-t5-small-v1 -o ./models/doc2query-small
  %(prog)s lmqg/flan-t5-small-squad-qg -o ./models/flan-t5-qg
  %(prog)s lmqg/flan-t5-small-squad-qg -o ./models/flan-t5-qg --test
  %(prog)s lmqg/flan-t5-small-squad-qg -o ./models/flan-t5-qg --test --test-input "generate question: <hl> Python <hl> Python is a programming language."
        """
    )

    parser.add_argument(
        "model",
        help="HuggingFace model name (e.g., 'doc2query/msmarco-t5-small-v1', 'lmqg/flan-t5-small-squad-qg')"
    )
    parser.add_argument(
        "-o", "--output",
        required=True,
        help="Output directory for exported model"
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Test the exported model after export"
    )
    parser.add_argument(
        "--test-input",
        type=str,
        default=None,
        help="Custom input text for testing (e.g., 'generate question: <hl> Answer <hl> Context')"
    )

    args = parser.parse_args()

    # Check dependencies
    check_dependencies()

    # Export model
    export_model(args.model, args.output)

    # Optionally test
    if args.test:
        test_exported_model(args.output, args.test_input)


if __name__ == "__main__":
    main()
