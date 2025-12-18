#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "huggingface_hub",
# ]
# ///
"""
Download pre-converted GLiNER ONNX model from Hugging Face.

Usage:
    uv run scripts/export_gliner_onnx.py

This will download the GLiNER ONNX model to ./models/gliner-small-v2.1/
"""

from pathlib import Path
from huggingface_hub import snapshot_download


def main():
    # Use the onnx-community pre-converted model
    model_id = "onnx-community/gliner_small-v2.1"
    output_dir = Path("./models/gliner-small-v2.1")

    print(f"Downloading pre-converted ONNX model: {model_id}")
    print(f"Output directory: {output_dir}")

    # Download the model
    snapshot_download(
        repo_id=model_id,
        local_dir=str(output_dir),
        local_dir_use_symlinks=False,
    )

    # List created files
    print("\nDownloaded files:")
    for f in sorted(output_dir.rglob("*")):
        if f.is_file():
            size = f.stat().st_size
            if size > 1024 * 1024:
                size_str = f"{size / (1024*1024):.1f} MB"
            elif size > 1024:
                size_str = f"{size / 1024:.1f} KB"
            else:
                size_str = f"{size} B"
            rel_path = f.relative_to(output_dir)
            print(f"  {rel_path}: {size_str}")

    print("\n✅ Done! You can now run the GLiNER integration tests.")


if __name__ == "__main__":
    main()
