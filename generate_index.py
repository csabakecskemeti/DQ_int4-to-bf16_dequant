#!/usr/bin/env python3
"""
Generate model.safetensors.index.json from converted BF16 safetensor files
"""

import os
import json
from glob import glob
from argparse import ArgumentParser
from safetensors.torch import load_file
from tqdm import tqdm


def generate_index(bf16_path):
    """
    Scan all .safetensors files and build the weight_map index.

    Args:
        bf16_path: Path to directory with converted BF16 safetensor files
    """
    weight_map = {}

    # Find all safetensor files
    safetensor_files = glob(os.path.join(bf16_path, "*.safetensors"))
    safetensor_files.sort()

    if not safetensor_files:
        print(f"Error: No .safetensors files found in {bf16_path}")
        return

    print(f"Found {len(safetensor_files)} safetensor files")
    print("Scanning files and building index...")

    # Scan each file and record tensor names
    for safetensor_file in tqdm(safetensor_files):
        file_name = os.path.basename(safetensor_file)

        # Load just the metadata (fast, doesn't load tensor data)
        tensors = load_file(safetensor_file, device="cpu")

        # Map each tensor to this file
        for tensor_name in tensors.keys():
            weight_map[tensor_name] = file_name

    # Create the index structure
    index_data = {
        "metadata": {
            "total_size": sum(os.path.getsize(f) for f in safetensor_files)
        },
        "weight_map": weight_map
    }

    # Save to model.safetensors.index.json
    index_file = os.path.join(bf16_path, "model.safetensors.index.json")
    with open(index_file, "w") as f:
        json.dump(index_data, f, indent=2)

    print(f"\n✓ Generated index file: {index_file}")
    print(f"  Total tensors: {len(weight_map)}")
    print(f"  Total size: {index_data['metadata']['total_size'] / 1024**3:.2f} GB")

    # Show sample entries
    print(f"\nSample weight_map entries:")
    for i, (key, value) in enumerate(list(weight_map.items())[:5]):
        print(f"  {key}: {value}")
    if len(weight_map) > 5:
        print(f"  ... and {len(weight_map) - 5} more")


def main():
    parser = ArgumentParser(description="Generate model.safetensors.index.json from BF16 files")
    parser.add_argument("bf16_path", type=str, help="Path to BF16 model directory")
    args = parser.parse_args()

    if not os.path.isdir(args.bf16_path):
        print(f"Error: {args.bf16_path} is not a directory")
        return

    generate_index(args.bf16_path)


if __name__ == "__main__":
    main()
