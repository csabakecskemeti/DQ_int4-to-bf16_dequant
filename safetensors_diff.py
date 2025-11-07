#!/usr/bin/env python3
"""
SafeTensors Diff Utility - Like diff but for safetensors files
"""

import os
import sys
import torch
from safetensors.torch import load_file

def diff_tensors(file1, file2):
    """Show vimdiff-like side by side comparison"""
    tensors1 = load_file(file1, device="cpu")
    tensors2 = load_file(file2, device="cpu")

    all_names = sorted(set(tensors1.keys()) | set(tensors2.keys()))

    print(f"{'FILE 1: ' + os.path.basename(file1):<120} | {'FILE 2: ' + os.path.basename(file2):<120}")
    print(f"{'-'*120} | {'-'*120}")

    for name in all_names:
        in_file1 = name in tensors1
        in_file2 = name in tensors2

        if in_file1 and in_file2:
            tensor1 = tensors1[name]
            tensor2 = tensors2[name]

            left = f"{name} {str(tensor1.shape)} {str(tensor1.dtype)}"
            right = f"{name} {str(tensor2.shape)} {str(tensor2.dtype)}"

            print(f"{left:<120} | {right:<120}")

        elif in_file1 and not in_file2:
            tensor1 = tensors1[name]
            left = f"{name} {str(tensor1.shape)} {str(tensor1.dtype)}"
            print(f"{left:<120} | {'<MISSING>':<120}")

        elif not in_file1 and in_file2:
            tensor2 = tensors2[name]
            right = f"{name} {str(tensor2.shape)} {str(tensor2.dtype)}"
            print(f"{'<MISSING>':<120} | {right:<120}")

def show_file(file_path):
    """Show file contents like cat"""
    tensors = load_file(file_path, device="cpu")
    print(f"File: {file_path}")
    print(f"Size: {os.path.getsize(file_path) / 1024 / 1024:.1f} MB")
    print(f"Tensors: {len(tensors)}")
    print()

    for name, tensor in sorted(tensors.items()):
        print(f"{name:<80} {str(tensor.shape):<20} {str(tensor.dtype)}")

def main():
    if len(sys.argv) not in [2, 3]:
        print("Usage:")
        print("  python safetensors_diff.py <file>           # Show file contents")
        print("  python safetensors_diff.py <file1> <file2> # Diff two files")
        sys.exit(1)

    if len(sys.argv) == 2:
        show_file(sys.argv[1])
    else:
        diff_tensors(sys.argv[1], sys.argv[2])

if __name__ == "__main__":
    main()