import os
import json
from argparse import ArgumentParser
from glob import glob
from tqdm import tqdm

import torch
from safetensors.torch import load_file, save_file

from kernel import weight_dequant_int4


def main(int4_path, bf16_path):
    torch.set_default_dtype(torch.bfloat16)
    os.makedirs(bf16_path, exist_ok=True)

    model_index_file = os.path.join(int4_path, "model.safetensors.index.json")
    with open(model_index_file, "r") as f:
        model_index = json.load(f)
    weight_map = model_index["weight_map"]

    # Cache for loaded safetensor files
    loaded_files = {}
    int4_weight_names = []
    # Track which .weight was created in which file
    dequantized_weight_map = {}

    # Helper function to get tensor from the correct file
    def get_tensor(tensor_name):
        file_name = weight_map[tensor_name]
        if file_name not in loaded_files:
            file_path = os.path.join(int4_path, file_name)
            loaded_files[file_name] = load_file(file_path, device="cpu")
        return loaded_files[file_name][tensor_name]

    safetensor_files = list(glob(os.path.join(int4_path, "*.safetensors")))
    safetensor_files.sort()

    for safetensor_file in tqdm(safetensor_files):
        file_name = os.path.basename(safetensor_file)
        current_state_dict = load_file(safetensor_file, device="cpu")
        loaded_files[file_name] = current_state_dict

        new_state_dict = {}

        # Step 1: Process all existing tensors (following FP8 pattern)
        for weight_name, weight in current_state_dict.items():
            # Skip quantization metadata tensors (equivalent to skipping _scale_inv)
            if any(suffix in weight_name for suffix in ['.weight_packed', '.weight_scale', '.weight_shape']):
                continue

            # Keep non-quantized tensors as-is (attention layers, embeddings, etc.)
            new_state_dict[weight_name] = weight.to(torch.bfloat16)

        # Step 2: CREATE dequantized weights from packed tensors (INT4-specific step)
        packed_tensors = []
        for name in current_state_dict.keys():
            if name.endswith('.weight_packed'):
                packed_tensors.append(name)

        print(f"  Found {len(packed_tensors)} packed tensors in {file_name}")

        # Dequantize each packed tensor
        for packed_name in packed_tensors:
            base_name = packed_name[:-len('.weight_packed')]
            weight_name = f"{base_name}.weight"
            scale_name = f"{base_name}.weight_scale"
            shape_name = f"{base_name}.weight_shape"

            try:
                # Get tensors (may be in different files)
                if packed_name in current_state_dict:
                    packed = current_state_dict[packed_name]
                else:
                    packed = get_tensor(packed_name)

                if scale_name in current_state_dict:
                    scale = current_state_dict[scale_name]
                else:
                    scale = get_tensor(scale_name)

                if shape_name in current_state_dict:
                    shape = current_state_dict[shape_name]
                else:
                    shape = get_tensor(shape_name)

                # Dequantize INT4 → BF16
                dequantized_weight = weight_dequant_int4(packed, scale, shape)
                new_state_dict[weight_name] = dequantized_weight
                int4_weight_names.append(weight_name)
                # Track this weight for the index file
                dequantized_weight_map[weight_name] = file_name

            except KeyError as e:
                print(f"Warning: Missing quantization data for {base_name}: {e}")
            except Exception as e:
                print(f"Warning: Dequantization failed for {base_name}: {e}")

        new_safetensor_file = os.path.join(bf16_path, file_name)
        save_file(new_state_dict, new_safetensor_file)

        # Memory management: keep only the 2 most recently used files
        if len(loaded_files) > 2:
            oldest_file = next(iter(loaded_files))
            del loaded_files[oldest_file]

    # Update model index - remove quantization metadata from weight_map
    new_model_index_file = os.path.join(bf16_path, "model.safetensors.index.json")
    new_weight_map = {}

    for key, value in weight_map.items():
        # Remove quantization metadata entries
        if not any(suffix in key for suffix in ['.weight_packed', '.weight_scale', '.weight_shape']):
            new_weight_map[key] = value

    # Add the dequantized .weight entries
    for weight_name, file_name in dequantized_weight_map.items():
        new_weight_map[weight_name] = file_name

    # Save updated model index
    with open(new_model_index_file, "w") as f:
        json.dump({"metadata": {}, "weight_map": new_weight_map}, f, indent=2)

    # Copy other config files
    config_files = [
        "config.json",
        "generation_config.json",
        "tokenizer_config.json",
        "chat_template.jinja",
        "configuration_deepseek.py",
        "modeling_deepseek.py",
        "tokenization_kimi.py",
        "tiktoken.model",
        "README.md",
        "LICENSE"
    ]

    for config_file in config_files:
        src_path = os.path.join(int4_path, config_file)
        if os.path.exists(src_path):
            dst_path = os.path.join(bf16_path, config_file)

            if config_file == "config.json":
                # Remove quantization_config from config.json
                with open(src_path, 'r') as f:
                    config = json.load(f)

                if "quantization_config" in config:
                    print("Removing quantization_config from config.json")
                    del config["quantization_config"]

                with open(dst_path, 'w') as f:
                    json.dump(config, f, indent=2)
            else:
                # Copy other files as-is
                import shutil
                shutil.copy2(src_path, dst_path)

    # Copy directories if they exist
    for dir_name in ["docs", "figures"]:
        src_dir = os.path.join(int4_path, dir_name)
        if os.path.exists(src_dir):
            dst_dir = os.path.join(bf16_path, dir_name)
            import shutil
            shutil.copytree(src_dir, dst_dir, dirs_exist_ok=True)

    print(f"\nConversion completed!")
    print(f"Converted {len(int4_weight_names)} INT4 weights to BF16")
    print(f"Output directory: {bf16_path}")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--input-int4-hf-path", type=str, required=True)
    parser.add_argument("--output-bf16-hf-path", type=str, required=True)
    args = parser.parse_args()
    main(args.input_int4_hf_path, args.output_bf16_hf_path)