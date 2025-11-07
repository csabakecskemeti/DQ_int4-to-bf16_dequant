# DQ_int4-to-bf16_dequant

INT4 dequantization to BF16 for models like moonshotai/Kimi-K2-Thinking

Inspired and based on the Deepseek V3 FP8 to BF16 dequantizer
https://huggingface.co/deepseek-ai/DeepSeek-V3-Base/blob/main/inference/fp8_cast_bf16.py

## Usage
usage: int4_cast_bf16_fixed.py [-h] --input-int4-hf-path INPUT_INT4_HF_PATH --output-bf16-hf-path OUTPUT_BF16_HF_PATH
int4_cast_bf16_fixed.py: error: the following arguments are required: --input-int4-hf-path, --output-bf16-hf-path

NOTE:
generate_index.py is added as a temp solution when the first version has not generated the safetensor indes json file.
Now the conversion script should generating it.

## safetensors_diff.py 
Debug utility I've used to compare the original and converted safetensors side by side

### Usage
  python safetensors_diff.py <file>           # Show file contents
  python safetensors_diff.py <file1> <file2> # Diff two files

## TEST-PROOF
Converted moonshotai/Kimi-K2-Thinking to BF16 then converted to GGUF and qunatized to Q3 GGUF seems working:
<img width="1740" height="535" alt="Screenshot From 2025-11-07 06-12-41" src="https://github.com/user-attachments/assets/4782b5b0-37b7-461f-9a35-0b1215cc5f41" />
