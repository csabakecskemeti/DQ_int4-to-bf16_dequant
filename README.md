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
