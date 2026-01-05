# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This repository implements **AutoCompressors** - language models that can compress context information into summary vectors and reason over these compressed representations. The project is based on the EMNLP'23 paper "Adapting Language Models to Compress Long Contexts".

AutoCompressors use two special control tokens:
- `<sum>` - appended at the end of each segment to trigger compression
- `<eoc>` - emitted internally to stop compression (never exposed as visible text)

## Training Commands

### Train OPT-based AutoCompressor
```bash
bash run/train.sh
```

Key environment variables you can override:
- `BASE` - base model (default: "opt-125m")
- `BATCH` - total batch size (default: 16)
- `LR` - learning rate (default: 2e-5)
- `NUM_GPUS` - number of GPUs (default: 1)
- `CML` - compression_max_len (default: 50)

Example with custom settings:
```bash
BASE=opt-2.7b BATCH=32 LR=1e-4 NUM_GPUS=4 bash run/train.sh
```

### Train Llama-based AutoCompressor
```bash
bash run/train_llama.sh
```

Uses LoRA fine-tuning by default for Llama models. Key differences:
- Uses `meta-llama/Llama-2-7b-hf` as base
- Pre-configured with LoRA settings (r=16, alpha=16, targets: q_proj, v_proj, o_proj, k_proj)
- Uses RedPajama dataset from HuggingFace hub

### Evaluation
```bash
bash run/eval_llama.sh
```

Set `MODE=FA` for full-attention baseline evaluation (no compression).

### Direct Training Script
```bash
python train.py \
  --model_name_or_path facebook/opt-125m \
  --do_train \
  --do_eval \
  --output_dir checkpoints/my_run \
  --compression_max_len 50 \
  --segments_per_substep 2 \
  --training_substeps 2
```

## Architecture

### Core Components

**`auto_compressor.py`** - Main model implementations
- `AutoCompressorMixin` - Base mixin class with compression logic
- `OPTAutoCompressorModel` - OPT-based implementation (default)
- `LlamaAutoCompressorModel` - Llama-based implementation
- `SummaryConfig` - Tracks token constitution (softprompt length, compression length, etc.)

Key methods:
- `compress_context()` - Variable-length compression loop that generates summary vectors
- `forward()` - Handles both training/eval (with segmentation) and generation (with past_key_values)
- Model automatically segments input and compresses when `<sum>` token is encountered

**`substep_trainer.py`** - Custom trainer for BPTT with gradient detaching
- `SubstepTrainer` - Extends HuggingFace Trainer with substep logic
- `training_step()` - Processes multiple substeps per training step
- `training_substep()` - Single substep with softprompt detaching
- `segment_input()` - Handles input segmentation (random or fixed)
- Implements truncated backpropagation through time (BPTT) by detaching softprompts between substeps

**`train.py`** - Main training script
- Parses arguments from `args.py` (ModelArguments, DataTrainingArguments, TrainingArguments)
- Loads preprocessed datasets or processes raw data
- Dynamically imports correct model class based on model name (LlamaAutoCompressorModel vs OPTAutoCompressorModel)
- Sets up LoRA if specified
- Instantiates SubstepTrainer

**`modeling_flash_llama.py`** - Llama implementation with Flash Attention
- Custom Llama model using Flash Attention kernels for memory efficiency
- Required for Llama-based AutoCompressors

### Data Pipeline

**`data.py`** - Dataset loading and preprocessing
- `load_preprocessed_datasets()` - Loads pre-tokenized datasets from HuggingFace hub or disk
- `load_raw_dataset()` - Loads raw text data
- `preprocess_datasets()` - Tokenizes and chunks text into fixed-size blocks
- Training uses pre-tokenized datasets (e.g., "awettig/Pile-Books3-0.5B-6K-opt")

### Key Concepts

**Variable-Length Compression**:
- At each segment boundary marked by `<sum>`, the model runs a compression loop
- Loop continues for up to `compression_max_len` steps
- At training time: uses differentiable stopping with survival probabilities
- At inference time: stops when `<eoc>` is predicted or threshold exceeded

**Summary Vectors (Softprompt)**:
- Compressed representation of context, prepended to subsequent segments
- Can be obtained explicitly via `output_softprompt=True` or implicitly via `segment_lengths`
- Accumulates across segments if `accumulate_summary=True` (default)
- Detached between substeps for truncated BPTT

**Positional Embeddings**:
- Softprompt and compression tokens receive zero positional embeddings
- `OPTLearnedPositionalEmbeddingWithPadding` handles this for OPT models
- Llama models use custom rotary embeddings handling

## Important Configuration

**Training hyperparameters** (in TrainingArguments):
- `compression_max_len` - Max compression loop steps (default: 8)
- `compression_lambda` - Weight on length penalty (default: 0.0)
- `compression_alpha` - Scaling for original length in penalty calculation (default: 1.0)
- `truncate_bptt_segments` - Detach memory after N segments for BPTT (default: 1)
- `compress_stop_threshold` - Deterministic stopping threshold at inference (default: None)
- `training_substeps` - Number of substeps per training step (default: 1)
- `randomize_substeps` - Randomize segment boundaries (default: False)
- `segments_per_substep` - Number of segments per substep (default: 2)
- `segment_lengths` - Fixed segment lengths when not randomizing (default: [])
- `segment_gradient_checkpointing` - Use gradient checkpointing per segment (default: False)

**Model-specific notes**:
- Llama models require `torch_dtype=torch.bfloat16` and CUDA for Flash Attention
- OPT models support experimental fast attention via `--fast_attention` flag
- LoRA is commonly used for Llama models to reduce memory requirements

## Pre-trained Models

Available on HuggingFace hub:
- `princeton-nlp/AutoCompressor-Llama-2-7b-6k` - Llama-2-7b trained on 6K sequences
- `princeton-nlp/AutoCompressor-2.7b-6k` - OPT-2.7b trained on 6K sequences
- `princeton-nlp/AutoCompressor-2.7b-30k` - OPT-2.7b trained on 30K sequences

Load with:
```python
from auto_compressor import LlamaAutoCompressorModel  # or OPTAutoCompressorModel
model = LlamaAutoCompressorModel.from_pretrained("princeton-nlp/AutoCompressor-Llama-2-7b-6k")
```

## Usage Patterns

**Getting summary vectors explicitly**:
```python
out = model(input_ids, attention_mask, output_softprompt=True)
summary_vectors = out.softprompt
# Reuse in next forward pass
out = model(next_input_ids, softprompt=summary_vectors)
```

**Implicit compression with segment_lengths**:
```python
# Automatically compresses after each segment if segment ends with <sum>
out = model(input_ids, segment_lengths=[2048, 2048, 2048])
```

**Generation with summary vectors**:
```python
# Compress context
summary_vecs = model(context_tokens, output_softprompt=True).softprompt
# Generate with compressed context
output = model.generate(prompt_tokens, softprompt=summary_vecs, max_new_tokens=50)
```

## Dependencies

- PyTorch 2.1.0
- transformers==4.34.0
- flash-attn==2.3.5 (for Flash Attention)
- Flash rotary embeddings: `pip install git+https://github.com/Dao-AILab/flash-attention.git#subdirectory=csrc/rotary`
- Other: datasets, accelerate, sentencepiece, wandb

## Branch Context

Current branch: `codex/implement-variable-length-autocompressor-townb8`

Recent commits show implementation of variable-length auto-compression with `<sum>`/`<eoc>` control tokens and compression gating.
