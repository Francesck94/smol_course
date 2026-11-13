---
license: apache-2.0
base_model: unsloth/Qwen3-1.7B-Base
datasets:
  - allenai/tulu-3-sft-mixture
tags:
  - unsloth
  - lora
  - qwen3
  - sft
  - instruction-tuning
library_name: transformers
pipeline_tag: text-generation
---

# Qwen3-1.7B SFT on Tulu-3

Instruction-tuned version of [`unsloth/Qwen3-1.7B-Base`](https://huggingface.co/unsloth/Qwen3-1.7B-Base),
fine-tuned with LoRA adapters on a 20k random subset of
[`allenai/tulu-3-sft-mixture`](https://huggingface.co/datasets/allenai/tulu-3-sft-mixture) using
[Unsloth](https://unsloth.ai/) and TRL's `SFTTrainer`.

Training notebook: `instruction-tuning/finetuning_qwen/qwen-3b-sft-tulu.ipynb` in
[Francesck94/smol_course](https://github.com/Francesck94/smol_course).

## Training data

- **Dataset:** `allenai/tulu-3-sft-mixture` (~939k multi-turn conversations in `{"role", "content"}` format).
- **Sampling:** 20,000 random examples, split 80/20 into 16,000 train / 4,000 validation (`seed=42`).
- **System prompt:** a fixed system prompt is prepended to every conversation (helpful, honest,
  concise; says "I don't know" rather than guessing; refuses illegal or harmful requests).
- **Rendering:** conversations normalized with Unsloth's `standardize_sharegpt`, then rendered to a
  single `text` field with the `qwen-3` chat template.

## Training procedure

- **Method:** parameter-efficient fine-tuning with LoRA on a 4-bit quantized base model.
- **Loss masking:** `train_on_responses_only` — loss is computed only on assistant turns;
  system/user turns are masked out.
- **Frameworks:** Unsloth `FastLanguageModel`, TRL `SFTTrainer`, `DataCollatorForSeq2Seq`.
- **Tracking:** Weights & Biases (project `Unsloth-FineTuning`).

### Hyperparameters

| Parameter | Value |
|---|---|
| Base model | `unsloth/Qwen3-1.7B-Base` |
| Load in 4-bit | `True` |
| `max_seq_length` | 2048 |
| LoRA `r` / `alpha` / dropout | 64 / 64 / 0 |
| LoRA target modules | `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj` |
| Gradient checkpointing | `"unsloth"` |
| `learning_rate` | 5e-5 |
| `weight_decay` | 0.001 |
| LR scheduler | linear |
| Optimizer | `adamw_8bit` |
| `max_steps` | 100 (quick run; use `num_train_epochs=1` for a full pass) |
| Warmup steps | 5 |
| Per-device train batch size | 2 |
| Gradient accumulation | 4 (effective batch size 8) |
| Eval | every 10 steps on the validation split |
| Precision | bf16 if supported, else fp16 |
| Seed | 3407 |

### Hardware & runtime

Trained on Kaggle (2× Tesla T4, 14.5 GB VRAM each; model fits on a single GPU), CUDA 12.8,
torch 2.11, Unsloth 2026.10.8, transformers 4.56.2, trl 0.22.2.

## Evaluation results

Evaluated with **lm-evaluation-harness 0.4.13** (full raw outputs in `sft_results/`).

| Benchmark | Metric | Score | Stderr | Samples |
|---|---|---|---|---|
| **MMLU** | acc | **55.30%** | ±0.40 | 14,042 |
| &nbsp;&nbsp;MMLU — STEM | acc | 52.55% | ±0.87 | 3,153 |
| &nbsp;&nbsp;MMLU — Humanities | acc | 48.03% | ±0.69 | 4,705 |
| &nbsp;&nbsp;MMLU — Social Sciences | acc | 64.19% | ±0.85 | 3,077 |
| &nbsp;&nbsp;MMLU — Other | acc | 60.28% | ±0.86 | 3,107 |
| **GSM8K** | exact match (strict) | **63.46%** | ±1.33 | 1,319 |
| &nbsp;&nbsp;GSM8K | exact match (flexible extract) | 63.15% | ±1.33 | 1,319 |
| **TruthfulQA MC2** | acc | **52.68%** | ±1.50 | 817 |

## Inference

```python
messages = [{"role": "user", "content": "Resolve this equation: 2x+1=0"}]
inputs = tokenizer.apply_chat_template(
    messages, tokenize=True, add_generation_prompt=True,
    return_tensors="pt", return_dict=True,
).to("cuda")
outputs = model.generate(**inputs, max_new_tokens=128, use_cache=True,
                         temperature=0.7, min_p=0.1)  # min-p sampling
```

Recommended sampling: `temperature=0.7`, `min_p=0.1`.

## Limitations

- Quick run (100 steps, 20k examples): not a full pass over Tulu-3 — expect further gains from a
  full-epoch training.
- The base model is a 1.7B base (not instruction) model; outputs may still ramble or refuse poorly
  on edge cases outside the training distribution.
- Not evaluated for safety beyond TruthfulQA; the system-prompt refusal behavior is a training
  convention, not a guarantee.

## License

Apache-2.0 (inherited from Qwen3). Notebook based on Unsloth notebooks (LGPL-3.0).
