---
license: apache-2.0
base_model: FrancescoArno94/Qwen3-1.7B-sft-tulu-16bit
datasets:
  - HuggingFaceH4/ultrafeedback_binarized
tags:
  - unsloth
  - lora
  - qwen3
  - dpo
  - preference-alignment
library_name: transformers
pipeline_tag: text-generation
---

# Qwen3-1.7B DPO on UltraFeedback

Direct Preference Optimization (DPO) on top of the SFT checkpoint
[`FrancescoArno94/Qwen3-1.7B-sft-tulu-16bit`](https://huggingface.co/FrancescoArno94/Qwen3-1.7B-sft-tulu-16bit)
(itself a LoRA fine-tune of [`unsloth/Qwen3-1.7B-Base`](https://huggingface.co/unsloth/Qwen3-1.7B-Base)
on a 20k subset of `allenai/tulu-3-sft-mixture`), trained with
[Unsloth](https://unsloth.ai/) and TRL's `DPOTrainer`.

Training notebook: `preference-alignment/qwen-3-1.7b-dpo-fine-tune.ipynb` in
[Francesck94/smol_course](https://github.com/Francesck94/smol_course).

## Training data

- **Dataset:** [`HuggingFaceH4/ultrafeedback_binarized`](https://huggingface.co/datasets/HuggingFaceH4/ultrafeedback_binarized)
  (`train_prefs` / `test_prefs` splits).
- **Preprocessing:**
  1. Drop ties (`score_chosen == score_rejected`) — no preference signal.
  2. Subsample 20,000 training examples to bound run time; filtered `test_prefs` used as eval set.
  3. Split into `prompt` (all turns before the final reply — multi-turn safe) / `chosen` /
     `rejected` (final assistant message).
  4. Render to text with the Qwen-3 chat template, appending EOS to the responses.

## Training procedure

- **Method:** DPO with LoRA adapters on the 4-bit quantized SFT checkpoint; the SFT weights stay
  frozen and serve as the reference policy (`ref_model=None`).
- **Frameworks:** Unsloth `FastLanguageModel` + `PatchDPOTrainer`, TRL `DPOTrainer`.
- **Tracking:** Weights & Biases (project `Unsloth-DPOFineTuning`; training resumed from checkpoint).

### Hyperparameters

| Parameter | Value |
|---|---|
| Base model | `FrancescoArno94/Qwen3-1.7B-sft-tulu-16bit` |
| Load in 4-bit | `True` |
| `max_seq_length` | 2048 |
| LoRA `r` / `alpha` / dropout | 64 / 64 / 0 |
| LoRA target modules | `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj` |
| Gradient checkpointing | `"unsloth"` |
| DPO `beta` | 0.1 |
| `learning_rate` | 5e-6 |
| LR scheduler / warmup | linear / 10% warmup ratio |
| Optimizer | `adamw_8bit` |
| `num_train_epochs` | 1 |
| Per-device train batch size | 2 |
| Gradient accumulation | 4 (effective batch size 8) |
| `max_length` / `max_prompt_length` | 1024 / 512 |
| Eval | every 100 steps on the filtered `test_prefs` split |
| `weight_decay` | 0.0 |
| Seed | 42 |

### Hardware & runtime

Trained on Kaggle (single GPU, CUDA 12.8), transformers 4.56.2, trl 0.22.2, Unsloth.

## Evaluation results

Evaluated with **lm-evaluation-harness** (full raw outputs in `preference-alignment/results/`).
The SFT baseline column is the pre-DPO checkpoint from the same pipeline.

| Benchmark | Metric | SFT baseline | **DPO** | Δ |
|---|---|---|---|---|
| **MMLU** | acc | 55.30% | **58.16%** | +2.86 |
| &nbsp;&nbsp;MMLU — STEM | acc | 52.55% | 56.39% | +3.84 |
| &nbsp;&nbsp;MMLU — Humanities | acc | 48.03% | 48.95% | +0.92 |
| &nbsp;&nbsp;MMLU — Social Sciences | acc | 64.19% | 69.09% | +4.90 |
| &nbsp;&nbsp;MMLU — Other | acc | 60.28% | 63.08% | +2.80 |
| **GSM8K** | exact match (strict) | 63.46% | **68.01%** | +4.55 |
| &nbsp;&nbsp;GSM8K | exact match (flexible extract) | 63.15% | 64.52% | +1.37 |
| **TruthfulQA MC2** | acc | 52.68% | **53.17%** | +0.49 |

Sample counts: MMLU 14,042; GSM8K 1,319; TruthfulQA MC2 817.

## Inference

```python
messages = [{"role": "user", "content": "Resolve this equation: 2x+1=0"}]
inputs = tokenizer.apply_chat_template(
    messages, tokenize=True, add_generation_prompt=True,
    return_tensors="pt", return_dict=True,
).to("cuda")
outputs = model.generate(**inputs, max_new_tokens=256, use_cache=True,
                         temperature=0.7, min_p=0.1)  # min-p sampling
```

Recommended sampling: `temperature=0.7`, `min_p=0.1`.

## Limitations

- 20k-subsample, single-epoch DPO run — not a full pass over UltraFeedback.
- Preference signal comes from GPT-4-annotated UltraFeedback ratings; biases of the annotation
  pipeline may carry over.
- Safety/bias behavior was probed qualitatively (harmful and bias-related prompts) but not
  systematically; benchmark gains on MMLU/GSM8K likely reflect continued optimization rather than
  pure alignment.

## License

Apache-2.0 (inherited from Qwen3). Notebook based on Unsloth notebooks (LGPL-3.0).
