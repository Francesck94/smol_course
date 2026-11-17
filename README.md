# smol_course

Practical exercises from the [Hugging Face Smol Course](https://github.com/huggingface/smol-course) — a hands-on course for fine-tuning and aligning small language models.

## Repository Structure

```
├── instruction-tuning/                     # Supervised fine-tuning (SFT)
│   ├── smol_course_notebook/               # Original smol-course exercises
│   │   ├── sft-notebook.ipynb              # SFT on SmolLM3-3B (pushes to Hub)
│   │   ├── lighteval-colab.ipynb           # LightEval evaluation on Colab
│   │   └── setup-sft.ipynb                 # Chat-template basics
│   ├── finetuning_gemma/                   # SFT on Gemma
│   │   └── gemma-sft.ipynb
│   └── finetuning_qwen/                    # SFT on Qwen
│       ├── qwen-3-1.7b-sft-tulu.ipynb      # SFT on Qwen3-1.7B-Base (Tulu-3 mixture, Unsloth)
│       ├── qwen-finetuning.ipynb           # SFT on Qwen3-1.7B-Base (FineTome-100k, Unsloth)
│       ├── qwen-sft.ipynb                  # SFT on Qwen3-1.7B-Base
│       ├── benchmark-sft-model.ipynb       # lm-eval benchmark of the SFT model
│       └── model-card.md                   # Hub model card (SFT)
├── sft-data-creation/                      # Dataset creation for SFT
│   ├── create-dataset.ipynb                # Builds sft_data/ and gemma_data/ (local, gitignored)
│   └── instruction-sft-dataset.ipynb       # distilabel synthetic-data exercise
├── preference-alignment/                   # Preference alignment
│   ├── orpo-finetuning-example.ipynb       # ORPO fine-tuning example
│   ├── qwen-3-1.7b-dpo-fine-tune.ipynb     # DPO on the SFT'd Qwen3-1.7B (UltraFeedback)
│   ├── benchmark-dpo-model.ipynb           # lm-eval benchmark of the DPO model
│   └── model-card.md                       # Hub model card (DPO)
├── reasoning-training/                     # Reasoning training (GRPO)
│   ├── qwen-grpo.ipynb                     # GRPO v1 (math + code)
│   ├── qwen-grpo-v2.ipynb                  # GRPO v2 (math + code)
│   ├── qwen-grpo-v2-only-math.ipynb        # GRPO v2, math only (best run)
│   ├── qwen-grpo-v3-only-math.ipynb        # GRPO v3, math only
│   ├── benchmark-grpo-model.ipynb          # lm-eval benchmark of the GRPO models
│   ├── grpo-training-summary.md            # GRPO runs, rewards and caveats
│   └── model-card-only-math.md             # Hub model card (GRPO)
├── docs/
│   └── kaggle-run-and-download.md          # Kaggle CLI runbook
└── AGENTS.md                               # Agent guidance / repo conventions
```

### Qwen3 fine-tuning pipeline

The Qwen notebooks form an end-to-end SFT → DPO → GRPO pipeline on `unsloth/Qwen3-1.7B-Base` (run on Kaggle with Unsloth, LoRA on a 4-bit base, merged to 16-bit):

1. **SFT** — [qwen-3-1.7b-sft-tulu.ipynb](instruction-tuning/finetuning_qwen/qwen-3-1.7b-sft-tulu.ipynb) fine-tunes the base model on a 20k subset of `allenai/tulu-3-sft-mixture` (100 steps, LoRA r=64).
2. **DPO** — [qwen-3-1.7b-dpo-fine-tune.ipynb](preference-alignment/qwen-3-1.7b-dpo-fine-tune.ipynb) aligns the SFT checkpoint on 20k pairs from `HuggingFaceH4/ultrafeedback_binarized` (1 epoch, beta=0.1).
3. **GRPO** — [qwen-grpo-v2-only-math.ipynb](reasoning-training/qwen-grpo-v2-only-math.ipynb) primes the `<think>…</think><answer>…</answer>` format with a short SFT, then runs 200 GRPO steps on `open-r1/DAPO-Math-17k-Processed`.

#### Results

Evaluated with lm-evaluation-harness (chat template). Scores in %.

| Stage | GSM8K strict | GSM8K flexible | MMLU | TruthfulQA MC2 |
|---|---|---|---|---|
| SFT (Tulu-3) | 63.5 | 63.2 | 55.3 | 52.7 |
| DPO (UltraFeedback) | 68.0 | 64.5 | 58.2 | **53.2** |
| GRPO v2, math + code | 55.2 | 60.5 | 57.9 | 49.3 |
| **GRPO v2, math only** | **70.0** | **68.7** | **58.4** | 49.9 |

DPO improved every benchmark over SFT. Math-only GRPO added +2.0 (strict) / +4.2 (flexible) on GSM8K over DPO and held MMLU, at the cost of ~3 points on TruthfulQA. Mixing code into GRPO hurt GSM8K: most completions hit the length limit before closing the answer tag. Details in the model cards and [grpo-training-summary.md](reasoning-training/grpo-training-summary.md).

#### Models on the Hugging Face Hub

| Stage | 16-bit (Transformers) | GGUF (llama.cpp) | Model card |
|---|---|---|---|
| SFT | [Qwen3-1.7B-sft-tulu-16bit](https://huggingface.co/FrancescoArno94/Qwen3-1.7B-sft-tulu-16bit) | [Qwen3-1.7B-sft-tulu-16bit-GGUF](https://huggingface.co/FrancescoArno94/Qwen3-1.7B-sft-tulu-16bit-GGUF) | [model-card.md](instruction-tuning/finetuning_qwen/model-card.md) |
| DPO | [Qwen3-1.7B-dpo-ultrafeedback_binarized-16bit](https://huggingface.co/FrancescoArno94/Qwen3-1.7B-dpo-ultrafeedback_binarized-16bit) | [Qwen3-1.7B-dpo-ultrafeedback_binarized-16bit-gguf](https://huggingface.co/FrancescoArno94/Qwen3-1.7B-dpo-ultrafeedback_binarized-16bit-gguf) | [model-card.md](preference-alignment/model-card.md) |
| GRPO (math) | [Qwen3-1.7B-GRPO-16bit-DAPO-math](https://huggingface.co/FrancescoArno94/Qwen3-1.7B-GRPO-16bit-DAPO-math) | [Qwen3-1.7B-GRPO-16bit-DAPO-math-gguf](https://huggingface.co/FrancescoArno94/Qwen3-1.7B-GRPO-16bit-DAPO-math-gguf) | [model-card-only-math.md](reasoning-training/model-card-only-math.md) |

## Requirements

- Python >= 3.13
