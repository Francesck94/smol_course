# smol_course

Practical exercises from the [Hugging Face Smol Course](https://github.com/huggingface/smol-course) — a hands-on course for fine-tuning and aligning small language models.

## Repository Structure

```
├── instruction-tuning/                # Supervised fine-tuning (SFT)
│   ├── sft-notebook.ipynb             # SFT on SmolLM3-3B (pushes to Hub)
│   ├── lighteval-colab.ipynb          # LightEval evaluation on Colab
│   ├── setup-sft.ipynb                # Chat-template basics
│   ├── benchmark-sft-model.ipynb      # Benchmark SFT'd Qwen3-1.7B (base vs adapter)
│   ├── finetuning_gemma/              # SFT on Gemma
│   │   └── gemma-sft.ipynb
│   └── finetuning_qwen/               # SFT on Qwen
│       ├── qwen-sft.ipynb             # SFT on Qwen3-1.7B-Base
│       ├── qwen-finetuning.ipynb      # SFT on Qwen3-1.7B-Base (FineTome-100k, Unsloth)
│       └── qwen-3b-sft-tulu.ipynb     # SFT on Qwen3-1.7B-Base (Tulu-3 mixture, Unsloth)
├── sft-data-creation/                 # Dataset creation for SFT
│   ├── create-dataset.ipynb           # Builds sft_data/ and gemma_data/ (local, gitignored)
│   └── instruction-sft-dataset.ipynb  # distilabel synthetic-data exercise
├── preference-alignment/              # Preference alignment
│   ├── orpo-finetuning-example.ipynb  # ORPO fine-tuning example
│   └── qwen-dpo-fine-tune.ipynb       # DPO on the SFT'd Qwen3-1.7B (UltraFeedback)
├── reasoning-training/                # Reasoning training (GRPO)
│   ├── qwen-grpo.ipynb                # GRPO on Qwen3 (OpenR1 Math dataset)
│   ├── qwen-grpo-v2.ipynb             # Second GRPO iteration
│   └── benchmark-grpo-model.ipynb     # Benchmark the GRPO model
├── AGENTS.md                          # Agent guidance / repo conventions
└── kaggle-run-and-download.md         # Kaggle CLI runbook
```

### Qwen3 fine-tuning pipeline

The Qwen notebooks form an end-to-end pipeline (designed for Colab/Kaggle with Unsloth):

1. **SFT** — [qwen-3b-sft-tulu.ipynb](instruction-tuning/finetuning_qwen/qwen-3b-sft-tulu.ipynb) fine-tunes `unsloth/Qwen3-1.7B-Base` on the Tulu-3 SFT mixture; [qwen-finetuning.ipynb](instruction-tuning/finetuning_qwen/qwen-finetuning.ipynb) does the same with FineTome-100k.
2. **DPO** — [qwen-dpo-fine-tune.ipynb](preference-alignment/qwen-dpo-fine-tune.ipynb) aligns the SFT checkpoint with UltraFeedback preferences.
3. **GRPO** — [qwen-grpo.ipynb](reasoning-training/qwen-grpo.ipynb) trains reasoning on the DPO model with OpenR1's Math dataset.

## Requirements

- Python >= 3.13
