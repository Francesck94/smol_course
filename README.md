# smol_course

Practical exercises from the [Hugging Face Smol Course](https://github.com/huggingface/smol-course) — a hands-on course for fine-tuning and aligning small language models.

## Repository Structure

```
├── instruction_tuning/               # Supervised fine-tuning (SFT)
│   ├── sft_notebook.ipynb            # General SFT notebook
│   ├── lighteval_colab.ipynb         # LightEval evaluation on Colab
│   ├── setup_sft.ipynb               # Environment setup for SFT
│   ├── finetuning_gemma/             # SFT on Gemma
│   │   └── gemma-sft.ipynb
│   └── finetuning_qwen/              # SFT on Qwen
│       ├── qwen-sft.ipynb            # SFT on Qwen3-1.7B-Base
│       ├── qwen-finetuning.ipynb     # SFT on Qwen3-1.7B-Base (FineTome-100k, Unsloth)
│       ├── qwen-3b-sft-tulu.ipynb    # SFT on Qwen3-1.7B-Base (Tulu-3 mixture, Unsloth)
│       └── download_sft_results.ipynb
├── sft_data_creation/                # Dataset creation for SFT
│   ├── create_dataset.ipynb
│   └── instruction_sft_dataset.ipynb
├── preference_alignment/             # Preference alignment
│   ├── orpo_finetuning_example.ipynb # ORPO fine-tuning example
│   └── qwen-dpo-fine-tune.ipynb      # DPO on the SFT'd Qwen3-1.7B (UltraFeedback)
└── qwen-reasoning-training/          # Reasoning training
    └── qwen-grpo.ipynb               # GRPO on Qwen3 (OpenR1 Math dataset)
```

### Qwen3 fine-tuning pipeline

The Qwen notebooks form an end-to-end pipeline (designed for Colab/Kaggle with Unsloth):

1. **SFT** — [qwen-3b-sft-tulu.ipynb](instruction_tuning/finetuning_qwen/qwen-3b-sft-tulu.ipynb) fine-tunes `unsloth/Qwen3-1.7B-Base` on the Tulu-3 SFT mixture; [qwen-finetuning.ipynb](instruction_tuning/finetuning_qwen/qwen-finetuning.ipynb) does the same with FineTome-100k.
2. **DPO** — [qwen-dpo-fine-tune.ipynb](preference_alignment/qwen-dpo-fine-tune.ipynb) aligns the SFT checkpoint with UltraFeedback preferences.
3. **GRPO** — [qwen-grpo.ipynb](qwen-reasoning-training/qwen-grpo.ipynb) trains reasoning on the DPO model with OpenR1's Math dataset.

## Requirements

- Python >= 3.10
