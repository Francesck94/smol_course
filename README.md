# smol_course

Practical exercises from the [Hugging Face Smol Course](https://github.com/huggingface/smol-course) — a hands-on course for fine-tuning and aligning small language models.

## Repository Structure

```
├── instruction_tuning/          # Supervised fine-tuning (SFT)
│   ├── sft_notebook.ipynb       # General SFT notebook
│   ├── lighteval_colab.ipynb    # LightEval evaluation on Colab
│   ├── finetuning_gemma/        # SFT on Gemma
│   └── finetuning_qwen/         # SFT on Qwen
├── sft_data_creation/           # Dataset creation for SFT
│   └── create_dataset.ipynb
├── preference_alignment/        # Preference alignment (WIP)
├── setup_sft.ipynb              # Environment setup for SFT
├── instruction_sft_dataset.ipynb
└── Copia_di_orpo_finetuning_example.ipynb  # ORPO fine-tuning example
```

## Requirements

- Python >= 3.10
