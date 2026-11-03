# AGENTS.md

Personal working copy of exercises from the Hugging Face [smol-course](https://github.com/huggingface/smol-course). **Notebooks only** — there is no Python package, no tests, no CI (`.github/` is empty), and no lint/typecheck config. Do not invent a test/build step.

## Environment

- `pyproject.toml` declares only `kaggle>=1.7.4.5` (for the CLI runbook in `kaggle-run-and-download.md`). Real ML deps live in `.venv/` (uv-created, CPython 3.13) and in `!pip`/`!uv pip` cells inside the notebooks. `uv.lock` exists and is tracked; do not "fix" pyproject by guessing pins.
- Installed in `.venv`: torch 2.10, transformers 4.57, trl 0.24, datasets 4.3, peft, unsloth, bitsandbytes, ipykernel. Use `.venv/bin/python` rather than creating a new env.
- Local machine is macOS/Apple Silicon. Most notebooks are written for **CUDA + Unsloth** (Colab/Kaggle: `torch.cuda.get_device_properties(0)`, `.to("cuda")`, `kaggle_secrets`). They will not run locally as-is. Only `preference-alignment/orpo-finetuning-example.ipynb` has a cuda/mps/cpu fallback.
- Secrets come from notebook-host secret stores (`user_secrets.get_secret("HF_TOKEN")`, `"wandb-key"`) or `os.getenv("HF_TOKEN")`. Never hardcode tokens; several cells still contain `"YOUR_HF_TOKEN"` placeholders on purpose.

## Layout

Folders and notebooks are kebab-case; `finetuning_qwen/` and `finetuning_gemma/` subfolders keep snake_case.

- `instruction-tuning/` — SFT work: `sft-notebook.ipynb` (SmolLM3-3B, pushes to `FrancescoArno94/SmolLM3-3B-math`), `finetuning_qwen/` (Qwen3-1.7B-Base), `finetuning_gemma/` (gemma-4-E2B), `lighteval-colab.ipynb` (LightEval + vLLM eval of the pushed model), `setup-sft.ipynb` (chat-template basics), `benchmark-sft-model.ipynb`.
- `sft-data-creation/` — dataset building; `create-dataset.ipynb` writes `sft_data/` (HF `save_to_disk` arrow output, gitignored) and `../gemma_data/gemma_sft_dataset`. `instruction-sft-dataset.ipynb` uses distilabel.
- `preference-alignment/` — `orpo-finetuning-example.ipynb` (ORPO on SmolLM2-135M), `qwen-dpo-fine-tune.ipynb` (DPO on the SFT'd Qwen3-1.7B).
- `reasoning-training/` — GRPO on Qwen3: `qwen-grpo.ipynb`, `qwen-grpo-v2.ipynb`, `benchmark-grpo-model.ipynb`.
- `download-sft-results.ipynb` (in `instruction-tuning/finetuning_qwen/`) is gitignored — local Kaggle-results puller, not in the repo.

## Gotchas

- Training outputs and generated data (`outputs/`, `results/`, `eval_results/`, `lora_model/`, `sft_data/`, `gemma_data/`, `sft_results/`) are gitignored — keep them uncommitted.
- `.DS_Store` files are ignored and untracked since the repo-reorganization cleanup; never stage them.
- Notebooks are committed with execution outputs; keep diffs limited to cells you actually changed rather than re-running everything.
