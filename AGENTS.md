# AGENTS.md

Personal working copy of exercises from the Hugging Face [smol-course](https://github.com/huggingface/smol-course). **Notebooks only** — there is no Python package, no tests, no CI (`.github/` is empty), and no lint/typecheck config. Do not invent a test/build step.

## Environment

- `pyproject.toml` declares **no dependencies** (`dependencies = []`) — it is a stub. Real deps live in `.venv/` (uv-created, CPython 3.13) and in `!pip`/`!uv pip` cells inside the notebooks. There is no lockfile; do not "fix" pyproject by guessing pins.
- Installed in `.venv`: torch 2.10, transformers 4.57, trl 0.24, datasets 4.3, peft, unsloth, bitsandbytes, ipykernel. Use `.venv/bin/python` rather than creating a new env.
- Local machine is macOS/Apple Silicon. Most notebooks are written for **CUDA + Unsloth** (Colab/Kaggle: `torch.cuda.get_device_properties(0)`, `.to("cuda")`, `kaggle_secrets`). They will not run locally as-is. Only `preference_alignment/orpo_finetuning_example.ipynb` has a cuda/mps/cpu fallback.
- Secrets come from notebook-host secret stores (`user_secrets.get_secret("HF_TOKEN")`, `"wandb-key"`) or `os.getenv("HF_TOKEN")`. Never hardcode tokens; several cells still contain `"YOUR_HF_TOKEN"` placeholders on purpose.

## Layout

- `instruction_tuning/` — SFT work: `sft_notebook.ipynb` (SmolLM3-3B, pushes to `FrancescoArno94/SmolLM3-3B-math`), `finetuning_qwen/` (Qwen3-1.7B-Base), `finetuning_gemma/` (gemma-4-E2B), `lighteval_colab.ipynb` (LightEval + vLLM eval of the pushed model), `setup_sft.ipynb` (chat-template basics).
- `sft_data_creation/` — dataset building; `create_dataset.ipynb` writes `sft_data/` (HF `save_to_disk` arrow output, untracked) and `../gemma_data/gemma_sft_dataset`. `instruction_sft_dataset.ipynb` uses distilabel.
- `preference_alignment/` — ORPO on SmolLM2-135M.
- `setup_sft.ipynb` at repo root is a **stale duplicate** of `instruction_tuning/setup_sft.ipynb` (untracked leftover from commit `5da2d33 re-order repo`). Edit the one under `instruction_tuning/`.

## Gotchas

- `README.md`'s "Repository Structure" block is out of date (lists root-level files that were moved). Update it if you touch structure.
- `.DS_Store` files are **tracked** despite `.gitignore`, so they show up as modified constantly. Never stage them; stage notebook paths explicitly instead of `git add -A`.
- Training outputs (`outputs/`, `./results/`, `eval_results/`, `lora_model/`, `sft_data/`) are not gitignored — do not commit them.
- Notebooks are committed with execution outputs; keep diffs limited to cells you actually changed rather than re-running everything.
