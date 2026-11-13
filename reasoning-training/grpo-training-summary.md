# GRPO Training Summary: Qwen3-1.7B (SFT → DPO → GRPO)

The final math-only GRPO run beat the DPO model it started from on GSM8K (+2.0 pts strict, +4.2 pts flexible) and held MMLU level. TruthfulQA dropped by about 3 points. The earlier run that mixed math and code made GSM8K worse.

## Pipeline

- **Starting model:** `FrancescoArno94/Qwen3-1.7B-dpo-ultrafeedback_binarized-16bit`, the output of the DPO stage.
- **Setup:** Unsloth with vLLM generation, loaded in 4-bit. LoRA rank 32 (alpha 64) on all attention and MLP projections — about 34.9M trainable parameters (3.3% of the model). Max sequence length 2048. Ran on Kaggle.
- **Output format:** `<think>…</think><answer>…</answer>`, set by a custom chat template and system prompt.

### Stage 1: Format priming (SFT)

- **Data:** about 64 short, numeric-answer examples from `unsloth/OpenMathReasoning-mini`, rewritten into the output format above.
- **Settings:** 100 steps, learning rate 2e-5, effective batch size 4.
- **Result:** loss fell from 0.98 to 0.12 in about 4 minutes, and the model learned the tag format.

### Stage 2: GRPO

- **Settings:** learning rate 5e-6 with cosine schedule and 10% warmup, AdamW 8-bit. Batch size 4 with 4 generations per prompt. Sampling at temperature 1.0 with min_p 0.1.
- **Loss:** Unsloth's default `dapo` loss with beta 0, so KL is always 0 and there is no reference-model penalty.
- **Prompt length:** the longest 10% of prompts were removed.
- **Reward functions:**
  - `match_format_exactly`: +3 for the exact format.
  - `match_format_approximately`: ±0.5 per closing tag.
  - `check_answer`: +5 for an exact match, partial credit if within 10–20%, down to −4.5.
  - `check_numbers`: +3.5 or −1.5 based on the parsed number.
  - The math+code runs also used `code_reward` and `code_format_reward`.

## Run variants

| Notebook | Data | Steps | Notes |
|---|---|---|---|
| `qwen-grpo.ipynb` | DAPO-Math-17k + verifiable Python coding (44,868 prompts) | 100 (config says 300; the run log says 100) | Only run with saved reward logs |
| `qwen-grpo-v2.ipynb` | Math + code | 50 | Cosine schedule, Adam beta2 0.99 |
| `qwen-grpo-v2-only-math.ipynb` | DAPO-Math-17k only (14,116 prompts) | 200 | `enforce_eager=True` (CUDA graphs off) to work around a vLLM crash on the T4 GPU; saved as `qwen_grpo_16bit` |

### What the v1 reward logs show (32 logged steps)

- Mean reward stayed around −3.7 and improved only slightly (−3.88 → −3.53).
- 75–93% of completions hit the length limit (about 1241 tokens) before closing `</answer>`.
- Because of that, `match_format_exactly` was always 0 and the answer extractors usually returned `None`.
- The code rewards stayed near 0.

Long math and code reasoning traces didn't fit in the completion budget, which is likely why dropping code helped.

## Benchmarks (lm-eval with chat template)

| Model | GSM8K strict | GSM8K flexible | MMLU | TruthfulQA MC2 |
|---|---|---|---|---|
| SFT | 63.5 | 63.2 | 55.3 | 52.7 |
| DPO (GRPO starting point) | 68.0 | 64.5 | 58.2 | **53.2** |
| GRPO v2, math+code\* | 55.2 | 60.5 | 57.9 | 49.3 |
| GRPO only-math (Nov 11) | 56.6 | 62.5 | 57.8 | 49.8 |
| **GRPO only-math (Nov 12)** | **70.0** | **68.7** | **58.4** | 49.9 |

\*The file `results/qwen3_grpo_v2_results.json` doesn't say which model it evaluated; it was matched to the math+code run by its name.

## Caveats

- **Nov 11 vs Nov 12:** both runs evaluated `/kaggle/input/qwen-grpo-v2-only-math/qwen_grpo_16bit`. The notebook was edited at 17:32 on Nov 11, after the first eval, so the Nov 12 result probably reflects that updated version. Nothing records exactly what changed.
- **Missing logs:** the only-math notebook has no saved training output, so there are no reward curves or loss values for the best run. To get them, download the Kaggle run output or set `report_to="wandb"` next time.
