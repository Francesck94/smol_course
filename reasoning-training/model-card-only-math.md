---
license: apache-2.0
base_model: FrancescoArno94/Qwen3-1.7B-dpo-ultrafeedback_binarized-16bit
datasets:
  - unsloth/OpenMathReasoning-mini
  - open-r1/DAPO-Math-17k-Processed
tags:
  - qwen3
  - grpo
  - reasoning
  - math
  - lora
  - unsloth
library_name: transformers
pipeline_tag: text-generation
---

# Qwen3-1.7B GRPO: Math Only

This experimental reasoning model is fine-tuned from
[`FrancescoArno94/Qwen3-1.7B-dpo-ultrafeedback_binarized-16bit`](https://huggingface.co/FrancescoArno94/Qwen3-1.7B-dpo-ultrafeedback_binarized-16bit).
It was produced by [`qwen-grpo-v2-only-math.ipynb`](./qwen-grpo-v2-only-math.ipynb)
using format-priming supervised fine-tuning followed by GRPO on math prompts.

The notebook saves merged 16-bit weights locally to `qwen_grpo_16bit/`. Its
Hub-upload commands are disabled placeholders; this card describes the saved
artifact and does not imply a particular Hub repository hosts it.

## Intended use

This checkpoint is intended for educational and research experiments in
mathematical problem solving, especially prompts that benefit from a worked
solution and a clearly marked final answer. It is not a general-purpose
assistant and has not been validated for production or high-stakes use.

## Training data

- **Format priming:** a numeric-answer subset of
  [`unsloth/OpenMathReasoning-mini`](https://huggingface.co/datasets/unsloth/OpenMathReasoning-mini)
  (`cot` split), rendered to teach the response format
  `<think>...</think><answer>...</answer>`.
- **GRPO:** the English training split of
  [`open-r1/DAPO-Math-17k-Processed`](https://huggingface.co/datasets/open-r1/DAPO-Math-17k-Processed).
  After preprocessing, the notebook's training summary reports 14,116 prompts;
  the longest 10% by prompt length are removed before training.

## Training procedure

The notebook loads the DPO checkpoint in 4-bit, attaches LoRA adapters, and
exports a merged 16-bit model. It uses Unsloth with vLLM for generation.

| Setting | Notebook configuration |
|---|---|
| Architecture | Qwen3-1.7B |
| Maximum sequence length | 2,048 tokens |
| LoRA rank / alpha | 32 / 64 |
| LoRA target modules | `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj` |
| Format-priming SFT | 100 steps; learning rate `2e-5`; effective batch size 4 |
| GRPO steps | 200 |
| GRPO learning rate | `5e-6` |
| GRPO optimizer / schedule | `adamw_8bit` / cosine; 10% warmup |
| Per-device batch / generations | 4 / 4 |
| Gradient accumulation | 1 |
| GRPO loss | `dapo`, beta 0 |
| Sampling during GRPO | temperature 1.0, `min_p=0.1` |
| Seed | 3407 |
| vLLM setting | `enforce_eager=True` to avoid a CUDA-graph sampler crash on the Kaggle T4 setup |

The math reward functions combine response-format rewards with exact and
partial numeric-answer scoring. A separate numeric check adds reward when the
answer can be parsed and matches the expected value. These rewards are proxies
for answer quality, not proof that the reasoning is valid.

## Evaluation

The following results are from the repository's November 12, 2026
lm-evaluation-harness run with a chat template. The result manifest identifies
the evaluated model as
`/kaggle/input/qwen-grpo-v2-only-math/qwen_grpo_16bit`.

| Benchmark | Metric | Score | Std. error | Samples |
|---|---|---:|---:|---:|
| GSM8K | Exact match, strict | 70.0% | 1.26 pts | 1,319 |
| GSM8K | Exact match, flexible extraction | 68.7% | 1.28 pts | 1,319 |
| MMLU | Accuracy | 58.4% | 0.40 pts | 14,042 |
| TruthfulQA MC2 | Accuracy | 49.9% | 1.49 pts | 817 |

These are single-run benchmark results, not a comprehensive evaluation. The
saved benchmark output identifies the model path, but does not record a model
revision hash; the precise training artifact revision is therefore not
reproducible from the manifest alone.

## Usage

Load the local merged export with Transformers. The notebook's tokenizer
template includes the system instruction for the `<think>` and `<answer>` tags;
the explicit system message below makes the expected format clear:

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "qwen_grpo_16bit"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype="auto",
    device_map="auto",
)

messages = [
    {
        "role": "system",
        "content": (
            "You are given a problem. Think about the problem and provide your "
            "working out. Place it between <think> and </think>. Then, provide "
            "your solution between <answer></answer>."
        ),
    },
    {"role": "user", "content": "What is the square root of 101?"},
]
inputs = tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_tensors="pt",
    return_dict=True,
).to(model.device)

with torch.no_grad():
    output = model.generate(
        **inputs,
        max_new_tokens=512,
        do_sample=True,
        temperature=1.0,
        top_k=50,
    )

print(tokenizer.decode(output[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True))
```

## Limitations and risks

- The model received a short, 200-step GRPO run on a single math dataset. It
  can still produce incorrect answers or plausible but invalid reasoning.
- Reward optimization favors the expected answer and output format. It does
  not verify the correctness or completeness of the generated reasoning.
- Performance on GSM8K, MMLU, and TruthfulQA does not establish broad
  mathematical competence, factual reliability, or safety. No systematic
  bias or safety evaluation is reported.
- Do not use this model as the sole basis for educational grading, financial
  or scientific decisions, or other high-stakes outcomes.

## License

The base Qwen3 checkpoint is distributed under Apache-2.0. Check the terms of
the upstream checkpoints and training datasets before redistributing this
fine-tune.
