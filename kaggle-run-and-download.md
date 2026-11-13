# Running and downloading a Kaggle notebook from this repo

Commands used for `france94/benchmark-grpo-model`. Replace the kernel slug to reuse them for another notebook (e.g. `france94/qwen-grpo`).

## Prerequisites

- Kaggle CLI ≥ 2.x in `.venv` (1.7.4.5 returned 401 with the `KGAT_` key; upgraded with `uv pip install -U kaggle`).
- Auth: `~/.kaggle/access_token` holds a Kaggle API token. Note: `~/.kaggle/kaggle.json` (with a `KGAT_` key) gave 401; exporting `KAGGLE_API_TOKEN` works:

```bash
export KAGGLE_API_TOKEN=$(cat ~/.kaggle/access_token)
K=./.venv/bin/kaggle
```

### Corporate TLS proxy (SSLCertVerificationError)

This machine sits behind a TLS-intercepting proxy ("SSL Decrypt Certificate Trust"). The venv's certifi bundle doesn't trust it, and Python 3.13's strict X.509 mode rejects the chain ("Missing Authority Key Identifier"). One-time setup:

```bash
# export proxy root from the macOS keychain and append to a certifi bundle
security find-certificate -c "SSL Decrypt Certificate Trust" -p /Library/Keychains/System.keychain > /tmp/corp-root.pem
cat $(./.venv/bin/python -c "import certifi; print(certifi.where())") /tmp/corp-root.pem > /tmp/combined-ca-bundle.pem

# relax VERIFY_X509_STRICT for urllib3 (verification stays ON)
mkdir -p /tmp/kaggle-ssl   # see repo memory or history for sitecustomize.py contents
```

Then prefix every `kaggle` call with:

```bash
export REQUESTS_CA_BUNDLE=/tmp/combined-ca-bundle.pem
export SSL_CERT_FILE=/tmp/combined-ca-bundle.pem
export PYTHONPATH=/tmp/kaggle-ssl
```

## Push the notebook and run it

1. Make a staging folder and copy the notebook plus a `kernel-metadata.json`:

```json
{
  "id": "france94/benchmark-grpo-model",
  "title": "benchmark_grpo_model",
  "code_file": "benchmark-grpo-model.ipynb",
  "language": "python",
  "kernel_type": "notebook",
  "is_private": true,
  "enable_gpu": true,
  "enable_tpu": false,
  "enable_internet": true,
  "keywords": ["gpu"],
  "dataset_sources": [],
  "kernel_sources": [],
  "competition_sources": [],
  "model_sources": [],
  "machine_shape": "NvidiaTeslaT4"
}
```

- `machine_shape`: `NvidiaTeslaT4` (free tier) or `NvidiaTeslaP100`.
- Set `is_private: false` to make it public.
- Enable Internet, or `!pip install` and HF downloads will fail.
- `HF_TOKEN` / `wandb-key`: add them under **Add-ons → Secrets** on the Kaggle page and attach them to the notebook; do not hardcode them.

2. Push (this also starts the run):

```bash
$K kernels push -p <staging-folder>
```

3. Check status and open the run:

```bash
$K kernels status france94/benchmark-grpo-model
# statuses: RUNNING, COMPLETE, ERROR, CANCELED
```

URL: https://www.kaggle.com/code/france94/benchmark-grpo-model

Notes:
- Each push creates a new version and starts a new run. The CLI cannot stop a run; stop it from the notebook page (Save Version / Stop) to save GPU quota.
- Background runs are limited to ~9–12 h; interactive sessions ~20–60 min of inactivity. A full benchmark without `limit` (gsm8k + mmlu + truthfulqa) takes hours.

## Download the notebook and its outputs

```bash
# notebook source only
$K kernels pull france94/benchmark-grpo-model -p ./kpull

# source + metadata (kernel-metadata.json)
$K kernels pull france94/benchmark-grpo-model -p ./kpull -m

# run outputs: everything the kernel wrote to /kaggle/working (e.g. results/, logs)
$K kernels output france94/benchmark-grpo-model -p ./out
```

The evaluation tracker in the notebook writes to `results/`, which appears under `out/` after `kernels output`.

## Useful extras

```bash
$K kernels list --mine            # list your kernels and last run time
$K kernels list -s benchmark      # search kernels
```

For scheduling: use **Save Version → Schedule** on the Kaggle page (daily/weekly/monthly), or wrap `kernels push` in `cron`/`launchd`/GitHub Actions for custom timing.
