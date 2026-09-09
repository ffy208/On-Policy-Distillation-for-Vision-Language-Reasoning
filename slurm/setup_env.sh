#!/usr/bin/env bash
# One-time environment setup on PACE (run on a login node). Adjust the module names to what
# `module avail` shows on your cluster. Installs into a uv virtualenv under $HOME.
set -euo pipefail

REPO_DIR="${REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"   # the repo this script lives in
ENV_DIR="${ENV_DIR:-$HOME/vlm_opd_env}"
export HF_HOME="${HF_HOME:-${SCRATCH:-$HOME/scratch}/hf_cache}"   # models and datasets go to scratch, not home quota

module load python/3.12.5 2>/dev/null || module load python 2>/dev/null || true
module load cuda/12.6.1 2>/dev/null || module load cuda 2>/dev/null || true

command -v uv >/dev/null 2>&1 || pip install --user uv
uv venv "$ENV_DIR" --python 3.12
# shellcheck disable=SC1091
source "$ENV_DIR/bin/activate"
cd "$REPO_DIR"
uv pip install -e ".[dev]"
# vLLM pins its own torch build; install it first so torch matches the CUDA runtime it was built for
uv pip install "vllm>=0.8" "transformers>=4.57" "peft>=0.13" "accelerate>=1.0" qwen-vl-utils matplotlib
mkdir -p "$HF_HOME" logs
python - <<'PY'
import torch, transformers, vllm, peft
print("torch", torch.__version__, "cuda", torch.version.cuda, "| transformers", transformers.__version__, "| vllm", vllm.__version__, "| peft", peft.__version__)
PY
echo "Environment ready at $ENV_DIR (HF_HOME=$HF_HOME)"
echo "Put your HuggingFace token in ~/.hf_token (chmod 600); the job scripts export it as HF_TOKEN."
