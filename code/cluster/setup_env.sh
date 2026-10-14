#!/usr/bin/env bash
# ONE-TIME setup. Run inside the terminal of a Jupyter workspace on the cluster
# (conda is not available on the login node bhn20):
#     bash ~/pegasus_lora/code/cluster/setup_env.sh
set -eo pipefail
source "$(dirname "$0")/env.sh"
eval "$(conda shell.bash hook)"

echo "== 1/4  conda: keep environments in ~/env =="
conda config --prepend envs_dirs ~/env 2>/dev/null || true
conda config --show envs_dirs

echo "== 2/4  create environment '$CONDA_ENV_NAME' (python 3.11) =="
if conda env list > /tmp/envs_$$.txt && grep -q "^$CONDA_ENV_NAME " /tmp/envs_$$.txt; then
  echo "already exists - skipping creation"
else
  conda create -y -n "$CONDA_ENV_NAME" --override-channels -c conda-forge python=3.11
fi
conda activate "$CONDA_ENV_NAME"

echo "== 3/4  install packages (~5 min) =="
pip install --upgrade pip
pip install -r "$PROJ/code/requirements.txt"

echo "== 4/4  download model, datasets and tokenizer data into $PROJ (~10 min) =="
cd "$PROJ/code"
python prefetch.py

python - << 'PY'
import torch, transformers, peft, datasets
print(f"torch {torch.__version__} | transformers {transformers.__version__} | peft {peft.__version__} | datasets {datasets.__version__}")
print("CUDA visible here:", torch.cuda.is_available(), "(a CPU-only workspace is fine for setup)")
PY
echo "SETUP COMPLETE. Next: from bhn20 run  bash ~/pegasus_lora/code/cluster/submit.sh smoke"
