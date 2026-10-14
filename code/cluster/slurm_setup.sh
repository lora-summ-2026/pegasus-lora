#!/bin/bash
# ONE-TIME setup job for the SLURM cluster.
# Submit from ~/pegasus_lora:      sbatch code/cluster/slurm_setup.sh
# Log:                             ~/pegasus_lora/logs/slurm_setup_<jobid>.out
#SBATCH --job-name=pl_setup
#SBATCH --output=logs/slurm_%x_%j.out
#SBATCH --time=0-02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --gres=gpu:1
set -eo pipefail
source "$HOME/pegasus_lora/code/cluster/env.sh"
source "$HOME/pegasus_lora/code/cluster/conda_init.sh"
echo "== setup on $(hostname) at $(date) =="
nvidia-smi --query-gpu=name,memory.total --format=csv || true

if ! command -v conda >/dev/null 2>&1; then
  echo "== no conda found: installing Miniconda into ~/miniconda3 =="
  curl -fsSL -o /tmp/miniconda_$$.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
  bash /tmp/miniconda_$$.sh -b -p "$HOME/miniconda3"
  rm -f /tmp/miniconda_$$.sh
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
fi
echo "conda: $(command -v conda)"

if conda env list > /tmp/envs_$$.txt && grep -q "^$CONDA_ENV_NAME " /tmp/envs_$$.txt; then
  echo "== environment $CONDA_ENV_NAME already exists =="
else
  echo "== creating environment $CONDA_ENV_NAME (python 3.11) =="
  conda create -y -n "$CONDA_ENV_NAME" --override-channels -c conda-forge python=3.11
fi
conda activate "$CONDA_ENV_NAME"
pip install --upgrade pip
pip install -r "$PROJ/code/requirements.txt"

cd "$PROJ/code"
python prefetch.py
python - << 'PY'
import torch, transformers, peft, datasets
print(f"torch {torch.__version__} | transformers {transformers.__version__} | peft {peft.__version__}")
ok = torch.cuda.is_available()
print("GPU:", torch.cuda.get_device_name(0) if ok else "none", "| bf16:", ok and torch.cuda.is_bf16_supported())
PY
echo "SETUP COMPLETE"
