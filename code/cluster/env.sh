# Shared environment for every step on the HPC cluster. Sourced by the other scripts.
# Everything lives under ~/pegasus_lora (the home directory is shared between bhn20 and all jobs).
export PROJ="${PROJ:-$HOME/pegasus_lora}"
export OUT="$PROJ/runs"                  # all experiment outputs
export HF_HOME="$PROJ/hf_cache"          # model + datasets cache (~6 GB)
export NLTK_DATA="$PROJ/nltk_data"
export CONDA_ENV_NAME="pegasus_lora"
export TOKENIZERS_PARALLELISM=false
mkdir -p "$OUT" "$HF_HOME" "$NLTK_DATA" "$PROJ/logs"
# If jobs fail with connection errors, compute nodes have no internet:
# run prefetch.py once from the Jupyter workspace and submit jobs with OFFLINE=1.
if [ "${OFFLINE:-0}" = "1" ]; then
  export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
fi
