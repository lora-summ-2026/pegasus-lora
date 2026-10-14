#!/usr/bin/env bash
# Runs INSIDE a Run:AI training job (started by submit.sh). Do not run it on bhn20.
#   job.sh <target>   target = smoke | samsum | cnn | ablation-samsum | ablation-cnn
#                              | seeds-samsum | seeds-cnn | judge-samsum | judge-cnn | report
set -eo pipefail
TARGET="${1:?usage: job.sh <target>}"
source "$(dirname "$0")/env.sh"
if command -v conda >/dev/null 2>&1; then          # --conda already activates it; this is a safety net
  eval "$(conda shell.bash hook)" && conda activate "$CONDA_ENV_NAME" || true
fi
cd "$PROJ/code"
LOG="$PROJ/logs/${TARGET}_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1
echo "== job $TARGET on $(hostname) at $(date) | log: $LOG =="

if [ "$TARGET" != "report" ]; then
  nvidia-smi --query-gpu=name,memory.total --format=csv || true
  python - << 'PY'
import os, sys, torch
ok = torch.cuda.is_available() and torch.cuda.get_device_capability(0) >= (8, 0)
name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none"
print(f"GPU: {name} | bf16: {ok}")
if not ok and "--allow_fp32" not in os.environ.get("EXTRA_ARGS", ""):
    sys.exit("This GPU has no bf16 support (or no GPU was allocated). Resubmit on a newer GPU "
             "(SLURM: sbatch --gres=gpu:<type>:1 ..., type from `sres`, e.g. rtx_3090/a100; "
             "Run:AI: NODE_TYPE=<type>), or set EXTRA_ARGS=--allow_fp32 (much slower).")
PY
fi

case "$TARGET" in
  smoke)            python train_eval.py --dataset samsum --method frozen --test_subset 50 --out_root "$OUT"
                    cat "$OUT/samsum__frozen__sub50/metrics.json" ;;
  samsum)           STAGE=main     DATASETS=samsum bash run_all.sh ;;
  cnn)              STAGE=main     DATASETS=cnn_dm bash run_all.sh ;;
  ablation-samsum)  STAGE=ablation DATASETS=samsum bash run_all.sh ;;
  ablation-cnn)     STAGE=ablation DATASETS=cnn_dm bash run_all.sh ;;
  seeds-samsum)     STAGE=seeds DATASETS=samsum bash run_all.sh ;;
  seeds-cnn)        STAGE=seeds DATASETS=cnn_dm bash run_all.sh ;;
  seeds-min-cnn)    STAGE=seeds-min DATASETS=cnn_dm bash run_all.sh ;;
  seeds-min-samsum) STAGE=seeds-min DATASETS=samsum bash run_all.sh ;;
  longer-samsum)    STAGE=longer DATASETS=samsum bash run_all.sh ;;
  longer-cnn)       STAGE=longer DATASETS=cnn_dm bash run_all.sh ;;
  judge-samsum)     STAGE=judge DATASETS=samsum bash run_all.sh
                    python judge_sig.py --dataset samsum --out_root "$OUT" ;;
  judge-cnn)        STAGE=judge DATASETS=cnn_dm bash run_all.sh
                    python judge_sig.py --dataset cnn_dm --out_root "$OUT" ;;
  report)           for DS in samsum cnn_dm; do python bootstrap.py --dataset "$DS" --out_root "$OUT" || true; done
                    python seed_stats.py --runs "$OUT" || true
                    for DS in samsum cnn_dm; do python judge_sig.py --dataset "$DS" --out_root "$OUT" || true; done
                    python make_latex.py --runs "$OUT" --out "$PROJ/paper/results" ;;
  *)                echo "unknown target: $TARGET"; exit 2 ;;
esac
echo "== job $TARGET finished at $(date) =="
