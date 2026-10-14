#!/usr/bin/env bash
# Run on the login node bhn20. Submits one Run:AI training job.
#   bash ~/pegasus_lora/code/cluster/submit.sh <target>
#   targets: smoke | samsum | cnn | ablation-samsum | ablation-cnn | report
# Optional environment overrides (put them before the command):
#   CPU=8 MEM=48Gi GPU=1        resources (defaults shown)
#   TEMPLATE=<name>             use a Run:AI user template instead of CPU/MEM/GPU
#   PREEMPTIBLE=1               run over-quota (starts sooner, may be paused - the code resumes)
#   OFFLINE=1                   compute nodes have no internet (run prefetch first)
#   EXTRA_ARGS="--per_device_bs 2 --gen_bs 8"   smaller batches for a 24 GB GPU
#   NODE_TYPE=<type>            request a specific GPU node type (ask HPC support for names)
set -eo pipefail
TARGET="${1:?usage: submit.sh smoke|samsum|cnn|ablation-samsum|ablation-cnn|report}"
CPU="${CPU:-8}"; MEM="${MEM:-48Gi}"; GPU="${GPU:-1}"
NAME="pl-${TARGET}-$(date +%m%d-%H%M)"          # unique job name (lowercase, digits, dashes)
JOB='~/pegasus_lora/code/cluster/job.sh'   # '~' is expanded inside the job container, not on bhn20

ARGS=(python -n "$NAME" --conda pegasus_lora --backoff-limit 3)
if [ -n "${TEMPLATE:-}" ]; then
  ARGS+=(--ut "$TEMPLATE")
elif [ "$TARGET" = "report" ]; then
  ARGS+=(-c 2 -m 8Gi)                             # CPU-only job
else
  ARGS+=(-c "$CPU" -m "$MEM" -g "$GPU")
fi
[ "${PREEMPTIBLE:-0}" = "1" ] && ARGS+=(--preemptible)
[ -n "${NODE_TYPE:-}" ] && ARGS+=(--node-type "$NODE_TYPE")
ENVS="OFFLINE=${OFFLINE:-0} EXTRA_ARGS='${EXTRA_ARGS:-}'"

echo "Submitting $NAME ..."
runai submit "${ARGS[@]}" -- "$ENVS bash $JOB $TARGET"
echo
echo "Follow the log:   runai logs $NAME --follow"
echo "Job status:       runai list"
echo "Full log file:    ls -t ~/pegasus_lora/logs | head"
