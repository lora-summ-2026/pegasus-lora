#!/usr/bin/env bash
# Queue everything that still has to happen AFTER the currently running training jobs,
# so nothing has to be watched or launched by hand.
#
#   bash code/cluster/finalize.sh <longer-samsum-jobid> <longer-cnn-jobid>
#
# It submits three jobs held by SLURM dependencies:
#   A  after the SAMSum job : judge the 10-epoch models of the second seed   (GPU, ~10 min)
#   B  after the CNN job    : judge the 4-epoch CNN models                   (GPU, ~20 min)
#   C  after A and B        : every analysis + regenerate the LaTeX numbers  (CPU, ~1 min)
#
# "afterok" means a job starts only if its dependency SUCCEEDED. If a training job fails,
# the dependent job is never released and sits as DependencyNeverSatisfied - that is the
# intended behaviour, not a hang: it stops a broken run from being analysed as if it were
# fine. Cancel such a job with scancel once you have looked at what went wrong.
set -euo pipefail

SAM_JOB="${1:?usage: finalize.sh <longer-samsum-jobid> <longer-cnn-jobid>}"
CNN_JOB="${2:?usage: finalize.sh <longer-samsum-jobid> <longer-cnn-jobid>}"
P="$HOME/pegasus_lora"
ENV_SETUP="source $P/code/cluster/conda_init.sh && conda activate pegasus_lora && cd $P/code"
GPU_OPTS="--gres=gpu:rtx_3090:1 --cpus-per-task=8 --mem=48G --time=06:00:00"
LOG="--output=$P/logs/slurm_%x_%j.out"

echo "waiting on: SAMSum job $SAM_JOB, CNN job $CNN_JOB"

# ---- A: judge the second seed of the 10-epoch SAMSum models ---------------
A=$(sbatch --parsable $GPU_OPTS $LOG --job-name=pl_judge_ep10s13 \
    --dependency=afterok:"$SAM_JOB" \
    --wrap "$ENV_SETUP && python local_judge.py --dataset samsum --epochs 10 --seed 13 \
            --systems lora8,full --out_root $P/runs")
echo "  A $A  judge SAMSum 10-epoch seed 13   (after $SAM_JOB)"

# ---- B: judge the 4-epoch CNN models --------------------------------------
B=$(sbatch --parsable $GPU_OPTS $LOG --job-name=pl_judge_cnn_ep4 \
    --dependency=afterok:"$CNN_JOB" \
    --wrap "$ENV_SETUP && python local_judge.py --dataset cnn_dm --epochs 4 \
            --systems lora8,full --out_root $P/runs")
echo "  B $B  judge CNN 4-epoch               (after $CNN_JOB)"

# ---- C: all analyses, once every model above has been judged --------------
C=$(sbatch --parsable --cpus-per-task=2 --mem=8G --time=01:00:00 $LOG \
    --job-name=pl_analyse --dependency=afterok:"$A":"$B" \
    --wrap "$ENV_SETUP && \
      python seed_stats.py --runs $P/runs && \
      python judge_sig.py --dataset samsum --out_root $P/runs && \
      python judge_sig.py --dataset samsum --variant ep10 --out_root $P/runs && \
      python judge_sig.py --dataset samsum --variant ep10_seed13 --out_root $P/runs && \
      python judge_sig.py --dataset cnn_dm --out_root $P/runs && \
      python judge_sig.py --dataset cnn_dm --variant ep4 --out_root $P/runs && \
      python judge_vs_rouge.py --dataset samsum --out_root $P/runs && \
      python judge_vs_rouge.py --dataset cnn_dm --out_root $P/runs && \
      python make_latex.py --runs $P/runs --out $P/paper/results")
echo "  C $C  all analyses + LaTeX numbers    (after $A and $B)"

cat <<EOF

Submitted. Nothing else to launch by hand.

Watch progress:      squeue -u ${USER:-$(whoami)}
Read the summary:    cat $P/logs/slurm_pl_analyse_$C.out
Numbers for the paper end up in: $P/paper/results/

A job shown as (Dependency) is waiting on purpose. A job shown as
(DependencyNeverSatisfied) means the run it waited for did not succeed - look at that
run's log before doing anything else, then scancel the stuck job.
EOF
