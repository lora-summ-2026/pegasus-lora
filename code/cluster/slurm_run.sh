#!/bin/bash
# Runs one experiment target on a GPU node of a SLURM cluster.
# Submit from ~/pegasus_lora:      sbatch code/cluster/slurm_run.sh <target>
#   target = smoke | samsum | cnn | ablation-samsum | ablation-cnn
#            | seeds-samsum | seeds-cnn | judge-samsum | judge-cnn | report
#
# The judge and the seed sweep need an Ampere-or-newer GPU (the 7B judge wants ~16 GB):
#   sbatch --gres=gpu:rtx_3090:1 code/cluster/slurm_run.sh judge-samsum
# Log:                             ~/pegasus_lora/logs/slurm_pl_run_<jobid>.out
# If a job stops (time limit / pre-emption), submit the same command again: it resumes.
#SBATCH --job-name=pl_run
#SBATCH --output=logs/slurm_%x_%j.out
#SBATCH --time=2-00:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
set -eo pipefail
source "$HOME/pegasus_lora/code/cluster/conda_init.sh"
conda activate pegasus_lora
bash "$HOME/pegasus_lora/code/cluster/job.sh" "${1:?usage: sbatch code/cluster/slurm_run.sh <target>}"
