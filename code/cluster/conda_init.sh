# Makes `conda` available in the current shell (sourced by the SLURM scripts).
# Order: an existing ~/miniconda3, then the cluster's anaconda module, then a conda already on PATH.
if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif command -v module >/dev/null 2>&1 && module avail 2>&1 | grep -qi anaconda; then
  module load "$(module avail 2>&1 | grep -oi 'anaconda[^ ]*' | head -1)" >/dev/null 2>&1 || true
fi
if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
fi
