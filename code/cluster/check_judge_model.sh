#!/usr/bin/env bash
# Probe the cluster for an already-available judge model, BEFORE downloading 15 GB.
# Safe to run on the LOGIN node: it only reads directory listings and env vars.
#
#   bash code/cluster/check_judge_model.sh
#
# It reports, in order:
#   1. which Hugging Face cache directories exist and what is already in them
#   2. any shared//site-wide model directory the cluster may provide
#   3. free space in the places a 15 GB download would land
#   4. whether the login node can reach huggingface.co at all
echo "=================================================================="
echo " 1. Hugging Face caches visible from this account"
echo "=================================================================="
for d in "${HF_HOME:-}" "${HUGGINGFACE_HUB_CACHE:-}" "${TRANSFORMERS_CACHE:-}" \
         "$HOME/.cache/huggingface/hub" "$HOME/.cache/huggingface"; do
  [ -z "$d" ] && continue
  if [ -d "$d" ]; then
    echo
    echo "--- $d  ($(du -sh "$d" 2>/dev/null | cut -f1) used)"
    ls -1 "$d" 2>/dev/null | grep -i "^models--" | head -40
  fi
done
echo
echo "env: HF_HOME=${HF_HOME:-<unset>}  HUGGINGFACE_HUB_CACHE=${HUGGINGFACE_HUB_CACHE:-<unset>}"

echo
echo "=================================================================="
echo " 2. Shared model directories provided by the cluster"
echo "=================================================================="
found=0
for d in /storage/models /data/models /shared/models /opt/models /usr/local/models \
         /storage/shared /sise/shared /cs/shared /mnt/shared; do
  if [ -d "$d" ]; then
    found=1
    echo "--- $d"
    ls -1 "$d" 2>/dev/null | head -20
  fi
done
[ "$found" -eq 0 ] && echo "(none of the usual shared model paths exist)"
echo
echo "anything with 'qwen', 'llama' or 'mistral' in its name under /storage /data /sise:"
for root in /storage /data /sise; do
  [ -d "$root" ] || continue
  find "$root" -maxdepth 4 -iname "*qwen*" -o -maxdepth 4 -iname "*llama*" \
       -o -maxdepth 4 -iname "*mistral*" 2>/dev/null | head -10
done

echo
echo "=================================================================="
echo " 3. Free space (a 7B model in bf16 needs about 16 GB)"
echo "=================================================================="
df -h "$HOME" 2>/dev/null | tail -1 | awk '{printf "  HOME   %s free of %s\n", $4, $2}'
for d in /storage /scratch /tmp; do
  [ -d "$d" ] && df -h "$d" 2>/dev/null | tail -1 | awk -v p="$d" '{printf "  %-7s %s free of %s\n", p, $4, $2}'
done
echo
echo "  quota (if the cluster enforces one):"
quota -s 2>/dev/null | tail -3 || echo "  (no quota command)"

echo
echo "=================================================================="
echo " 4. Can this node reach the model hub?"
echo "=================================================================="
if curl -s -o /dev/null -w "  huggingface.co -> HTTP %{http_code}\n" --max-time 15 \
        https://huggingface.co 2>/dev/null; then :; else
  echo "  could not reach huggingface.co from the login node"
  echo "  (compute nodes are sometimes the ones with outbound access - try there)"
fi

echo
echo "=================================================================="
echo " What to do with this output"
echo "=================================================================="
cat <<'EOF'
  * A line like "models--Qwen--Qwen2.5-7B-Instruct" under a cache in section 1
    means the model is ALREADY downloaded - nothing to fetch, just run the judge.
  * A shared directory in section 2 holding a usable instruct model means we can
    point JUDGE_MODEL at that path instead of a hub name.
  * If section 4 shows no access from the login node, the download has to happen
    inside an sbatch job on a compute node.
  * Send this whole output back and we pick the model from what is actually there.
EOF
