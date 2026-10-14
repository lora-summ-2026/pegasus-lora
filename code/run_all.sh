#!/usr/bin/env bash
# Runs the complete experimental grid in priority order. Every step is skipped
# automatically if its results already exist, so after a disconnect just run the
# script again and it continues where it stopped.
#
#   bash run_all.sh                 # round 1: main table + ablations
#   STAGE=main bash run_all.sh      # only the main table (frozen / full / LoRA-4/8/16)
#   STAGE=ablation bash run_all.sh  # only the ablations
#   STAGE=seeds bash run_all.sh     # round 2: repeat headline configs under extra seeds
#   STAGE=judge bash run_all.sh     # round 2: local open-weights LLM judge (no API, no cost)
#   STAGE=seeds-min bash run_all.sh # round 2: extra seeds for lora8 + full only (cheap)
#   STAGE=longer bash run_all.sh    # round 2: longer training, same configs
#   LONG_SEEDS="13" STAGE=longer bash run_all.sh   # ... and repeat it under seed 13
#   DATASETS="samsum" bash run_all.sh   # restrict to one dataset (SAMSum is ~10x cheaper)
#   OUT=/content/drive/MyDrive/pegasus_lora/runs bash run_all.sh   # store runs elsewhere
#   EXTRA_ARGS="--per_device_bs 2 --gen_bs 8" bash run_all.sh      # smaller batches (24GB GPU)
set -euo pipefail
cd "$(dirname "$0")"
OUT=${OUT:-runs}
STAGE=${STAGE:-all}
DATASETS=${DATASETS:-"samsum cnn_dm"}
ABL_SUB=2000            # must match ABL_SUB in make_latex.py
EXTRA_ARGS=${EXTRA_ARGS:-}   # e.g. "--per_device_bs 2 --gen_bs 8" on a 24GB GPU
SEEDS=${SEEDS:-"13 7"}       # extra seeds; 42 is the first round and is never repeated
JUDGE_MODEL=${JUDGE_MODEL:-Qwen/Qwen2.5-7B-Instruct}
JUDGE_N=${JUDGE_N:-300}      # documents sampled per dataset, same docs for every system
JUDGE_REV=${JUDGE_REV:-}     # optional hub commit/tag, pinned for reproducibility
run() { echo; echo "=== python train_eval.py $* $EXTRA_ARGS ==="; python train_eval.py --out_root "$OUT" "$@" $EXTRA_ARGS; }

for DS in $DATASETS; do
  SUB=0; [ "$DS" = "cnn_dm" ] && SUB=$ABL_SUB

  if [ "$STAGE" = "all" ] || [ "$STAGE" = "main" ]; then
    run --dataset "$DS" --method frozen --measure_latency
    run --dataset "$DS" --method lora --rank 8 --measure_latency
    run --dataset "$DS" --method full --measure_latency
    run --dataset "$DS" --method lora --rank 4
    run --dataset "$DS" --method lora --rank 16
    python bootstrap.py --dataset "$DS" --out_root "$OUT"
  fi

  if [ "$STAGE" = "all" ] || [ "$STAGE" = "ablation" ]; then
    for T in q v qkvo ffn; do run --dataset "$DS" --method lora --targets "$T" --test_subset $SUB; done
    for P in encoder decoder; do run --dataset "$DS" --method lora --placement "$P" --test_subset $SUB; done
    for A in 8 32; do run --dataset "$DS" --method lora --alpha "$A" --test_subset $SUB; done
  fi

  # --- round 2, requested by the co-author ---------------------------------
  # Repeat the headline configurations under extra seeds. run_name() puts the seed
  # in the directory name for every seed but 42, so these do NOT collide with, and
  # are not skipped because of, the runs already completed.
  if [ "$STAGE" = "seeds" ]; then
    for S in $SEEDS; do
      run --dataset "$DS" --method lora --rank 8 --seed "$S"
      run --dataset "$DS" --method full          --seed "$S"
      for A in 8 32; do
        run --dataset "$DS" --method lora --alpha "$A" --test_subset $SUB --seed "$S"
      done
      for R in 4 16; do
        run --dataset "$DS" --method lora --rank "$R" --seed "$S"
      done
    done
  fi

  # Only the two headline configurations under extra seeds. For CNN/DailyMail the full
  # seed sweep would cost days, while lora8 + full is what the main claim rests on.
  if [ "$STAGE" = "seeds-min" ]; then
    for S in $SEEDS; do
      run --dataset "$DS" --method lora --rank 8 --seed "$S"
      run --dataset "$DS" --method full          --seed "$S"
    done
  fi

  # Longer training, to show the LoRA-vs-full gap is not an artefact of the fixed budget
  # (one method might simply converge faster). run_name() puts the epoch count in the
  # directory name, so these do not collide with the default-budget runs.
  if [ "$STAGE" = "longer" ]; then
    case "$DS" in
      samsum) LE=${LONG_EPOCHS:-10} ;;
      cnn_dm) LE=${LONG_EPOCHS:-4}  ;;
      *)      LE=${LONG_EPOCHS:-10} ;;
    esac
    run --dataset "$DS" --method lora --rank 8 --epochs "$LE"
    run --dataset "$DS" --method full          --epochs "$LE"
    # Extra seeds at the longer budget. Empty by default, so the two runs above keep
    # their unsuffixed directory names; LONG_SEEDS="13" adds one repeat of each.
    for S in ${LONG_SEEDS:-}; do
      run --dataset "$DS" --method lora --rank 8 --epochs "$LE" --seed "$S"
      run --dataset "$DS" --method full          --epochs "$LE" --seed "$S"
    done
  fi

  # Local open-weights judge: no API key, no cost, reproducible from the model name.
  if [ "$STAGE" = "judge" ]; then
    python local_judge.py --dataset "$DS" --out_root "$OUT" --n "$JUDGE_N" \
        --judge "$JUDGE_MODEL" ${JUDGE_REV:+--judge_revision "$JUDGE_REV"}
  fi
done

if [ "$STAGE" = "seeds" ] || [ "$STAGE" = "seeds-min" ] || [ "$STAGE" = "longer" ]; then
  python seed_stats.py --runs "$OUT"
fi

python make_latex.py --runs "$OUT" --out ../paper/results
echo "Done. Upload ../paper/results/ to Overleaf (next to main.tex)."
