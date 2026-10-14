# Fine-Tuning PEGASUS with Low-Rank Adaptation for Abstractive Summarization

Code and experiment scripts for the paper. Everything reported in the paper is produced
by the scripts in `code/`; no number in the paper was entered by hand.

## What is here

```
code/
  train_eval.py      train and evaluate one configuration (LoRA or full fine-tuning)
  common.py          run naming, defaults, dataset and model loading
  bootstrap.py       paired bootstrap significance and TOST equivalence on ROUGE-1
  seed_stats.py      aggregate repeated runs; Welch's t-test across random seeds
  local_judge.py     LLM-as-judge scoring with a local open-weights model
  judge_sig.py       paired bootstrap on the judge's four SummEval dimensions
  judge_vs_rouge.py  Spearman correlation between judge scores and ROUGE
  judge_cost.py      GPU cost of the evaluation, from the scheduler's accounting
  meta_eval.py       optional meta-evaluation against human-annotated scores
  prefetch.py        download datasets and checkpoints from a networked machine
  run_all.sh         the full experiment programme, stage by stage
  cluster/           SLURM and Run:AI submission scripts
  requirements.txt   pinned dependencies
```

## Reproducing the experiments

```bash
pip install -r code/requirements.txt
python code/prefetch.py                     # datasets and the PEGASUS checkpoint
bash  code/run_all.sh main                  # the headline comparison
bash  code/run_all.sh seeds                 # repeated runs for the variance analysis
bash  code/run_all.sh longer                # the doubled training schedule
bash  code/run_all.sh judge                 # LLM-as-judge scoring
```

Each run writes `runs/<run-name>/metrics.json` and `runs/<run-name>/per_doc.jsonl`.
The analyses read those files:

```bash
python code/seed_stats.py     --runs runs
python code/judge_sig.py      --dataset samsum --out_root runs
python code/judge_vs_rouge.py --dataset samsum --out_root runs
```

## Notes on reproducibility

* The judge is an open-weights instruction-tuned model run locally with greedy decoding,
  and `local_judge.py` records the exact model revision in its output. A commercial
  endpoint would not be reproducible in the same way.
* `local_judge.py` carries a prompt version. Cached scores from an older prompt version
  are ignored rather than reused, and a degeneracy detector refuses to return scores
  whose distribution has collapsed onto a single value.
* Run names encode the dataset, method, rank, scaling factor, adapted projections,
  placement, random seed and training length, so two runs that differ in any of these
  can never be pooled by accident.

## Hardware

Every result was produced on a single 24 GB consumer GPU. No run used more than one GPU
and no form of model, tensor or pipeline parallelism was used.

## Licence

MIT, see `LICENSE`.
