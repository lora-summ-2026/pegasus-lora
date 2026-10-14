"""Shared helpers: dataset loading, run naming, ROUGE scoring.

Every other script imports from here, so the dataset splits, subset
selection and ROUGE configuration are identical across all runs.
"""
import json
import os
import random

MODEL_NAME = "google/pegasus-large"
EVAL_SUBSET_SEED = 1234     # fixed test subset used by CNN/DM ablations
TRAIN_SUBSET_SEED = 42      # fixed training subset for CNN/DM

DATASETS = {
    "cnn_dm": dict(
        hub=[("abisee/cnn_dailymail", "3.0.0"), ("cnn_dailymail", "3.0.0")],
        src="article", tgt="highlights", id="id",
        max_src=1024, max_tgt=128,
    ),
    "samsum": dict(
        hub=[("knkarthick/samsum", None), ("Samsung/samsum", None)],
        src="dialogue", tgt="summary", id="id",
        max_src=512, max_tgt=128,
    ),
}

# Adapter target patterns (PEFT treats a string as a full-match regex over module names).
# Pegasus module names look like: model.encoder.layers.3.self_attn.q_proj
#                                  model.decoder.layers.3.encoder_attn.v_proj  (cross-attention)
_PROJ = {"qv": "q_proj|v_proj", "q": "q_proj", "v": "v_proj",
         "qkvo": "q_proj|k_proj|v_proj|out_proj"}
_STACK = {"both": "encoder|decoder", "encoder": "encoder", "decoder": "decoder"}


def lora_target_regex(targets: str, placement: str) -> str:
    stack = _STACK[placement]
    if targets == "ffn":
        return rf"model\.({stack})\.layers\.\d+\.(fc1|fc2)"
    return rf"model\.({stack})\.layers\.\d+\.(self_attn|encoder_attn)\.({_PROJ[targets]})"


DEFAULT_SEED = 42           # the seed every run in the first round of experiments used

# Per-dataset training budget; keep in sync with the Experimental Setup section.
# run_name() compares against these so that a run with a DIFFERENT budget gets its own
# directory instead of silently landing in - and being skipped because of - the default one.
DEFAULTS = {
    "cnn_dm": dict(max_train_samples=50000, epochs=2, per_device_bs=4, gen_bs=16),
    "samsum": dict(max_train_samples=0, epochs=5, per_device_bs=8, gen_bs=32),
}


def run_name(dataset, method, rank=8, alpha=16, targets="qv", placement="both", test_subset=0,
             seed=DEFAULT_SEED, epochs=None, max_train_samples=None):
    """Directory name for one run.

    The seed is part of the name for every seed EXCEPT DEFAULT_SEED, whose name is
    left unsuffixed so that the runs already completed with seed 42 keep matching and
    are not recomputed. Without the seed in the name, a second seed would land in the
    directory of the first and `run_all.sh` would skip it as already done, silently
    producing duplicate numbers instead of a variance estimate.
    """
    if method in ("frozen", "full"):
        name = f"{dataset}__{method}"
    else:
        name = f"{dataset}__lora_r{rank}_a{alpha}_{targets}_{placement}"
    if test_subset:
        name += f"__sub{test_subset}"
    if seed != DEFAULT_SEED:
        name += f"__seed{seed}"
    d = DEFAULTS.get(dataset, {})
    if epochs is not None and d and float(epochs) != float(d["epochs"]):
        name += f"__ep{epochs:g}"
    if max_train_samples is not None and d and int(max_train_samples) != int(d["max_train_samples"]):
        name += f"__tr{int(max_train_samples)}"
    return name


def load_dataset_splits(name):
    """Load a DatasetDict with columns: id, source, target (empty rows removed)."""
    from datasets import load_dataset
    cfg = DATASETS[name]
    last_err = None
    for repo, conf in cfg["hub"]:
        try:
            ds = load_dataset(repo, conf) if conf else load_dataset(repo)
            break
        except Exception as e:  # try the next mirror
            last_err = e
            ds = None
    if ds is None:
        raise RuntimeError(f"Could not load {name}: {last_err}")

    def _norm(ex, idx):
        return {"id": str(ex.get(cfg["id"]) or idx),
                "source": ex[cfg["src"]] or "",
                "target": ex[cfg["tgt"]] or ""}

    out = {}
    for split in ("train", "validation", "test"):
        d = ds[split].map(_norm, with_indices=True, remove_columns=ds[split].column_names)
        d = d.filter(lambda ex: len(ex["source"].strip()) > 0 and len(ex["target"].strip()) > 0)
        out[split] = d
    return out


def fixed_subset(dataset, n, seed):
    if not n or n >= len(dataset):
        return dataset
    idx = list(range(len(dataset)))
    random.Random(seed).shuffle(idx)
    return dataset.select(sorted(idx[:n]))


# ---------------------------------------------------------------- ROUGE
_scorer = None


def _ensure_nltk():
    import nltk
    for pkg in ("punkt", "punkt_tab"):
        try:
            nltk.data.find(f"tokenizers/{pkg}")
        except LookupError:
            nltk.download(pkg, quiet=True)


def clean(text):
    return text.replace("<n>", " ").strip()


def rouge_per_doc(pred, ref):
    """ROUGE-1/2 F1 and summary-level ROUGE-L (rougeLsum) F1, x100.

    Standard `rouge-score` configuration: Porter stemming, NO stopword removal;
    rougeLsum is computed on newline-separated sentences (NLTK punkt).
    """
    global _scorer
    import nltk
    from rouge_score import rouge_scorer
    if _scorer is None:
        _ensure_nltk()
        _scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeLsum"], use_stemmer=True)
    pred, ref = clean(pred), clean(ref)
    p = "\n".join(nltk.sent_tokenize(pred))
    r = "\n".join(nltk.sent_tokenize(ref))
    s = _scorer.score(r, p)
    return {k: 100.0 * s[k].fmeasure for k in ("rouge1", "rouge2", "rougeLsum")}


def read_jsonl(path):
    """Read a JSONL file; a line cut off by a pre-empted job is skipped, not fatal."""
    if not os.path.exists(path):
        return []
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return rows


def append_jsonl(path, rows):
    with open(path, "a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
