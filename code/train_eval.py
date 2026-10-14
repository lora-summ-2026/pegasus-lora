"""Train (optional) and evaluate one configuration of PEGASUS-large.

Examples
--------
  # zero-shot baseline (no training)
  python train_eval.py --dataset samsum --method frozen
  # full fine-tuning
  python train_eval.py --dataset samsum --method full
  # LoRA, default configuration (W_Q, W_V in every attention sub-layer, r=8, alpha=16)
  python train_eval.py --dataset samsum --method lora --rank 8
  # ablation: adapt only the decoder, evaluate CNN/DM on the fixed 2,000-doc test subset
  python train_eval.py --dataset cnn_dm --method lora --placement decoder --test_subset 2000

Outputs (in <out_root>/<run_name>/):
  train_stats.json   trainable/total params, peak GPU memory, wall time, steps
  predictions.jsonl  one line per test document: id, prediction, reference
  per_doc.jsonl      one line per test document: id, rouge1, rouge2, rougeLsum
  metrics.json       all of the above aggregated + the full configuration
The script is resumable: re-running the same command skips finished stages
and continues generation from the last saved document.
"""
import argparse
import glob
import json
import os
import time

import numpy as np
import torch

from common import (DATASETS, DEFAULTS, EVAL_SUBSET_SEED, MODEL_NAME, TRAIN_SUBSET_SEED,
                    append_jsonl, clean, fixed_subset, load_dataset_splits,
                    lora_target_regex, read_jsonl, rouge_per_doc, run_name)


LR = {"full": 5e-5, "lora": 5e-4}


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True, choices=list(DATASETS))
    p.add_argument("--method", required=True, choices=["frozen", "full", "lora"])
    p.add_argument("--rank", type=int, default=8)
    p.add_argument("--alpha", type=int, default=16)
    p.add_argument("--targets", default="qv", choices=["qv", "q", "v", "qkvo", "ffn"])
    p.add_argument("--placement", default="both", choices=["both", "encoder", "decoder"])
    p.add_argument("--test_subset", type=int, default=0,
                   help="0 = full test set; N = fixed random subset of N documents")
    p.add_argument("--max_train_samples", type=int, default=None)
    p.add_argument("--max_val_samples", type=int, default=500)
    p.add_argument("--epochs", type=float, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--effective_bs", type=int, default=64)
    p.add_argument("--per_device_bs", type=int, default=None)
    p.add_argument("--gen_bs", type=int, default=None)
    p.add_argument("--num_beams", type=int, default=4)
    p.add_argument("--length_penalty", type=float, default=0.8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out_root", default="runs")
    p.add_argument("--measure_latency", action="store_true",
                   help="LoRA only: time merged vs. unmerged generation on 100 test docs")
    p.add_argument("--save_steps", type=int, default=200,
                   help="evaluate + checkpoint every N optimizer steps (bounds work lost to pre-emption)")
    p.add_argument("--keep_checkpoints", action="store_true",
                   help="keep intermediate checkpoints after training finishes (default: delete to save disk)")
    p.add_argument("--save_only_model", action="store_true",
                   help="do not store optimizer state in checkpoints (saves disk; resume restarts the optimizer)")
    p.add_argument("--allow_fp32", action="store_true",
                   help="run without bf16 on GPUs that lack it (slow). fp16 is NOT supported: PEGASUS overflows.")
    a = p.parse_args()
    d = DEFAULTS[a.dataset]
    for k, v in d.items():
        if getattr(a, k, None) is None:
            setattr(a, k, v)
    if a.lr is None and a.method != "frozen":
        a.lr = LR[a.method]
    return a


def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return train, total


def build_model(a):
    from transformers import AutoModelForSeq2SeqLM
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float32)
    if a.method == "frozen":
        for p in model.parameters():
            p.requires_grad = False
    elif a.method == "lora":
        from peft import LoraConfig, TaskType, get_peft_model
        cfg = LoraConfig(task_type=TaskType.SEQ_2_SEQ_LM, r=a.rank, lora_alpha=a.alpha,
                         lora_dropout=0.1, bias="none",
                         target_modules=lora_target_regex(a.targets, a.placement))
        model = get_peft_model(model, cfg)
    return model


def tokenize_fn(tok, cfg):
    def f(batch):
        enc = tok(batch["source"], max_length=cfg["max_src"], truncation=True)
        lab = tok(text_target=batch["target"], max_length=cfg["max_tgt"], truncation=True)
        enc["labels"] = lab["input_ids"]
        return enc
    return f


def train(a, model, tok, splits, out_dir, use_bf16):
    from transformers import (DataCollatorForSeq2Seq, Seq2SeqTrainer,
                              Seq2SeqTrainingArguments)
    cfg = DATASETS[a.dataset]
    tr = fixed_subset(splits["train"], a.max_train_samples, TRAIN_SUBSET_SEED)
    va = fixed_subset(splits["validation"], a.max_val_samples, EVAL_SUBSET_SEED)
    cols = tr.column_names
    tr = tr.map(tokenize_fn(tok, cfg), batched=True, remove_columns=cols)
    va = va.map(tokenize_fn(tok, cfg), batched=True, remove_columns=cols)

    def compute_metrics(ev):
        preds, labels = ev
        preds = np.where(preds < 0, tok.pad_token_id, preds)
        labels = np.where(labels < 0, tok.pad_token_id, labels)
        P = tok.batch_decode(preds, skip_special_tokens=True)
        R = tok.batch_decode(labels, skip_special_tokens=True)
        s = [rouge_per_doc(p, r) for p, r in zip(P, R)]
        return {k: float(np.mean([x[k] for x in s])) for k in s[0]}

    grad_accum = max(1, a.effective_bs // a.per_device_bs)
    ckpt_dir = os.path.join(out_dir, "checkpoints")
    args = Seq2SeqTrainingArguments(
        output_dir=ckpt_dir,
        per_device_train_batch_size=a.per_device_bs,
        per_device_eval_batch_size=a.gen_bs,
        gradient_accumulation_steps=grad_accum,
        learning_rate=a.lr, weight_decay=0.01, warmup_ratio=0.05,
        lr_scheduler_type="linear", num_train_epochs=a.epochs,
        label_smoothing_factor=0.1,
        bf16=use_bf16,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        eval_strategy="steps", eval_steps=a.save_steps,
        save_strategy="steps", save_steps=a.save_steps, save_total_limit=2,
        save_only_model=a.save_only_model,
        load_best_model_at_end=True, metric_for_best_model="rougeLsum", greater_is_better=True,
        predict_with_generate=True, generation_max_length=cfg["max_tgt"],
        generation_num_beams=a.num_beams,
        logging_steps=20, report_to="none", seed=a.seed,
        dataloader_num_workers=2,
    )
    trainer = Seq2SeqTrainer(model=model, args=args, train_dataset=tr, eval_dataset=va,
                             data_collator=DataCollatorForSeq2Seq(tok, model=model),
                             processing_class=tok, compute_metrics=compute_metrics)
    resume = bool(glob.glob(os.path.join(ckpt_dir, "checkpoint-*")))
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    out = trainer.train(resume_from_checkpoint=True if resume else None)
    wall = time.time() - t0
    peak = torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else None
    stats = dict(train_wall_s=wall, train_steps=out.global_step,
                 s_per_step=wall / max(1, out.global_step), peak_mem_gb=peak,
                 resumed=resume, n_train=len(tr), grad_accum=grad_accum,
                 best_val=trainer.state.best_metric)
    final = os.path.join(out_dir, "final")
    trainer.model.save_pretrained(final)
    tok.save_pretrained(final)
    return trainer.model, stats


def load_trained(a, out_dir):
    from transformers import AutoModelForSeq2SeqLM
    final = os.path.join(out_dir, "final")
    if a.method == "full":
        return AutoModelForSeq2SeqLM.from_pretrained(final, torch_dtype=torch.float32)
    from peft import PeftModel
    base = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float32)
    return PeftModel.from_pretrained(base, final)


@torch.no_grad()
def generate(model, tok, texts, cfg, a, device, dtype):
    enc = tok(texts, max_length=cfg["max_src"], truncation=True, padding=True,
              return_tensors="pt").to(device)
    with torch.autocast(device_type=device.type, dtype=dtype, enabled=dtype != torch.float32):
        out = model.generate(**enc, num_beams=a.num_beams, length_penalty=a.length_penalty,
                             max_new_tokens=cfg["max_tgt"], early_stopping=True)
    return [clean(t) for t in tok.batch_decode(out, skip_special_tokens=True)]


def time_generation(model, tok, docs, cfg, a, device, dtype):
    model.eval()
    generate(model, tok, docs[:2], cfg, a, device, dtype)  # warm-up
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    for d in docs:
        generate(model, tok, [d], cfg, a, device, dtype)
    if device.type == "cuda":
        torch.cuda.synchronize()
    return 1000.0 * (time.time() - t0) / len(docs)


def main():
    a = parse()
    from transformers import AutoTokenizer, set_seed
    set_seed(a.seed)
    name = run_name(a.dataset, a.method, a.rank, a.alpha, a.targets, a.placement, a.test_subset,
                    a.seed, a.epochs, a.max_train_samples)
    out_dir = os.path.join(a.out_root, name)
    os.makedirs(out_dir, exist_ok=True)
    metrics_path = os.path.join(out_dir, "metrics.json")
    if os.path.exists(metrics_path):
        print(f"[skip] {name}: metrics.json already exists")
        return

    cuda = torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")
    use_bf16 = cuda and torch.cuda.is_bf16_supported()
    if not use_bf16 and not a.allow_fp32:
        raise SystemExit("This GPU has no bf16 support (e.g. T4/P100/V100). PEGASUS is unstable in "
                         "fp16. Use an A100/L4/H100 runtime, or pass --allow_fp32 (much slower).")
    dtype = torch.bfloat16 if use_bf16 else torch.float32
    print(f"[run] {name} | device={device} bf16={use_bf16}")

    cfg = DATASETS[a.dataset]
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    splits = load_dataset_splits(a.dataset)

    stats_path = os.path.join(out_dir, "train_stats.json")
    if a.method == "frozen":
        model = build_model(a)
        trainable, total = count_params(model)
        stats = dict(trainable=trainable, total=total)
    elif os.path.exists(stats_path):
        stats = json.load(open(stats_path))
        model = load_trained(a, out_dir)
    else:
        model = build_model(a)
        trainable, total = count_params(model)
        print(f"[params] trainable={trainable:,} total={total:,} ({100*trainable/total:.3f}%)")
        model, stats = train(a, model, tok, splits, out_dir, use_bf16)
        stats.update(trainable=trainable, total=total)
        json.dump(stats, open(stats_path, "w"), indent=2)
        if not a.keep_checkpoints:  # only after final/ and train_stats.json are safely on disk
            import shutil
            shutil.rmtree(os.path.join(out_dir, "checkpoints"), ignore_errors=True)

    model.to(device).eval()
    test = fixed_subset(splits["test"], a.test_subset, EVAL_SUBSET_SEED)

    if a.method == "lora" and a.measure_latency and "latency" not in stats:
        docs = list(fixed_subset(test, 100, EVAL_SUBSET_SEED)["source"])
        unmerged = time_generation(model, tok, docs, cfg, a, device, dtype)
        model = model.merge_and_unload()
        merged = time_generation(model, tok, docs, cfg, a, device, dtype)
        stats["latency"] = dict(unmerged_ms=unmerged, merged_ms=merged)
        json.dump(stats, open(stats_path, "w"), indent=2)
    if a.method == "lora" and hasattr(model, "merge_and_unload"):
        model = model.merge_and_unload()
    if a.method == "frozen" and a.measure_latency and "latency" not in stats:
        docs = list(fixed_subset(test, 100, EVAL_SUBSET_SEED)["source"])
        stats["latency"] = dict(merged_ms=time_generation(model, tok, docs, cfg, a, device, dtype))

    # ---- generation (resumable) -------------------------------------------------
    pred_path = os.path.join(out_dir, "predictions.jsonl")
    done = {r["id"] for r in read_jsonl(pred_path)}
    todo = [ex for ex in test if ex["id"] not in done]
    todo.sort(key=lambda ex: len(ex["source"]), reverse=True)  # length-bucketing for speed
    t0 = time.time()
    for i in range(0, len(todo), a.gen_bs):
        batch = todo[i:i + a.gen_bs]
        preds = generate(model, tok, [b["source"] for b in batch], cfg, a, device, dtype)
        append_jsonl(pred_path, [{"id": b["id"], "prediction": p, "reference": b["target"]}
                                 for b, p in zip(batch, preds)])
        print(f"  generated {len(done) + i + len(batch)}/{len(test)}", flush=True)
    gen_wall = time.time() - t0

    # ---- scoring -----------------------------------------------------------------
    preds = read_jsonl(pred_path)
    per_doc = [{"id": r["id"], **rouge_per_doc(r["prediction"], r["reference"])} for r in preds]
    with open(os.path.join(out_dir, "per_doc.jsonl"), "w") as f:
        for r in per_doc:
            f.write(json.dumps(r) + "\n")
    agg = {k: float(np.mean([r[k] for r in per_doc])) for k in ("rouge1", "rouge2", "rougeLsum")}
    metrics = dict(run=name, config=vars(a), n_test=len(per_doc), rouge=agg,
                   gen_wall_s=gen_wall, gpu=torch.cuda.get_device_name(0) if cuda else "cpu",
                   **stats)
    json.dump(metrics, open(metrics_path, "w"), indent=2)
    print(json.dumps(agg, indent=2))


if __name__ == "__main__":
    main()
