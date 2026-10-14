"""Download everything the experiments need, ONCE, into the shared cache.

Run it from a place with internet access (on the cluster: the Jupyter
workspace terminal, inside the conda env). Afterwards the training jobs can run
fully offline (set OFFLINE=1 in cluster/job.sh if compute nodes have no internet).

  python prefetch.py
Uses HF_HOME and NLTK_DATA from the environment (cluster/env.sh sets both).
"""
import os

from common import DATASETS, MODEL_NAME, load_dataset_splits

print("HF_HOME   =", os.environ.get("HF_HOME", "(default ~/.cache/huggingface)"))
print("NLTK_DATA =", os.environ.get("NLTK_DATA", "(default ~/nltk_data)"))

# Download the files only (no loading into RAM - the setup workspace may be small).
from huggingface_hub import snapshot_download
snapshot_download(MODEL_NAME, ignore_patterns=["*.h5", "*.msgpack", "*.ot", "*.onnx"])
from transformers import AutoTokenizer
AutoTokenizer.from_pretrained(MODEL_NAME)
print(f"[ok] model {MODEL_NAME}")

for name in DATASETS:
    splits = load_dataset_splits(name)
    print(f"[ok] {name}: " + ", ".join(f"{k}={len(v)}" for k, v in splits.items()))

import nltk
target = os.environ.get("NLTK_DATA")
for pkg in ("punkt", "punkt_tab"):
    nltk.download(pkg, quiet=True, download_dir=target) if target else nltk.download(pkg, quiet=True)
print("[ok] nltk punkt")
print("Everything is cached. Training jobs can now run offline.")
