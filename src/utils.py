"""Utilidades compartidas: datos, métricas, semillas y registro de resultados."""
import json
import os
import random

import numpy as np
from sklearn.metrics import accuracy_score, f1_score

TWEET_LABELS = ["negative", "neutral", "positive"]
TWEET_URL = (
    "https://huggingface.co/datasets/cardiffnlp/tweet_sentiment_multilingual"
    "/resolve/main/data/spanish/{}.jsonl"
)


def set_seed(seed: int):
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_task(task: str):
    """Devuelve (DatasetDict con splits train/validation/test, lista de nombres de clase o None).

    Los jsonl se leen directamente: el dataset de Cardiff usa un script de carga,
    que las versiones recientes de `datasets` ya no soportan.
    """
    from datasets import DatasetDict, load_dataset

    if task == "tweets":
        files = {s: TWEET_URL.format(s) for s in ["train", "validation", "test"]}
        ds = load_dataset("json", data_files=files)
        ds = ds.map(lambda b: {"label": [int(x) for x in b["label"]]}, batched=True)  # vienen como str
        return ds, TWEET_LABELS
    if task == "news":
        from sklearn.model_selection import train_test_split

        full = load_dataset("mteb/SpanishNewsClassification", split="train")
        idx = np.arange(len(full))
        y = np.array(full["label"])
        tr, rest = train_test_split(idx, test_size=0.30, stratify=y, random_state=42)
        va, te = train_test_split(rest, test_size=2 / 3, stratify=y[rest], random_state=42)
        ds = DatasetDict(
            train=full.select(tr), validation=full.select(va), test=full.select(te)
        )
        return ds, None  # la tarjeta del dataset no documenta los nombres de categoría
    raise ValueError(task)


def compute_metrics_np(preds, labels):
    return {
        "acc": float(accuracy_score(labels, preds)),
        "macro_f1": float(f1_score(labels, preds, average="macro")),
        "per_class_f1": [float(x) for x in f1_score(labels, preds, average=None)],
    }


def save_result(out_dir: str, name: str, payload: dict):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, name + ".json")
    with open(path, "w") as f:
        json.dump(payload, f)
    return path
