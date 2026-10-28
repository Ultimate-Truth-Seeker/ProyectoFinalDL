"""Explicabilidad con SHAP sobre un modelo ya entrenado (guardado con --save_model).

  python -m src.explain --model_dir results/models/tweets_lora_r8_qv_seed0 --n_err 40
"""
import argparse
import collections
import json
import os

import numpy as np
import pandas as pd
import shap
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

from .utils import load_task


def load_for_inference(mdir):
    tok = AutoTokenizer.from_pretrained(mdir)
    if os.path.exists(os.path.join(mdir, "adapter_config.json")):
        from peft import PeftConfig, PeftModel

        cfg = PeftConfig.from_pretrained(mdir)
        base = AutoModelForSequenceClassification.from_pretrained(
            cfg.base_model_name_or_path, num_labels=3,
            id2label={0: "negative", 1: "neutral", 2: "positive"},
            label2id={"negative": 0, "neutral": 1, "positive": 2},
        )
        model = PeftModel.from_pretrained(base, mdir).merge_and_unload()
    else:
        model = AutoModelForSequenceClassification.from_pretrained(mdir)
    return model.eval(), tok


def clean(t):
    return t.replace("▁", "").strip().lower()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_dir", required=True)
    p.add_argument("--n_err", type=int, default=40)
    p.add_argument("--n_ok", type=int, default=20)
    p.add_argument("--out", default="results/explain")
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)

    model, tok = load_for_inference(a.model_dir)
    dev = 0 if torch.cuda.is_available() else -1
    pipe = pipeline("text-classification", model=model, tokenizer=tok, top_k=None,
                    truncation=True, device=dev,
                    max_length=min(128, model.config.max_position_embeddings - 2))
    ds, labels = load_task("tweets")
    texts, gold = ds["test"]["text"], np.array(ds["test"]["label"])

    scores = pipe(texts, batch_size=64)
    pred = np.array([max(range(3), key=lambda k: next(d["score"] for d in s if d["label"] == labels[k]))
                     for s in scores])
    wrong = np.where(pred != gold)[0]
    right = np.where(pred == gold)[0]
    rng = np.random.RandomState(0)
    sel = list(rng.choice(wrong, min(a.n_err, len(wrong)), replace=False)) + \
          list(rng.choice(right, min(a.n_ok, len(right)), replace=False))
    print(f"errores: {len(wrong)}/{len(gold)}; explicando {len(sel)} ejemplos")

    explainer = shap.Explainer(pipe)
    sv = explainer([texts[i] for i in sel])

    # tabla de ejemplos + HTML interactivo
    rows = [{"idx": int(i), "texto": texts[i], "real": labels[gold[i]], "pred": labels[pred[i]],
             "correcto": bool(pred[i] == gold[i])} for i in sel]
    pd.DataFrame(rows).to_csv(f"{a.out}/ejemplos.csv", index=False)
    with open(f"{a.out}/shap_text.html", "w") as f:
        f.write(shap.plots.text(sv, display=False))

    # palabras que más empujan hacia la clase predicha, separadas errores / aciertos
    names = list(sv.output_names)
    agg = {True: collections.defaultdict(list), False: collections.defaultdict(list)}
    for j, i in enumerate(sel):
        c = names.index(labels[pred[i]])
        for tkn, v in zip(sv[j].data, sv[j].values[:, c]):
            w = clean(str(tkn))
            if w and w not in ("@user", "http"):
                agg[bool(pred[i] == gold[i])][w].append(float(v))
    for ok, d in agg.items():
        df = pd.DataFrame([{"palabra": w, "shap_medio": np.mean(v), "n": len(v)} for w, v in d.items()])
        df.sort_values("shap_medio", ascending=False).to_csv(
            f"{a.out}/palabras_{'aciertos' if ok else 'errores'}.csv", index=False)
    print("listo:", a.out)


if __name__ == "__main__":
    main()
