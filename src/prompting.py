"""Clasificación sin entrenamiento (zero-shot / few-shot) con un LLM instruct pequeño.

La predicción se obtiene comparando los logits del primer token de cada etiqueta
(sin generación libre), lo que evita respuestas fuera de formato.

  python -m src.prompting --shots 0
  python -m src.prompting --shots 3 --seed 0     # 3 ejemplos por clase
"""
import argparse
import random
import time

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .utils import TWEET_LABELS, compute_metrics_np, load_task, save_result

LABEL_WORDS = {"negative": "negativo", "neutral": "neutro", "positive": "positivo"}
SYSTEM = (
    "Eres un clasificador de sentimiento de tweets en español. "
    "Responde únicamente con una palabra: negativo, neutro o positivo."
)


def build_prompt(tok, text, shots):
    ex = ""
    if shots:
        ex = "Ejemplos:\n" + "\n".join(f"Tweet: {t}\nSentimiento: {LABEL_WORDS[TWEET_LABELS[l]]}" for t, l in shots) + "\n\n"
    user = f"{ex}Clasifica el sentimiento del siguiente tweet.\nTweet: {text}\nSentimiento:"
    msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


@torch.no_grad()
def predict(model, tok, texts, shots, bs=16):
    first = [tok.encode(LABEL_WORDS[l], add_special_tokens=False)[0] for l in TWEET_LABELS]
    assert len(set(first)) == 3, "los primeros tokens de las etiquetas deben ser distintos"
    preds = []
    for i in range(0, len(texts), bs):
        prompts = [build_prompt(tok, t, shots) for t in texts[i : i + bs]]
        b = tok(prompts, return_tensors="pt", padding=True).to(model.device)
        logits = model(**b).logits[:, -1, :][:, first]
        preds += logits.argmax(-1).tolist()
    return preds


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    p.add_argument("--shots", type=int, default=0, help="ejemplos por clase")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="results")
    a = p.parse_args()

    method = "prompt_zero" if a.shots == 0 else f"prompt_few{a.shots}"
    name = f"tweets_{method}_seed{a.seed}"
    ds, _ = load_task("tweets")
    tok = AutoTokenizer.from_pretrained(a.model)
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.float16).cuda().eval()

    shots = []
    if a.shots:
        rng = random.Random(a.seed)
        tr = ds["train"]
        for c in range(3):
            idx = [i for i, l in enumerate(tr["label"]) if l == c]
            shots += [(tr[i]["text"], c) for i in rng.sample(idx, a.shots)]
        rng.shuffle(shots)

    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    out = {}
    for split in ["validation", "test"]:
        preds = predict(model, tok, ds[split]["text"], shots)
        out[split] = (preds, compute_metrics_np(preds, ds[split]["label"]))
    dt = time.time() - t0
    peak = torch.cuda.max_memory_allocated() / 2**20
    print(name, "TEST", out["test"][1]["acc"], out["test"][1]["macro_f1"])

    save_result(a.out, name, {
        "task": "tweets", "method": method, "seed": a.seed, "model": a.model,
        "val": out["validation"][1], "test": out["test"][1],
        "trainable_params": 0, "total_params": sum(p.numel() for p in model.parameters()),
        "peak_mem_mb": peak, "train_time_s": 0.0, "inference_time_s": dt, "history": [],
        "test_preds": out["test"][0], "test_labels": ds["test"]["label"],
    })


if __name__ == "__main__":
    main()
