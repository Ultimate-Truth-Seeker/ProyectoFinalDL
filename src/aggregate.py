"""Agrega los json de results/ en tablas (media±std) y figuras."""
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay

R, F = "results", "figures"
os.makedirs(F, exist_ok=True)
runs = [json.load(open(p)) for p in glob.glob(f"{R}/*.json")]
rows = [{
    "task": r["task"], "method": r["method"], "seed": r["seed"],
    "val_f1": r["val"]["macro_f1"], "test_acc": r["test"]["acc"], "test_f1": r["test"]["macro_f1"],
    "trainable": r["trainable_params"], "total": r["total_params"],
    "mem_mb": r["peak_mem_mb"], "time_s": r["train_time_s"] or r.get("inference_time_s", 0),
} for r in runs]
df = pd.DataFrame(rows)
g = df.groupby(["task", "method"])
ms = lambda c: g[c].apply(lambda s: f"{s.mean():.3f} ± {s.std(ddof=1) if len(s) > 1 else 0:.3f}")
tab = pd.DataFrame({
    "n_seeds": g.size(), "test_acc": ms("test_acc"), "test_macroF1": ms("test_f1"),
    "params_entrenables": g["trainable"].first(), "%_del_total": (100 * g["trainable"].first() / g["total"].first()).round(2),
    "mem_pico_MB": g["mem_mb"].mean().round(0), "tiempo_s": g["time_s"].mean().round(1),
})
tab.to_csv(f"{R}/tabla_comparativa.csv")
open(f"{R}/tabla_comparativa.md", "w").write(tab.to_markdown())
print(tab.to_markdown())

# curvas de entrenamiento (media entre semillas) para tweets
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
for method in sorted(df[df.task == "tweets"].method.unique()):
    if method.startswith("prompt") or method not in ("full", "lora_r8_qv"):
        continue
    hs = [r["history"] for r in runs if r["task"] == "tweets" and r["method"] == method]
    ev = [[h for h in hist if "eval_macro_f1" in h] for hist in hs]
    tr = [[h for h in hist if "loss" in h and "eval_loss" not in h and "train_loss" not in h] for hist in hs]
    n = min(len(e) for e in ev)
    ax[0].plot(range(1, n + 1), np.mean([[h["loss"] for h in t][:n] for t in tr], 0), label=f"{method} train")
    ax[0].plot(range(1, n + 1), np.mean([[h["eval_loss"] for h in e][:n] for e in ev], 0), "--", label=f"{method} val")
    ax[1].plot(range(1, n + 1), np.mean([[h["eval_macro_f1"] for h in e][:n] for e in ev], 0), label=method)
ax[0].set(title="Pérdida (entropía cruzada)", xlabel="época"); ax[0].legend()
ax[1].set(title="F1 macro en validación", xlabel="época"); ax[1].legend()
plt.tight_layout(); plt.savefig(f"{F}/curvas_entrenamiento.png", dpi=150); plt.close()

# F1 vs parámetros entrenables y memoria
t = df[df.task == "tweets"].groupby("method").agg(f1=("test_f1", "mean"), p=("trainable", "first"), m=("mem_mb", "mean"))
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
t["f1"].plot.bar(ax=ax[0], ylim=(0, 1)); ax[0].set_title("F1 macro (test)")
t["m"].plot.bar(ax=ax[1], color="tab:orange"); ax[1].set_title("Memoria pico GPU (MB)")
plt.tight_layout(); plt.savefig(f"{F}/f1_memoria.png", dpi=150); plt.close()

# matrices de confusión (semilla 0, tweets)
for r in runs:
    if r["task"] == "tweets" and r["seed"] == 0 and r["method"] in ("prompt_zero", "full", "lora_r8_qv"):
        fig, a = plt.subplots(figsize=(4, 4))
        ConfusionMatrixDisplay.from_predictions(r["test_labels"], r["test_preds"],
            display_labels=["neg", "neu", "pos"], ax=a, colorbar=False)
        a.set_title(r["method"]); plt.tight_layout()
        plt.savefig(f"{F}/cm_{r['method']}.png", dpi=150); plt.close()
