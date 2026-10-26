"""Fine-tuning completo vs LoRA de un encoder multilingüe para clasificación de texto.

Uso:
  python -m src.finetune --task tweets --mode full --seed 0
  python -m src.finetune --task tweets --mode lora --r 8 --targets qv --seed 0
"""
import argparse
import os
import time

import numpy as np
import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

from .utils import compute_metrics_np, load_task, save_result, set_seed


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", default="tweets", choices=["tweets", "news"])
    p.add_argument("--mode", required=True, choices=["full", "lora"])
    p.add_argument("--model", default="xlm-roberta-base")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--bs", type=int, default=32)
    p.add_argument("--max_len", type=int, default=128)
    p.add_argument("--wd", type=float, default=0.01)
    p.add_argument("--r", type=int, default=8)
    p.add_argument("--alpha", type=int, default=None, help="por defecto 2*r")
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--targets", default="qv", choices=["qv", "all"])
    p.add_argument("--out", default="results")
    p.add_argument("--save_model", action="store_true")
    a = p.parse_args()

    lr = a.lr or (2e-5 if a.mode == "full" else 4e-4)
    tag = "full" if a.mode == "full" else f"lora_r{a.r}_{a.targets}"
    name = f"{a.task}_{tag}_seed{a.seed}"
    if os.path.exists(os.path.join(a.out, name + ".json")):
        print("ya existe", name)
        return

    set_seed(a.seed)
    ds, label_names = load_task(a.task)
    n_labels = len(set(ds["train"]["label"]))
    tok = AutoTokenizer.from_pretrained(a.model)
    enc = ds.map(
        lambda b: tok(b["text"], truncation=True, max_length=a.max_len),
        batched=True,
        remove_columns=[c for c in ds["train"].column_names if c not in ("label",)],
    ).rename_column("label", "labels")

    id2label = {i: (label_names[i] if label_names else f"clase_{i}") for i in range(n_labels)}
    model = AutoModelForSequenceClassification.from_pretrained(
        a.model, num_labels=n_labels, id2label=id2label,
        label2id={v: k for k, v in id2label.items()},
    )
    if a.mode == "lora":
        from peft import LoraConfig, TaskType, get_peft_model

        cfg = LoraConfig(
            task_type=TaskType.SEQ_CLS, r=a.r, lora_alpha=a.alpha or 2 * a.r,
            lora_dropout=a.dropout,
            target_modules=["query", "value"] if a.targets == "qv" else "all-linear",
        )  # el clasificador se entrena completo (modules_to_save automático en SEQ_CLS)
        model = get_peft_model(model, cfg)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"{name}: entrenables {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

    def metrics(ep):
        return compute_metrics_np(ep.predictions.argmax(-1), ep.label_ids)

    def metrics_trainer(ep):
        m = metrics(ep)
        return {"acc": m["acc"], "macro_f1": m["macro_f1"]}

    ckpt = os.path.join("checkpoints", name)
    args = TrainingArguments(
        output_dir=ckpt, num_train_epochs=a.epochs, per_device_train_batch_size=a.bs,
        per_device_eval_batch_size=64, learning_rate=lr, weight_decay=a.wd,
        warmup_steps=int(0.1 * a.epochs * -(-len(enc['train']) // a.bs)), lr_scheduler_type="linear", optim="adamw_torch",
        eval_strategy="epoch", save_strategy="epoch", save_total_limit=1,
        load_best_model_at_end=True, metric_for_best_model="macro_f1",
        fp16=torch.cuda.is_available(), logging_strategy="epoch", seed=a.seed,
        report_to="none",
    )
    trainer = Trainer(
        model=model, args=args, train_dataset=enc["train"], eval_dataset=enc["validation"],
        data_collator=DataCollatorWithPadding(tok), compute_metrics=metrics_trainer,
    )

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    trainer.train()
    train_time = time.time() - t0
    peak = torch.cuda.max_memory_allocated() / 2**20 if torch.cuda.is_available() else 0.0

    val = metrics(trainer.predict(enc["validation"]))
    test_out = trainer.predict(enc["test"])
    test = metrics(test_out)
    print(f"VAL {val['acc']:.4f}/{val['macro_f1']:.4f}  TEST {test['acc']:.4f}/{test['macro_f1']:.4f}")

    if a.save_model:
        mdir = os.path.join(a.out, "models", name)
        trainer.model.save_pretrained(mdir)
        tok.save_pretrained(mdir)

    save_result(a.out, name, {
        "task": a.task, "method": tag, "seed": a.seed, "model": a.model, "lr": lr,
        "epochs": a.epochs, "val": val, "test": test,
        "trainable_params": trainable, "total_params": total,
        "peak_mem_mb": peak, "train_time_s": train_time,
        "history": trainer.state.log_history,
        "test_preds": test_out.predictions.argmax(-1).tolist(),
        "test_labels": [int(x) for x in test_out.label_ids],
    })
    import shutil
    shutil.rmtree(ckpt, ignore_errors=True)


if __name__ == "__main__":
    main()
