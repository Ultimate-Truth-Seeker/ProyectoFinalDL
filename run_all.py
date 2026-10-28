"""Ejecuta todo el experimento (idempotente: salta lo que ya existe en results/)."""
import subprocess
import sys

py = [sys.executable, "-m"]
run = lambda *a: subprocess.run(py + list(a), check=True)

MAIN_SEEDS, ABL_SEEDS = range(5), range(3)

run("src.prompting", "--shots", "0")
for s in ABL_SEEDS:
    run("src.prompting", "--shots", "3", "--seed", str(s))

for task in ["tweets", "news"]:
    for s in MAIN_SEEDS if task == "tweets" else ABL_SEEDS:
        sm = ["--save_model"] if (task == "tweets" and s == 0) else []
        run("src.finetune", "--task", task, "--mode", "full", "--seed", str(s), *sm)
        run("src.finetune", "--task", task, "--mode", "lora", "--r", "8", "--targets", "qv", "--seed", str(s), *sm)

# ablación de LoRA (solo tweets)
for s in ABL_SEEDS:
    for r, tg in [(2, "qv"), (16, "qv"), (8, "all")]:
        run("src.finetune", "--task", "tweets", "--mode", "lora", "--r", str(r), "--targets", tg, "--seed", str(s))

run("src.aggregate")
run("src.explain", "--model_dir", "results/models/tweets_lora_r8_qv_seed0")
