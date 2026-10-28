# LoRA vs fine-tuning completo vs prompting para análisis de sentimiento en tweets en español

Proyecto Final de Deep Learning (UVG). Compara tres estrategias de adaptación de modelos de lenguaje
sobre `cardiffnlp/tweet_sentiment_multilingual` (español, 3 clases) en una **GPU T4 de Colab gratis**:

| Estrategia | Modelo | Notas |
|---|---|---|
| Prompting sin entrenamiento (zero-shot y few-shot) | `Qwen/Qwen2.5-1.5B-Instruct` (fp16, ~3.5 GB) | predicción por logits del primer token de la etiqueta |
| Fine-tuning completo | `xlm-roberta-base` (278 M) | AdamW, lr 2e-5, wd 0.01, fp16 |
| LoRA | `xlm-roberta-base` | r=8, α=16, q/v, lr 4e-4; ablación en r y módulos |

Métricas: accuracy, F1 macro (media ± desv. sobre 5 semillas), parámetros entrenables, memoria pico de GPU, tiempo.
Segunda tarea opcional: `mteb/SpanishNewsClassification` (12 clases, splits propios 70/10/20 estratificados).
Explicabilidad: SHAP sobre los errores del modelo LoRA.

## Por qué estos modelos (T4 16 GB, sin bf16)
- **XLM-R base**: multilingüe, 278 M params; full FT cabe holgado (~4–5 GB con fp16) y es el escenario donde LoRA vs full FT es una comparación limpia. Se evita `cardiffnlp/twitter-xlm-roberta-base-sentiment` porque ya fue entrenado con estos tweets (fuga de datos).
- **Qwen2.5-1.5B-Instruct** para prompting: un encoder no puede recibir prompts; este LLM cabe en fp16 (la T4 no soporta bf16) y entiende español.
- Alternativas descartadas: mBERT (peor que XLM-R en general), XLM-R large / LLMs 7B (full FT no cabe sin trucos, desvía el foco del proyecto).

## Reproducir
Abrir `notebooks/proyecto_final_colab.ipynb` en Colab con GPU T4 y ejecutar todo, o:
```bash
pip install -r requirements.txt
python run_all.py          # idempotente; resultados en results/*.json
python -m src.aggregate    # tabla + figuras
```
Estructura: `src/` (código), `run_all.py` (experimento completo), `results/` (json, tablas, SHAP), `figures/`, `docs/` (informe, guion, slides).

Licencias de datos: CC BY 3.0 (tweets; respetar términos de Twitter) y MIT (noticias).

## Resultados (T4, test de 870 tweets, F1 macro)
| Método | F1 macro | Params entrenables | Mem. pico | Tiempo |
|---|---|---|---|---|
| Zero-shot Qwen2.5-1.5B | 0.478 | 0 | 3.6 GB | 21 s (inferencia) |
| Few-shot (3 por clase) | 0.479 ± 0.026 | 0 | 5.3 GB | 93 s (inferencia) |
| Fine-tuning completo XLM-R | 0.667 ± 0.012 | 278 M | 5.4 GB | 422 s |
| LoRA r=8 | 0.673 ± 0.013 | 0.89 M | 1.9 GB | 33 s |

Tabla completa en `results/tabla_comparativa.md`; análisis en `docs/informe.pdf`; presentación en `docs/presentacion.pptx`.
Nota: `results/explain/shap_text.html` (37 MB) no se versiona (ver `.gitignore`).
