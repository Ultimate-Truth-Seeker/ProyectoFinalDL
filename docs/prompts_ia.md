# Historial de prompts e uso crítico de IA (Claude Code)

Completar/editar con sus propias observaciones; lo siguiente es el registro real de esta sesión.

| # | Prompt (resumen) | Resultado | Evaluación |
|---|---|---|---|
| 1 | Adjuntar enunciado del proyecto + propuesta + datasets; "asume solo T4 gratis, ayúdame a elegir el modelo y completar el proyecto". | Eligió XLM-R base (full FT vs LoRA) y Qwen2.5-1.5B-Instruct (prompting); descartó el modelo `cardiffnlp/twitter-xlm-roberta-base-sentiment`. | Satisfactorio. Justificación verificable: la T4 no tiene bf16, 16 GB bastan para full FT de 278 M; el modelo de Cardiff fue entrenado con estos tweets (fuga). |
| 3 | Se entregaron el notebook ejecutado en Colab y el zip de resultados; "ayúdame a completar lo que falta con los resultados". | Informe con resultados reales, análisis SHAP, slides y PDF. | Se contrastó cada cifra contra `tabla_comparativa.md`. En un borrador la IA escribió un conteo de errores que no salía de los datos (se corrigió a 288 desde la matriz de confusión) y atribuyó palabras a ejemplos sin verificar; se suavizó citando solo `palabras_errores.csv`. |
| 2 | (Implícito) escribir código de fine-tuning, LoRA, prompting, SHAP, agregación. | Código modular en `src/`. | Parcialmente correcto: **falló** al probarlo (ver abajo). Se probó en CPU con un modelo diminuto antes de entregar. |

## Errores de la IA detectados al ejecutar
1. Supuso que `load_dataset("cardiffnlp/tweet_sentiment_multilingual")` funcionaría; el dataset usa script de carga, no soportado por `datasets>=4`. Se consultó el script del repo y se leyeron los `.jsonl` directamente.
2. Las etiquetas del jsonl llegaban como **strings** ("0"), lo que rompía el collator; se castearon a int.
3. Usó `warmup_ratio`, argumento eliminado en transformers 5; se reemplazó por `warmup_steps` calculado.
4. La ficha dice 1,838/323/869 ejemplos; la descarga real trae 1,839/324/870 (se documentó).
5. En el informe, un número de errores (328) fue inventado/mal tomado de una corrida de prueba; se corrigió contra la matriz de confusión real.
Lección: la IA genera código plausible pero asume versiones de librerías; hay que ejecutar y validar.

## Cómo se evaluó
Pruebas de humo en CPU (entrenar, guardar, agregar, SHAP), verificación del tokenizador de Qwen (tokens iniciales de etiquetas distintos), y comparación contra documentación. Las métricas finales provienen de las corridas reales en Colab, no de la IA.
