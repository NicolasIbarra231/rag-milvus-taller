# Carpeta `data/` (no se versionan los datos)

Se monta en la API como **solo lectura** (`/work/data`). Coloque aquí:

| Archivo | De dónde sale | Para qué |
|---|---|---|
| `encoder_config.json` | Notebook, Fase 3: `salidas/embeddings/struct256o32/encoder_config.json` | Contrato de codificación (modelo, prefijos, normalización, dimensión). **Sí se puede versionar.** |
| `vectores.parquet` | Notebook, Fase 3: `salidas/embeddings/struct256o32/vectores.parquet` | Ingesta rápida `--from-vectors` (mismos vectores del experimento). |
| `documentos_rag.parquet` | Corpus `dragons_rag` (librería `open_corpus_co_es`) | Ingesta desde cero `--from-corpus`. |
| `gold/gold_v1.jsonl` (+ CSV, anotaciones y validador) | Responsable del punto 6 | Evaluación con `scripts.evaluar_gold` (README, sección 7.1). |
| `muestras/muestra_500_seed42.txt` | Notebook, Fase 2: `salidas/muestras/` | Opcional: fija exactamente los mismos 500 documentos. |

`*.parquet` y `muestras/` están en `.gitignore`: contienen texto del corpus, que no se publica sin autorización.
