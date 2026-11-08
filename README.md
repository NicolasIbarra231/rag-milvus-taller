# Asistente RAG · Milvus standalone + LLM locales (Ollama)

## 1. Arquitectura

```
                 ┌──────────────┐  HTTP   ┌──────────────────────────────┐
 navegador ─────►│  frontend    │────────►│  api (FastAPI)               │
  :8501          │  Streamlit   │         │  · encoder E5 (preguntas)    │
                 └──────────────┘         │  · recuperar ─► Milvus       │
                                          │  · generar   ─► Ollama       │
                                          │  · validar citas + log JSONL │
                                          └───────┬───────────┬──────────┘
                                                  │           │
                         ┌────────────────────────▼──┐   ┌────▼─────────────┐
                         │ milvus (standalone,        │   │ ollama           │
                         │  etcd embebido + disco)    │   │ qwen2.5 / llama  │
                         └────────────────────────────┘   │ (GPU NVIDIA opc.)│
 ingesta: docker compose exec api python -m scripts.ingest └──────────────────┘
```

| Servicio | Imagen | Responsabilidad | Volumen persistente |
|---|---|---|---|
| `milvus` | `milvusdb/milvus:v2.6.6` | **Único** motor vectorial (standalone con etcd embebido y almacenamiento local) | `milvus_data` |
| `ollama` | `ollama/ollama` | LLM locales | `ollama_data` (pesos) |
| `api` | `rag-milvus-taller` (este repo) | Orquestación RAG, encoder, logs | `hf_cache` (encoder), `./reports` |
| `frontend` | `rag-milvus-taller` (misma imagen) | Interfaz web | — |


## 2. Requisitos

- Docker Engine con Compose v2 (`docker compose version`).
- **RAM:** 16 GB recomendados (los límites por defecto suman ~14 GB de techo; el uso real es menor).
- **Disco:** del orden de 15 GB libres (imágenes de Docker, tres LLM ~4,5 GB, encoder ~0,5 GB y datos).
- **GPU (opcional, recomendada):** NVIDIA con driver reciente +
  [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
  en Linux, o Docker Desktop con WSL2 en Windows. Verificación:
  ```bash
  docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
  ```
  Los tres modelos del experimento pesan entre 1 y 2 GB cada uno; se cargan de a uno.

## 3. Puesta en marcha desde cero

```bash
git clone <URL_DEL_REPOSITORIO> && cd <repositorio>
cp .env.example .env                 # Windows: copy .env.example .env
```

Copie a `data/` los archivos descritos en [`data/README.md`](data/README.md)
(como mínimo `encoder_config.json` y `vectores.parquet` **o** el Parquet del corpus).

**Con GPU NVIDIA (configuración de la demo):**
```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d --build
```
**Solo CPU:**
```bash
docker compose up -d --build
```

Espere a que todo esté `healthy` (la primera vez la API descarga el encoder, ~2 min):
```bash
docker compose ps
```

Descargue los LLM del experimento (una sola vez; quedan en el volumen `ollama_data`):
```bash
docker compose exec ollama ollama pull qwen2.5:1.5b
docker compose exec ollama ollama pull llama3.2:1b
docker compose exec ollama ollama pull llama3.2:3b
```

Cargue los vectores en Milvus (sección 4) y abra:

- Interfaz: <http://localhost:8501>
- Documentación interactiva de la API: <http://localhost:8000/docs>

> Para no escribir `-f … -f …` en cada comando con GPU, agregue a `.env`:
> `COMPOSE_FILE=docker-compose.yml:docker-compose.gpu.yml` (en Windows el separador es `;`).

## 3. API e interfaz

| Método | Ruta | Descripción |
|---|---|---|
| GET | `/health/live` | El proceso responde (healthcheck de Docker) |
| GET | `/health` | Estado de Milvus (conteo, dimensión), Ollama y encoder: `ok` o `degradado` |
| GET | `/config` | Parámetros por defecto, prompts con su huella, contrato del encoder |
| GET | `/models` | Modelos descargados y en memoria; faltantes del experimento |
| POST | `/models/load` · `/models/unload` | Precalentar / liberar un modelo |
| POST | `/retrieve` | Solo recuperación (sin LLM) |
| POST | `/ask` | Recuperación + generación + validación de citas |

```bash
curl -s localhost:8000/ask -H 'Content-Type: application/json' -d '{
  "query": "¿Qué impactos del cambio climático se mencionan en las noticias?",
  "model": "llama3.2:3b", "k": 5, "estrategia": "diversificada", "prompt_version": "rag_v5",
  "filtros": {"colecciones": ["url_crawls"], "idioma": "es"}
}'
```

## 4. Pruebas

```bash
# Unitarias e integración (Milvus Lite + Ollama simulado; no descargan modelos): 45 pruebas
# (2 comparan contra scikit-learn y se omiten si no está instalado)
docker compose exec api pytest

# Prueba de humo contra la pila real (salud, recuperación, citas, abstención, errores)
docker compose exec api python -m scripts.smoke_test --modelos qwen2.5:1.5b llama3.2:1b llama3.2:3b
```
## 5. Persistencia

Los datos sobreviven a `docker compose down` y a reinicios del equipo porque viven en
volúmenes con nombre (`docker volume ls`). Comprobación:

```bash
docker compose exec api python -m scripts.ingest --verificar   # anote "registros"
docker compose down && docker compose up -d                     # (agregue -f del GPU si aplica)
docker compose exec api python -m scripts.ingest --verificar   # mismo número
```

- `docker compose down -v` **borra** volúmenes: Milvus, pesos de LLM y encoder (empezar de cero).
- **Windows:** no ubique el proyecto dentro de OneDrive. Los archivos "solo en la nube" no
  son legibles por Docker y la sincronización interfiere con los volúmenes montados
  (`data/`, `reports/`, `milvus/`). Use, por ejemplo, `C:\proyectos\rag_milvus_taller`.
- Reindexar sin perder los modelos: `python -m scripts.ingest --from-vectors … --recrear`.

## 6. Logs y trazabilidad

| Qué | Dónde | Cómo verlo |
|---|---|---|
| Logs de servicio (rotados: 3 × 10 MB) | Docker | `docker compose logs -f api` (o `ollama`, `milvus`) |
| Una línea JSON por pregunta, incluidos los errores | `reports/logs/rag_runs.jsonl` | `tail -n 1 reports/logs/rag_runs.jsonl` |
| Resumen por modelo (p50/p95, tokens/s, tasa de error, abstención) | — | `docker compose exec api python -m scripts.resumen_logs --excluir-frio --csv reports/resumen_latencias.csv` |

## 7. Reproducibilidad

- Versiones de Python fijadas en `requirements.txt`; imagen de Milvus fijada por etiqueta (`v2.6.6`).
- Al terminar los experimentos, fije Ollama por digest en `.env` (`OLLAMA_IMAGE`):
  `docker image inspect ollama/ollama --format '{{index .RepoDigests 0}}'`.
- Registre el digest de cada LLM (`ollama list`), el sha256 del Parquet (lo imprime `--from-corpus`)
  y las huellas de muestra y vectores.
- Un cambio de encoder exige **otra** `MILVUS_COLLECTION` y regenerar los vectores.
- Un cambio de prompt exige una **versión nueva** en `app/prompts.py`; nunca editar una existente.

## 8. Estructura

```
app/
  config.py        configuración por entorno + contrato del encoder
  encoder.py       codificar(): único punto de codificación (query/passage)
  chunking.py      struct/fixed con offsets e IDs estables (Fase 2)
  corpus.py        universo, muestra estratificada, segmentación y deduplicación
  milvus_store.py  esquema explícito, upsert, búsqueda, filtros seguros
  retrieval.py     recuperación convencional y diversificada
  prompts.py       prompts versionados, presupuesto de contexto, validación de citas
  llm.py           cliente de Ollama con errores tipados
  rag.py           orquestación + registro JSONL
  evaluacion.py    métricas de recuperación a nivel de documento (gold estándar)
  api.py           FastAPI
  frontend.py      Streamlit
scripts/  ingest.py · smoke_test.py · resumen_logs.py · evaluar_gold.py
tests/    45 pruebas (pytest)
milvus/   embedEtcd.yaml · user.yaml (configuración de Milvus embebido)
docker-compose.yml · docker-compose.gpu.yml · Dockerfile · .env.example
```

**Datos:** no publicar el Parquet ni `vectores.parquet` (contienen texto del corpus);
están excluidos en `.gitignore`.
