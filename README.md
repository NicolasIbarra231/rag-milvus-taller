# Asistente RAG · Milvus standalone + LLM locales (Ollama)

Taller de investigación aplicada — Maestría en Inteligencia Artificial, Pontificia Universidad Javeriana.

Este repositorio lleva a producción (Docker) la lógica validada en el notebook
`taller_rag_preparacion_datos.ipynb` **sin cambiar la receta**: los mismos chunks
`struct256o32`, el mismo encoder y contrato de codificación, la misma recuperación
diversificada, los mismos prompts versionados (`rag_v5` por defecto) y los mismos
parámetros de generación del experimento principal. Parte del boilerplate oficial
[`puj-nlp/boilerplate_rag`](https://github.com/puj-nlp/boilerplate_rag).

No requiere servicios de pago ni APIs comerciales.

---

## 1. Arquitectura

Diagramas de componentes y de secuencia, decisiones y módulos: [`docs/arquitectura.md`](docs/arquitectura.md).

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

**¿Por qué no hay contenedores de etcd y MinIO?** El boilerplate del curso levantaba
Milvus standalone con etcd (metadatos) y MinIO (almacenamiento de segmentos) como
servicios aparte. La imagen oficial `minio/minio` dejó de distribuirse en Docker Hub
(septiembre de 2026), por lo que `docker compose up` fallaba con *pull access denied*.
Se adoptó el **modo embebido oficial de Milvus standalone** (el del script
`standalone_embed.sh`): etcd corre dentro del contenedor de Milvus y los segmentos se
guardan en disco local (`COMMON_STORAGETYPE=local`), todo en el volumen `milvus_data`.
Ventajas: no depende de imágenes de terceros, usa menos RAM y tiene menos piezas que
fallar. Costo: no escala a varios nodos, algo innecesario para este corpus. Milvus
sigue siendo el único motor vectorial y la API no cambia.

**¿Dónde vive el encoder?** En la API (para las preguntas) y en el trabajo de ingesta
(para los chunks). Ambos importan la misma función `app.encoder.codificar()` y leen el
mismo contrato `data/encoder_config.json` generado en la Fase 3 (modelo, prefijos
`query: `/`passage: `, normalización, dimensión 384, métrica COSINE). La ingesta
`--from-vectors` se niega a cargar vectores de un contrato distinto. Así la codificación
de preguntas y documentos es idéntica.

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

**Atajo:** después de copiar los datos a `data/`, un solo script levanta todo, descarga
los modelos, carga Milvus si está vacío y verifica la carga (es idempotente):

```bash
./scripts/inicializar.sh --gpu --build          # Linux / macOS / WSL (sin --gpu: solo CPU)
powershell -ExecutionPolicy Bypass -File .\scripts\inicializar.ps1 -Gpu -Build   # Windows
```

Paso a paso, lo mismo es:

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

## 4. Ingesta en Milvus (idempotente)

El `id` de cada chunk es un hash estable (documento + `struct256o32` + offsets) y la
carga usa `upsert`: reejecutar **no duplica**. `--recrear` borra y reconstruye la colección.

**Modo A — mismos vectores del experimento (rápido, no recalcula embeddings):**
```bash
docker compose exec api python -m scripts.ingest --from-vectors data/vectores.parquet
```
Debe imprimir la huella de vectores **`a022d3640ac21aa0`** (la misma de la Fase 3),
`conteo_ok: true` (2.248 chunks) y `busqueda_control_ok: true`.

**Modo B — desde cero, desde el corpus:**
```bash
docker compose exec api python -m scripts.ingest --from-corpus data/documentos_rag.parquet --muestra 500
# o fijando exactamente los IDs del notebook:
docker compose exec api python -m scripts.ingest --from-corpus data/documentos_rag.parquet \
    --ids data/muestras/muestra_500_seed42.txt
```
Reproduce universo de muestreo, muestra estratificada (semilla 42), chunking y embeddings.
La huella de IDs debe ser **`4397e40ea9ff0408`** (la de la Fase 2). Para la muestra de
10.000: `--muestra 10000` (en CPU tarda varios minutos; los embeddings van por lotes).
Con `--guardar-vectores reports/vectores_10000.parquet` deja los vectores para el Modo A.

**Verificar la colección en cualquier momento:**
```bash
docker compose exec api python -m scripts.ingest --verificar
```

## 5. Gestión de los LLM (descargar, cargar, liberar, borrar)

| Acción | Comando | Efecto |
|---|---|---|
| Descargar al disco | `docker compose exec ollama ollama pull llama3.2:3b` | Pesos al volumen `ollama_data` |
| Ver descargados | `docker compose exec ollama ollama list` | Tamaño, digest |
| Ver en memoria | `docker compose exec ollama ollama ps` | RAM/VRAM ocupada |
| Cargar en memoria (precalentar) | `curl -X POST localhost:8000/models/load -H 'Content-Type: application/json' -d '{"model":"llama3.2:3b"}'` o botón **Precalentar** | Evita medir el arranque en frío |
| Liberar memoria | `curl -X POST localhost:8000/models/unload -H 'Content-Type: application/json' -d '{"model":"llama3.2:3b"}'` o botón **Liberar** | Saca el modelo de RAM/VRAM sin borrarlo |
| Borrar del disco | `docker compose exec ollama ollama rm gemma2:2b` | Libera espacio |
| Cuantización y familia | `docker compose exec ollama ollama show qwen2.5:1.5b` | Para el informe |

`OLLAMA_MAX_LOADED_MODELS=1`: al cambiar de modelo se libera el anterior (evita quedarse
sin VRAM al alternar los tres LLM en la demo). `OLLAMA_KEEP_ALIVE=10m`: tiempo que un
modelo permanece cargado tras usarse.

## 6. API e interfaz

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

La respuesta de `/ask` incluye `respuesta`, `modelo`, `evidencias` (con `url`, `title`,
offsets, `score` y si fue `citada`), `citas` (estado: `referencias_existentes`,
`respuesta_sin_citas`, `referencias_inexistentes`, `abstencion`), `tiempos`
(codificación, búsqueda, recuperación, generación, carga del modelo, total), `tokens`
(entrada, salida, tokens/s), `contexto` (caracteres, si se truncó, huella) y `advertencias`.

**Errores** (siempre `{"error", "detalle", "sugerencia"}`):

| HTTP | `error` | Causa típica |
|---|---|---|
| 404 | `modelo_no_descargado` | Falta `ollama pull` |
| 409 | `coleccion_vacia` | No se ha ejecutado la ingesta |
| 422 | validación / `parametro_invalido` | Pregunta vacía, `k` fuera de 1–15, filtro no permitido |
| 503 | `ollama_no_disponible` / `milvus_no_disponible` | Contenedor caído |
| 504 | `ollama_timeout` | Arranque en frío o falta de memoria |

**Interfaz (Streamlit):** estado de los servicios; selector **dinámico** de modelos (lee
`/models`, marca con ★ los del experimento y muestra cómo descargar los faltantes);
`k`, estrategia, filtros, versión de prompt, temperatura y máximo de tokens; botones
precalentar/liberar; estados de carga; respuesta con métricas de tiempo, estado de las
citas y advertencias; evidencias con enlaces; pestaña **Comparar modelos** (misma
pregunta con varios LLM y verificación de que todos recibieron el mismo contexto) y
pestaña **Solo recuperación**.

## 7. Pruebas

```bash
# Unitarias e integración (Milvus Lite + Ollama simulado; no descargan modelos): 53 pruebas
# (2 comparan contra scikit-learn y se omiten si no está instalado)
docker compose exec api pytest

# Prueba de humo contra la pila real (salud, recuperación, citas, abstención, errores)
docker compose exec api python -m scripts.smoke_test --modelos qwen2.5:1.5b llama3.2:1b llama3.2:3b
```

Cubren: invariantes del chunking (offsets, sin partir palabras, oraciones completas),
IDs estables, presupuesto de contexto, estados de citas, esquema e idempotencia en
Milvus, filtros con lista blanca (sin inyección de expresiones), recuperación
diversificada, que los dos modos de ingesta producen la **misma** colección, y los
códigos de error de la API.

## 7.1 Evaluación con el gold estándar (punto 6)

Copie el gold en `data/gold/` (`gold_v1.jsonl` y los demás archivos de la entrega). Primero
valide que corresponde a los datos indexados (debe dar una sola advertencia, G005):

```bash
docker compose exec api python data/gold/validar_gold_v1.py data/gold data/vectores.parquet
```

Luego, en orden (todo queda en `reports/eval/`):

| Paso | Comando | Tiempo en CPU | Produce |
|---|---|---|---|
| 1. Recuperación | `docker compose exec api python -m scripts.evaluar_gold recuperacion` | minutos | Recall@k, MRR, nDCG@k, hit@k por pregunta, tipo y conjunto |
| 2. Generación | `docker compose exec api python -m scripts.evaluar_gold generacion` | ~2,5-3 h (46 × 3 modelos) | Respuesta de cada modelo con evidencias, citas y tiempos (`generacion_crudo.jsonl`) |
| 3. Resumen | `docker compose exec api python -m scripts.evaluar_gold resumen` | segundos | Abstención, citas, contexto con relevante, p50/p95, tokens/s, errores |
| 4. Planilla | `docker compose exec api python -m scripts.evaluar_gold planilla` | segundos | `planilla_evaluacion.xlsx` para los dos evaluadores |
| Ablación de recuperación | `docker compose exec api python -m scripts.evaluar_gold recuperacion --k-max 8 --estrategia convencional` | minutos | Mismas métricas con la recuperación convencional (archivos `*_convencional*`), para comparar con la diversificada |
| 5. Calificaciones | `docker compose exec api python -m scripts.evaluar_gold calificaciones` | segundos | Promedios humanos por modelo y kappa entre evaluadores |

- **Reanudable:** la generación guarda cada respuesta apenas llega. Si se corta, vuelva a
  ejecutar el mismo comando y continúa donde iba. Los errores se registran y se reintentan.
- **Un modelo a la vez:** se precalienta cada modelo y se responden todas las preguntas con él
  antes de pasar al siguiente, para no medir recargas de pesos. Los arranques en frío se
  excluyen de las latencias del resumen.
- **Conjuntos:** por defecto se corren las 46 preguntas y cada métrica se reporta para `test`,
  `dev` y `todos`; la cifra principal del informe es la de `test`. `--conjunto test` corre solo esas.
- **Métricas de recuperación a nivel de documento:** cada chunk cuenta para su `document_id`
  y para los documentos de su `provenance` (chunks deduplicados). Las `sin_respuesta` se
  excluyen (no tienen relevantes). Definiciones en `app/evaluacion.py`.
- **La calificación humana es la referencia.** El detector automático de abstención solo
  reconoce frases exactas; el resumen automático sirve de apoyo, no de veredicto.
- `config_recuperacion.json` y `config_generacion.json` guardan la instantánea de lo evaluado:
  huella del gold, parámetros RAG, colección, contrato del encoder y modelos con su digest.

### Gráficas y memoria

```bash
docker compose exec api pip install matplotlib==3.9.2     # una vez; no viene en la imagen
docker compose exec api python -m scripts.graficas_eval   # escribe reports/eval/graficas/*.png
```

Memoria por modelo (PowerShell): carga cada modelo, hace una pregunta y anota CPU y RAM
de cada contenedor en `reports/eval/memoria_por_modelo.csv`:

```powershell
foreach ($m in "qwen2.5:1.5b","llama3.2:1b","llama3.2:3b") {
  docker compose exec api python -c "import requests; b='http://localhost:8000'; requests.post(b+'/models/load', json={'model':'$m'}, timeout=600); requests.post(b+'/ask', json={'query':'Que impactos del cambio climatico se mencionan?','model':'$m'}, timeout=420)"
  docker compose exec ollama ollama ps
  docker stats --no-stream --format "$m,{{.Name}},{{.CPUPerc}},{{.MemUsage}}" | Add-Content reports\eval\memoria_por_modelo.csv
}
```

El índice de todos los artefactos está en [`reports/eval/README.md`](reports/eval/README.md).

## 7.2 Evidencias de funcionamiento

Corrida del 8 de noviembre de 2026 en Windows con Docker Desktop (WSL2), inferencia en
CPU (Ollama no detectó la GPU RTX 3050; ver sección 12):

| Verificación | Resultado |
|---|---|
| Ingesta modo A | 2.248 chunks de 384 dimensiones en 1,9 s; huella `a022d3640ac21aa0` igual a la del notebook; búsqueda de control con coseno 1,0 |
| Pruebas automáticas | 45 pruebas (2 se omiten sin scikit-learn) |
| Prueba de humo | 16 de 17 casos; el único fallo es una abstención que el detector de frases no reconoce |
| Gold: recuperación (test, k = 5) | hit@1 0,70 · MRR 0,71 · nDCG@5 0,69 · Recall@5 0,66 |
| Gold: generación | 138 de 138 respuestas; exactitud humana (test) qwen2.5:1.5b 0,41 · llama3.2:3b 0,26 · llama3.2:1b 0,24 |
| Acuerdo entre evaluadores | kappa ponderado 0,97 en corrección, 0,80 en fidelidad, 0,85 en citas |
| Latencia p50 en CPU (test) | qwen2.5:1.5b 34,6 s · llama3.2:1b 48,4 s · llama3.2:3b 69,5 s |

![Recuperación por idioma](reports/eval/graficas/1_recuperacion_por_idioma.png)
![Exactitud y latencia](reports/eval/graficas/3_exactitud_latencia.png)

## 7.3 Ablaciones, robustez, entorno y memoria (todo en un comando)

`scripts/ejecutar_pendientes.ps1` corre en orden todo lo que completa la evaluación y es
reanudable: cada paso deja una marca en `reports/.estado/` y, si se interrumpe, al volver a
ejecutarlo salta lo ya hecho. Tarda de 2,5 a 3 horas en CPU; conviene dejarlo de noche con la
suspensión del equipo desactivada. Antes, en `.env`, suba `API_MEM=8g` y `API_CPUS=6` y
recree la API (`docker compose ... up -d`): bge-m3 y la muestra de 10.000 lo necesitan.

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\ejecutar_pendientes.ps1
powershell -ExecutionPolicy Bypass -File .\scripts\ejecutar_pendientes.ps1 -Solo robustez,memoria   # solo algunos pasos
```

| Paso | Qué hace | Salida |
|---|---|---|
| `entorno` | CPU, RAM, GPU, Windows, Docker, WSL, `.wslconfig`, límites y metadatos de cada modelo (cuantización, digest) | `reports/entorno/` |
| `corpus` | Descarga `dragons_rag` (versión 2026-10-19) con `open_corpus_co_es` y verifica el SHA-256 publicado | `data/documentos_rag.parquet` (no se publica) |
| `base_corpus` | Reconstruye struct256o32 + e5-small desde el corpus, en otra colección: control de reproducibilidad (huellas de muestra y de vectores) | `reports/ablaciones/` |
| `chunk128`, `chunk512` | Ablación de tamaño de chunk (128/16 y 512/64 tokens), una colección cada una | `reports/ablaciones/` |
| `bge_m3` | Ablación de encoder: `BAAI/bge-m3` (`configs/encoders/bge-m3.json`), con los mismos chunks | `reports/ablaciones/` |
| `muestra10k` | Muestra de 10.000 documentos, su colección y la evaluación del gold (con cobertura) | `reports/ablaciones/` |
| `gen_convencional`, `gen_k3` | Ablación de contexto en la generación con qwen2.5:1.5b; solo las respuestas que cambian van a una planilla nueva | `reports/eval/*_ctx_convencional*`, `*_k3*` |
| `robustez` | 4 ataques de inyección en documentos × 3 preguntas y 3 casos de fuentes contradictorias, con los 3 modelos | `reports/robustez/` |
| `memoria` | RAM y CPU de cada contenedor durante la generación de cada modelo | `reports/eval/memoria_por_modelo.csv` |
| `resumen` | Resúmenes, gráficas y tablas de ablaciones | `reports/eval/graficas/`, `reports/ablaciones/` |

Las ablaciones de recuperación usan `scripts.evaluar_gold recuperacion --local`: la API se
ejecuta dentro del mismo proceso con la colección y el encoder de las variables de entorno
(`MILVUS_COLLECTION`, `ENCODER_CONFIG`), sin tocar el servidor ni la colección del experimento.
El gold se evalúa a nivel de documento, así que sus juicios valen para cualquier tamaño de chunk.

Con temperatura 0 y semilla fija, una respuesta solo cambia si cambia su contexto. Por eso
`planilla-ablacion` reutiliza la nota de la corrida base cuando el contexto y el texto son
idénticos, y deja en `planilla_<condición>.xlsx` solo las respuestas nuevas. Calificadas esas,
`calificaciones-ablacion --etiqueta <condición>` da las métricas humanas de la condición.

## 8. Persistencia

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

## 9. Logs y trazabilidad

| Qué | Dónde | Cómo verlo |
|---|---|---|
| Logs de servicio (rotados: 3 × 10 MB) | Docker | `docker compose logs -f api` (o `ollama`, `milvus`) |
| Una línea JSON por pregunta, incluidos los errores | `reports/logs/rag_runs.jsonl` | `tail -n 1 reports/logs/rag_runs.jsonl` |
| Resumen por modelo (p50/p95, tokens/s, tasa de error, abstención) | — | `docker compose exec api python -m scripts.resumen_logs --excluir-frio --csv reports/resumen_latencias.csv` |

Cada registro guarda `run_id`, fecha, pregunta, modelo, respuesta, `prompt_version` y su
huella, `k`, estrategia, filtro, temperatura, semilla, `num_ctx`, `num_predict`, tiempos,
tokens, huella del contexto, encoder, colección y los `chunk_id` usados (no su texto). Es
el insumo crudo para la evaluación (literal 6).

## 10. Límites de recursos

Definidos en `.env` y aplicados con `deploy.resources.limits` en `docker-compose.yml`:

| Servicio | CPU | RAM | Comentario |
|---|---|---|---|
| milvus | 2 | 4 GB | Incluye etcd embebido; suficiente para la muestra de 10.000 documentos |
| ollama | 4 | 6 GB | Con GPU los pesos van a VRAM; la RAM es respaldo |
| api | 2 | 3 GB | torch CPU + E5 (~1 GB) |
| frontend | 1 | 512 MB | |

Medición durante los experimentos (para la tabla de rendimiento del informe):
```bash
docker stats --no-stream            # CPU y RAM por contenedor
nvidia-smi                          # VRAM y uso de GPU (host)
docker compose exec ollama ollama ps   # qué modelo ocupa memoria y cuánta
```

Si un contenedor se reinicia con `OOMKilled` (`docker inspect rag-ollama --format '{{.State.OOMKilled}}'`),
suba su límite en `.env` y ejecute `docker compose up -d` de nuevo.

## 11. Reproducibilidad

- Versiones de Python fijadas en `requirements.txt`; imagen de Milvus fijada por etiqueta (`v2.6.6`).
- Al terminar los experimentos, fije Ollama por digest en `.env` (`OLLAMA_IMAGE`):
  `docker image inspect ollama/ollama --format '{{index .RepoDigests 0}}'`.
- Registre el digest de cada LLM (`ollama list`), el sha256 del Parquet (lo imprime `--from-corpus`)
  y las huellas de muestra y vectores.
- Un cambio de encoder exige **otra** `MILVUS_COLLECTION` y regenerar los vectores.
- Un cambio de prompt exige una **versión nueva** en `app/prompts.py`; nunca editar una existente.

## 12. Solución de problemas

| Síntoma | Causa probable | Solución |
|---|---|---|
| `milvus` nunca queda `healthy` | Poca RAM, puerto ocupado o falta `milvus/embedEtcd.yaml` | `docker compose logs milvus`; suba `MILVUS_MEM`; libere 19530/9091; verifique que exista la carpeta `milvus/` |
| `pull access denied for minio/minio` | Versión vieja del compose (la del boilerplate) | Use el `docker-compose.yml` de este repositorio (Milvus embebido, sin MinIO) |
| `docker: error during connect … docker_engine` | Docker Desktop no está abierto | Abrirlo y esperar "Engine running"; `docker version` debe mostrar **Server** |
| UI: "La colección está vacía" | Falta la ingesta | Sección 4 |
| `modelo_no_descargado` | Falta `ollama pull` | Sección 5 |
| Primera respuesta muy lenta / `ollama_timeout` | Arranque en frío | Botón **Precalentar**; suba `OLLAMA_TIMEOUT_SECONDS` |
| `ollama ps` muestra `100% CPU` con GPU | El override GPU no se aplicó o falta el toolkit | Levante con `-f docker-compose.gpu.yml`; pruebe `docker run --gpus all … nvidia-smi` |
| `could not select device driver "nvidia"` | NVIDIA Container Toolkit no instalado | Instálelo o use el modo CPU |
| La API no arranca: "contrato" | `encoder_config.json` con otra dimensión/modelo | Use el contrato de la Fase 3 o cree otra colección |
| `--from-vectors` rechaza el archivo | Vectores de otro contrato | No mezclar: regenerar con `--from-corpus` |
| La API tarda en estar `healthy` la primera vez | Descarga del encoder (~470 MB) | Esperar; queda en `hf_cache` |
| El `build` se queda horas en `pip install torch` (Windows) | El antivirus (en nuestro caso Kaspersky) analiza cada archivo que se escribe en el disco de WSL2, o falta RAM | Pausar el antivirus durante el `build` o excluir las carpetas de Docker; cerrar programas pesados; `wsl --shutdown` y reintentar con `docker compose --progress plain build api` |
| `request returned Internal Server Error for API route` en cualquier comando | El motor de Docker Desktop dejó de responder, típicamente tras suspender el equipo | Salir de Docker Desktop, `wsl --shutdown`, abrirlo de nuevo; desactivar la suspensión durante corridas largas. Los volúmenes y `reports/` no se pierden |
| Ollama registra `inference compute … library=cpu` aunque `nvidia-smi` funciona en un contenedor | Ollama no detecta la GPU en Docker Desktop con WSL2 (visto con driver 610.62) | Sin resolver en nuestro equipo; el sistema funciona en CPU. Para diagnosticar: `OLLAMA_DEBUG=1` en `.env` y recrear `ollama` |
| `ReadTimeout` en `scripts.evaluar_gold generacion` | La API dejó de responder (Docker detenido o equipo suspendido) | Recuperar Docker y ejecutar el mismo comando: continúa donde iba y repite las fallidas |
| `Worksheet named 'Evaluador 2' not found` | Se renombró o se copió la hoja de la planilla | Renombrar la hoja a `Evaluador 2` (o `Evaluador 1`) |
| `inicializar.ps1 no se puede cargar … ejecución de scripts deshabilitada` | Política de ejecución de PowerShell | Ejecutarlo con `powershell -ExecutionPolicy Bypass -File .\scripts\inicializar.ps1` |
| Archivos montados que no aparecen o builds erráticos en Windows | El proyecto está dentro de una carpeta sincronizada por OneDrive | Mover el proyecto a una carpeta local, por ejemplo `C:\proyectos\rag_milvus_taller` |

## 13. Estructura

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
scripts/  ingest.py · smoke_test.py · resumen_logs.py · evaluar_gold.py · graficas_eval.py
          robustez.py · resumen_ablaciones.py · descargar_corpus.py
          inicializar.sh · inicializar.ps1 (puesta en marcha de punta a punta)
          ejecutar_pendientes.ps1 · registrar_entorno.ps1 · medir_memoria.ps1 (ablaciones, entorno, memoria)
configs/  encoders/bge-m3.json (contrato del encoder de la ablación)
docs/     arquitectura.md (diagramas y decisiones) · EVALUACION.md (protocolo)
data/gold/  gold estándar (46 preguntas), anotaciones y validador
reports/  logs/rag_runs.jsonl (trazas, con versión de Ollama y digest por ejecución) · eval/ (gold)
          ablaciones/ · robustez/ · entorno/ (hardware y versiones)
tests/    53 pruebas (pytest)
milvus/   embedEtcd.yaml · user.yaml (configuración de Milvus embebido)
docker-compose.yml · docker-compose.gpu.yml · Dockerfile · .env.example
```

**Datos:** no publicar el Parquet ni `vectores.parquet` (contienen texto del corpus);
están excluidos en `.gitignore`.
