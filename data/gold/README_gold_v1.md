# Gold estándar v1 — Taller RAG (Grupo 5)

46 preguntas con respuesta verificada para evaluar el sistema RAG (recuperación en Milvus + generación con LLM locales). Cubre lo que pide el punto 6 del taller: ambas colecciones, varios idiomas y fechas, consultas directas y comparativas, casos ambiguos y preguntas sin respuesta documental suficiente, con división en desarrollo y prueba.

## Archivos

| Archivo | Para qué |
|---|---|
| `gold_v1.csv` | Las 46 preguntas, una por fila (UTF-8). Se abre directo en pandas o Google Sheets; en Excel, impórtalo como UTF-8 o usa el `.xlsx` de anotaciones. |
| `gold_v1.jsonl` | El mismo contenido con estructura (listas y diccionarios). **Recomendado para el código de evaluación.** |
| `gold_v1_anotaciones.xlsx` | Trazabilidad de la revisión: borrador de Gemini, versión corregida, decisión humana y motivo de cada pregunta (incluye las 7 descartadas). |
| `validar_gold_v1.py` | Verifica el gold contra `vectores.parquet` (IDs, evidencias textuales, relevantes, tipos, fugas entre dev y test). |

## Sobre qué datos está hecho (importante)

- Muestra piloto de **500 documentos** (semilla 42, huella de IDs `4397e40ea9ff0408`), chunking `struct256o32` (**2.248 chunks**) y vectores `multilingual-e5-small` (huella `a022d3640ac21aa0`): exactamente los del notebook de preparación.
- Si se indexa otra muestra (por ejemplo, la de 10.000): hay que volver a verificar las 6 preguntas `sin_respuesta` (con más documentos el dato podría aparecer) y revisar si surgen nuevos documentos relevantes, sobre todo en las ambiguas.

## Columnas

| Columna | Qué contiene |
|---|---|
| `gold_id` | Identificador estable (G001…G053; los números que faltan son preguntas descartadas). |
| `split` | `dev` o `test`. |
| `tipo` | `directa`, `temporal`, `comparativa`, `ambigua` o `sin_respuesta` (ver abajo). |
| `idioma_pregunta` | `es` o `en`. |
| `pregunta` | Lo único que recibe el sistema. |
| `respuesta_ref` | Respuesta correcta (en ambiguas y sin respuesta, describe cómo debe responder el sistema). |
| `criterio` | Regla para calificar: cuándo una respuesta cuenta como correcta y qué errores la invalidan. |
| `relevantes` | JSON `{document_id: grado}`. Grado 2 = el documento responde; 1 = responde en parte o es una de varias interpretaciones. **Vacío en `sin_respuesta`.** |
| `n_relevantes` | Número de documentos relevantes. |
| `rol_documentos` | `responde`, o `más parecido (no responde)` en las sin respuesta. |
| `document_ids`, `chunk_ids` | Documentos y fragmentos de donde sale la evidencia (separados por ` \| `). |
| `evidencia` | Cita textual que respalda la respuesta (una por documento, separadas por ` \| `). |
| `url`, `titulo`, `fecha_documento`, `coleccion`, `idioma_documento` | Referencia verificable y metadatos de esos documentos. |
| `origen_borrador` | `gemini` o `claude` (quién redactó el borrador). |
| `version_elegida` | `sin cambios`, `corregida`, `original de Gemini` o `nueva`. |

## Tipos de pregunta y cómo calificarlas

| Tipo | Qué prueba | Correcta si… |
|---|---|---|
| `directa` | Encontrar un dato en un documento. | Cumple el criterio. |
| `temporal` | Directa cuya respuesta es una fecha: que el sistema no confunda fechas. | Cumple el criterio. |
| `comparativa` | Recuperar **dos o más** documentos y cruzarlos. | Usa la información de todos (el criterio lo exige). |
| `ambigua` | Detectar que hay varias respuestas válidas. | Pide aclaración o cubre las interpretaciones. |
| `sin_respuesta` | No inventar (alucinación) y abstenerse. | Dice que no hay información; **incorrecta si da un dato**. |

## División dev / test

| Tipo | dev | test | total |
|---|---|---|---|
| directa | 7 | 21 | 28 |
| temporal | 1 | 4 | 5 |
| comparativa | 1 | 3 | 4 |
| ambigua | 1 | 2 | 3 |
| sin_respuesta | 2 | 4 | 6 |
| **Total** | **12** | **34** | **46** |

- Estratificada por tipo (25 % a dev, mínimo una de cada tipo), semilla 42.
- Agrupada por documento: **ningún documento de evidencia aparece en dev y en test a la vez** (las preguntas que comparten documento quedan juntas).
- `dev` (G006, G007, G015, G023, G025, G029, G030, G034, G039, G044, G048, G053): para ajustar prompt, `k`, estrategia de contexto, etc.
- `test`: **solo para la evaluación final.** No mirarlo mientras se ajusta el sistema.

## Cómo usarlo en la evaluación

- **Recuperación (Recall@k, MRR, nDCG@k, hit@k):** a nivel de `document_id`, con `relevantes` (para nDCG usar los grados). Milvus devuelve chunks: hay que pasar cada chunk a su `document_id` antes de calcular; un chunk deduplicado puede pertenecer a varios documentos (columna `provenance` del parquet). Excluir las `sin_respuesta` (no tienen relevantes).
- **Generación:** comparar la respuesta del LLM con `respuesta_ref` aplicando el `criterio` (evaluación humana o LLM como juez). **Abstención correcta:** en `sin_respuesta` debe abstenerse; en las demás no.
- **Robustez:** las `sin_respuesta` tienen trampas a propósito (por ejemplo, G052 pide el clima de Bogotá y solo hay el de Monterrey de esa misma fecha): sirven para medir alucinaciones.

## Distribución

- Colección de la evidencia: prensa 28, Bluesky 17, mixta 1.
- Idioma de la pregunta: español 39, inglés 7. Idioma de los documentos: inglés 26, español 19, mixto 1 (hay preguntas en español sobre documentos en inglés a propósito, para probar la recuperación entre idiomas).
- Años de los documentos de evidencia: 2017: 2, 2019: 5, 2020: 1, 2021: 3, 2022: 6, 2023: 6, 2024: 4, 2025: 4, 2026: 26.

## Cómo se construyó

1. **Borradores (G001–G038):** se tomaron 60 chunks al azar (semilla 42; 40 de prensa y 20 de Bluesky; uno por documento; mínimo 50 tokens) y Gemini redactó una pregunta por chunk. Llegaron 38; los otros fallaron por saturación del modelo.
2. **Revisión contra el artículo completo (Claude):** se corrigieron fidelidad, contexto (fechas, atribuciones de posts), criterios e idioma, y se marcaron descartes. Se redactaron 15 preguntas de los tipos que faltaban (G039–G053), verificando en los 2.248 chunks que la respuesta existe (o que **no** existe, en las sin respuesta).
3. **Auditoría con un segundo LLM:** se aplicaron solo las observaciones verificadas en el corpus.
4. **Revisión humana final:** 46 aceptadas y 7 descartadas.

La relevancia se juzgó leyendo los textos, no por las palabras clave de recolección del corpus (requisito del taller).

**Descartadas (7):** G003 (cultura general sacada de una nota al pie), G014 (respuesta obvia), G031 y G032 (descartadas en la revisión humana), G036 (cita de opinión difícil de calificar), G042 (comparaba dos ciudades en fechas sin relación) y G045 (ambigua descartada en la revisión humana).

## Limitaciones y hallazgos del corpus (útiles para el informe)

- **G005:** el texto original del corpus tiene tildes dañadas (problema de codificación). La evidencia está escrita con las tildes reparadas, así que no coincide carácter por carácter con el chunk indexado. Es la única advertencia del script de validación.
- Algunos `created_at` son fecha de recolección y no de publicación (por ejemplo, un documento de 2026 con contenido de 2020).
- Hay ruido de las páginas web dentro de los chunks (otras noticias, avisos de chatbot): por ejemplo, G017 y G024.
- Idioma mal etiquetado o nulo en los metadatos de varios posts; aquí se corrigió en `idioma_documento`.
- Las notas automatizadas de Infobae repiten el mismo párrafo: por eso G002 tiene 8 documentos relevantes.
- Varias respuestas vienen de afirmaciones de posts de Bluesky: el gold mide fidelidad al corpus, no si la afirmación es cierta en el mundo.
- G006, G015 y G025 son datos que un LLM puede saber sin consultar el corpus: sirven para medir la recuperación, pero conviene analizarlas aparte en exactitud de generación.

## Validar

```bash
python validar_gold_v1.py <carpeta_con_gold_v1> <ruta/a/vectores.parquet>
```

Resultado esperado: una sola advertencia (G005, por las tildes del corpus).
