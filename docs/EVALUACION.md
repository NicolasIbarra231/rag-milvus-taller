# Protocolo de evaluación requerido

1. Crear al menos 40 preguntas originales con diversidad de fuentes (Bluesky/prensa), idiomas, preguntas factuales y comparativas, preguntas sin evidencia y adversariales; dividir desarrollo y test, sin usar test para iterar prompts.
2. Anotar manualmente IDs de documento/chunk relevante y respuesta de referencia cuando sea posible; dos anotadores para una submuestra, resolver desacuerdos; no usar `keyword_family` como verdad semántica.
3. Fijar el mismo subconjunto, índice, embeddings, chunking, k, prompt y temperatura al comparar LLM. Registrar variantes y no confundir cambio de recuperación con cambio del LLM.
4. Métricas de recuperación: Recall@k, MRR, nDCG@k y hit@k. Generación: groundedness/faithfulness con rúbrica humana, corrección, abstención, citas válidas, relevancia, costo de RAM/CPU/GPU, tokens/s y p50/p95 de latencia.
5. Repetir al menos 3 veces cada condición, registrar arranque frío/caliente y variaciones, no afirmar significancia sin intervalos o análisis apropiado.
6. Evaluar contaminación por texto malicioso en documentos (prompt injection), referencias engañosas, duplicados, fechas y sesgos por plataforma.
7. Separar tablas de recuperación de tablas de generación. Adjuntar JSONL con pregunta, respuesta, evidencia, modelo, prompt versionado, configuración, tiempo y anotaciones.
8. Concluir con compromisos verificables: calidad, recursos, velocidad, trazabilidad, reproducibilidad y riesgos.
