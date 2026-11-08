"""Orquestación RAG: recuperar -> ensamblar contexto -> generar -> validar -> registrar.

Equivale a `preguntar_rag_v6` + `comparar_llm` del notebook, con dos añadidos que
pide el literal F/G: tiempos desglosados y un registro JSONL por ejecución.
"""
from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from app import llm, prompts
from app.config import contrato, settings
from app.retrieval import recuperar

log = logging.getLogger("rag")
_candado = threading.Lock()


def registrar_ejecucion(registro: dict) -> None:
    """Agrega una línea a reports/logs/rag_runs.jsonl (resultados crudos, literal 6)."""
    ruta = Path(settings().ruta_logs)
    ruta.parent.mkdir(parents=True, exist_ok=True)
    linea = json.dumps(registro, ensure_ascii=False, default=str)
    with _candado, ruta.open("a", encoding="utf-8") as f:
        f.write(linea + "\n")


def _evidencia_publica(fuente: dict, citadas: set[int]) -> dict:
    claves = ["referencia", "chunk_id", "document_id", "score", "title", "url", "collection_name",
              "source", "language", "created_at", "chunk_index", "char_start", "char_end",
              "n_tokens", "n_sources", "provenance", "text"]
    ev = {c: fuente.get(c) for c in claves}
    ev["citada"] = fuente.get("referencia") in citadas
    return ev


def preguntar(pregunta: str, modelo: str | None = None, k: int | None = None,
              estrategia: str | None = None, prompt_version: str | None = None,
              filtro: str = "", temperatura: float | None = None,
              num_predict: int | None = None) -> dict:
    s = settings()
    modelo = modelo or s.modelo_por_defecto
    k = k or s.k_por_defecto
    prompt_version = prompt_version or s.prompt_por_defecto
    plantilla = prompts.obtener(prompt_version)
    opciones = {
        "temperature": s.temperatura if temperatura is None else temperatura,
        "seed": s.semilla, "num_ctx": s.num_ctx,
        "num_predict": num_predict or s.num_predict,
    }
    run_id = uuid.uuid4().hex[:12]
    inicio = time.perf_counter()

    # 1. Recuperar
    evidencias, met_rec = recuperar(pregunta, k=k, estrategia=estrategia, filtro=filtro)

    # 2. Ensamblar contexto con presupuesto
    contexto, fuentes, truncado = prompts.construir_contexto(evidencias, s.max_contexto_chars)
    advertencias = []
    if truncado:
        advertencias.append("El contexto se recortó por el presupuesto de caracteres (MAX_CONTEXT_CHARS).")

    # 3. Generar (o abstenerse sin llamar al LLM si no hay evidencia)
    datos, t_gen = {}, 0.0
    if not fuentes:
        respuesta = prompts.RESPUESTA_SIN_EVIDENCIA
        advertencias.append("No se recuperaron fragmentos; se respondió sin llamar al LLM.")
    else:
        mensaje_usuario = plantilla["user"].format(pregunta=pregunta, contexto=contexto)
        t0 = time.perf_counter()
        datos = llm.chat(modelo, plantilla["system"], mensaje_usuario, opciones)
        t_gen = time.perf_counter() - t0
        respuesta = (datos.get("message") or {}).get("content", "").strip()

    # 4. Validar citas
    citas = prompts.evaluar_citas(respuesta, fuentes)
    tokens_salida = datos.get("eval_count")
    eval_s = (datos.get("eval_duration") or 0) / 1e9
    carga_s = (datos.get("load_duration") or 0) / 1e9
    if tokens_salida and tokens_salida >= opciones["num_predict"]:
        advertencias.append(f"La respuesta alcanzó el límite de {opciones['num_predict']} tokens y puede estar truncada.")
    if carga_s > 1:
        advertencias.append(f"Arranque en frío: cargar el modelo tomó {carga_s:.1f} s (no es latencia de generación).")
    if citas["estado"] == "referencias_inexistentes":
        advertencias.append(f"La respuesta cita fuentes que no existen: {citas['citas_inexistentes']}.")
    elif citas["estado"] == "respuesta_sin_citas":
        advertencias.append("La respuesta no incluye citas [n]; no es verificable.")

    total = time.perf_counter() - inicio
    resultado = {
        "run_id": run_id,
        "fecha_utc": datetime.now(timezone.utc).isoformat(),
        "pregunta": pregunta,
        "modelo": modelo,
        "respuesta": respuesta,
        "abstencion": citas["estado"] == "abstencion",
        "citas": citas,
        "evidencias": [_evidencia_publica(f, set(citas["citas"])) for f in fuentes],
        "parametros": {
            "prompt_version": prompt_version, "prompt_huella": prompts.huella_prompt(prompt_version),
            "estrategia": met_rec["estrategia"], "k": k, "candidatos": met_rec["candidatos"],
            "filtro": met_rec["filtro"], "max_contexto_chars": s.max_contexto_chars, **opciones,
        },
        "tiempos": {
            "codificacion_s": met_rec["tiempo_codificacion_s"],
            "busqueda_s": met_rec["tiempo_busqueda_s"],
            "recuperacion_s": met_rec["tiempo_recuperacion_s"],
            "generacion_s": round(t_gen, 3),
            "carga_modelo_s": round(carga_s, 3),
            "total_s": round(total, 3),
        },
        "tokens": {
            "entrada": datos.get("prompt_eval_count"),
            "salida": tokens_salida,
            "por_segundo": round(tokens_salida / eval_s, 2) if tokens_salida and eval_s else None,
        },
        "contexto": {
            "caracteres": len(contexto), "truncado": truncado,
            "documentos_unicos": met_rec["documentos_unicos"],
            "huella": hashlib.sha256(contexto.encode("utf-8")).hexdigest()[:16],
        },
        "encoder": contrato()["modelo"],
        "coleccion": s.coleccion,
        "advertencias": advertencias,
    }

    # 5. Registrar (sin el texto de las evidencias: se recupera por chunk_id)
    registro = {k_: v for k_, v in resultado.items() if k_ != "evidencias"}
    registro["evidencias"] = [
        {"referencia": e["referencia"], "chunk_id": e["chunk_id"], "document_id": e["document_id"],
         "score": e["score"], "citada": e["citada"]} for e in resultado["evidencias"]
    ]
    registrar_ejecucion(registro)
    log.info("ask run=%s modelo=%s total=%.2fs citas=%s", run_id, modelo, total, citas["estado"])
    return resultado
