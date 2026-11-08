"""Paso 1 del RAG: recuperar. Separado de la generación (literal E).

Dos estrategias, iguales a las del notebook:
- convencional: los k chunks más similares.
- diversificada: se piden `candidatos` (30) y se conserva el primer chunk de cada
  documento hasta llegar a k. En la ablación subió de 3 a 5 los documentos
  únicos del contexto con un costo de latencia despreciable.
"""
from __future__ import annotations

import time

from app import milvus_store
from app.config import settings
from app.encoder import codificar

ESTRATEGIAS = ("diversificada", "convencional")


def diversificar(resultados: list[dict], k: int) -> list[dict]:
    """Primer chunk (el de mayor score) de cada documento, hasta k."""
    seleccion, vistos = [], set()
    for r in resultados:
        doc = r.get("document_id")
        if doc in vistos:
            continue
        vistos.add(doc)
        seleccion.append(r)
        if len(seleccion) >= k:
            break
    return seleccion


def recuperar(pregunta: str, k: int, estrategia: str | None = None, candidatos: int | None = None,
              filtro: str = "") -> tuple[list[dict], dict]:
    """Devuelve (evidencias, métricas) con tiempos de codificación y búsqueda."""
    s = settings()
    estrategia = estrategia or s.estrategia_por_defecto
    if estrategia not in ESTRATEGIAS:
        raise ValueError(f"estrategia debe ser una de {ESTRATEGIAS}")
    candidatos = max(candidatos or s.candidatos, k)

    t0 = time.perf_counter()
    vector = codificar([pregunta], "query")[0]
    t1 = time.perf_counter()
    limite = candidatos if estrategia == "diversificada" else k
    crudos = milvus_store.buscar(vector, limite, filtro)
    t2 = time.perf_counter()

    evidencias = diversificar(crudos, k) if estrategia == "diversificada" else crudos[:k]
    metricas = {
        "estrategia": estrategia, "k": k, "candidatos": limite if estrategia == "diversificada" else None,
        "filtro": filtro or None, "recuperados": len(evidencias),
        "documentos_unicos": len({e.get("document_id") for e in evidencias}),
        "tiempo_codificacion_s": round(t1 - t0, 4), "tiempo_busqueda_s": round(t2 - t1, 4),
        "tiempo_recuperacion_s": round(t2 - t0, 4),
    }
    return evidencias, metricas
