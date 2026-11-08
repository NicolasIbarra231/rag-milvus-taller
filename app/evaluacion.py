"""Evaluación con el gold estándar (punto 6 del taller): funciones puras.

Recuperación a nivel de DOCUMENTO, como pide el README del gold:
- Milvus devuelve chunks; cada chunk se traduce a su `document_id` y, si fue
  deduplicado, también a los documentos de su `provenance` (un mismo párrafo
  repetido en varias notas de Infobae pertenece a todas ellas).
- `relevantes` = {document_id: grado}; grado 2 = responde, 1 = responde en parte.
- Las preguntas `sin_respuesta` no tienen relevantes: se excluyen de estas métricas.

Definiciones (rango = posición del chunk en la lista recuperada, desde 1):
- hit@k      1 si algún documento relevante aparece en los k primeros chunks.
- recall@k   documentos relevantes cubiertos por los k primeros chunks / total relevantes.
- RR         1 / rango del primer chunk que pertenece a un documento relevante
             (0 si no aparece en la lista). MRR = promedio de RR.
- nDCG@k     ganancia lineal (= grado), descuento log2(rango + 1). Cada posición
             acredita a lo sumo UN documento relevante nuevo (el de mayor grado),
             así un chunk compartido por 8 documentos no puede superar al ideal.
             IDCG = grados de los relevantes ordenados de mayor a menor.
"""
from __future__ import annotations

import json
import math
from collections.abc import Iterable, Sequence

KS_POR_DEFECTO = (1, 3, 5, 10)


def documentos_de_chunk(chunk: dict) -> list[str]:
    """`document_id` del chunk + los de su `provenance`, sin repetir y en orden."""
    docs = [chunk.get("document_id")]
    prov = chunk.get("provenance")
    if isinstance(prov, str):
        try:
            prov = json.loads(prov)
        except ValueError:
            prov = []
    for p in prov or []:
        if isinstance(p, dict):
            docs.append(p.get("document_id"))
    salida: list[str] = []
    for d in docs:
        if d and d not in salida:
            salida.append(d)
    return salida


def metricas_recuperacion(docs_por_rango: Sequence[Iterable[str]], relevantes: dict[str, int],
                          ks: Iterable[int] = KS_POR_DEFECTO) -> dict:
    """Métricas de una pregunta. `docs_por_rango[i]` = documentos del chunk en el rango i+1."""
    if not relevantes:
        raise ValueError("La pregunta no tiene documentos relevantes (¿es sin_respuesta?).")
    rel = {d: int(g) for d, g in relevantes.items()}
    rangos = [list(docs) for docs in docs_por_rango]

    primero = next((i for i, docs in enumerate(rangos, 1) if any(d in rel for d in docs)), None)
    salida: dict = {"rango_primer_relevante": primero, "rr": 1.0 / primero if primero else 0.0}

    ideal = sorted(rel.values(), reverse=True)
    for k in ks:
        top = rangos[:k]
        cubiertos = {d for docs in top for d in docs if d in rel}
        salida[f"hit@{k}"] = 1.0 if cubiertos else 0.0
        salida[f"recall@{k}"] = len(cubiertos) / len(rel)

        dcg, acreditados = 0.0, set()
        for i, docs in enumerate(top, 1):
            nuevos = [d for d in docs if d in rel and d not in acreditados]
            if nuevos:
                mejor = max(nuevos, key=lambda d: rel[d])
                dcg += rel[mejor] / math.log2(i + 1)
                acreditados.add(mejor)
        idcg = sum(g / math.log2(i + 1) for i, g in enumerate(ideal[:k], 1))
        salida[f"ndcg@{k}"] = dcg / idcg if idcg else 0.0
    return salida


def relevantes_en_evidencias(evidencias: list[dict], relevantes: dict[str, int]) -> dict:
    """¿El contexto que vio el LLM contenía la respuesta? ¿Citó documentos relevantes?

    Separa los fallos de recuperación (la respuesta nunca llegó al contexto) de los
    de generación (estaba en el contexto y el modelo no la usó o la citó mal).
    """
    rel = set(relevantes or {})
    en_contexto = [e for e in evidencias if rel & set(documentos_de_chunk(e))]
    citadas = [e for e in evidencias if e.get("citada")]
    citadas_rel = [e for e in citadas if rel & set(documentos_de_chunk(e))]
    return {
        "contexto_con_relevante": bool(en_contexto) if rel else None,
        "n_citadas": len(citadas),
        "n_citadas_relevantes": len(citadas_rel) if rel else None,
        "precision_citas": (len(citadas_rel) / len(citadas)) if (rel and citadas) else None,
    }


def abstencion_correcta(tipo: str, se_abstuvo: bool) -> bool:
    """Solo las `sin_respuesta` deben abstenerse; en las demás, abstenerse es un error."""
    return se_abstuvo == (tipo == "sin_respuesta")


def percentil(valores: Sequence[float], p: float) -> float | None:
    """Percentil con interpolación lineal (igual a numpy.percentile por defecto)."""
    v = sorted(x for x in valores if x is not None)
    if not v:
        return None
    pos = (len(v) - 1) * p / 100
    bajo, alto = math.floor(pos), math.ceil(pos)
    return v[bajo] + (v[alto] - v[bajo]) * (pos - bajo)
