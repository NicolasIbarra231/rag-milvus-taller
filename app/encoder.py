"""Encoder único del proyecto: la misma función `codificar()` del notebook (Fase 3).

Decisión de diseño (literal C/E): el encoder vive *dentro de la API* para las
preguntas y dentro del *trabajo de ingesta* para los chunks; ambos importan este
módulo y leen el mismo contrato (encoder_config.json). Así es imposible codificar
una pregunta con un prefijo, modelo o normalización distintos a los documentos.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np

from app.config import contrato, settings


class ErrorContrato(RuntimeError):
    """El encoder cargado no coincide con el contrato (dimensión, modelo)."""


@lru_cache(maxsize=1)
def modelo():
    from sentence_transformers import SentenceTransformer  # import tardío: pesado

    c = contrato()
    m = SentenceTransformer(c["modelo"], device=settings().dispositivo_encoder, revision=c.get("revision"))
    dim = m.get_sentence_embedding_dimension()
    if dim != int(c["dimension"]):
        raise ErrorContrato(
            f"El encoder {c['modelo']} produce {dim} dimensiones y el contrato dice {c['dimension']}. "
            "Si cambió de encoder, cree otra colección y regenere los vectores."
        )
    return m


def codificar(textos: list[str], tipo: str, prefijo: bool = True, batch: int = 64) -> np.ndarray:
    """Convierte textos en vectores con el contrato del proyecto.

    tipo: "query" para preguntas, "passage" para chunks.
    """
    if tipo not in ("query", "passage"):
        raise ValueError("tipo debe ser 'query' o 'passage'")
    c = contrato()
    p = (c["prefijo_query"] if tipo == "query" else c["prefijo_passage"]) if prefijo else ""
    vectores = modelo().encode(
        [p + t for t in textos],
        batch_size=batch,
        normalize_embeddings=bool(c["normalizado"]),
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    return vectores.astype(np.float32)
