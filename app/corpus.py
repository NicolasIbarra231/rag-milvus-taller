"""Muestra reproducible y segmentación del corpus (portado de la Fase 2).

Se usa en el modo `--from-corpus` de la ingesta: permite reproducir desde el
Parquet original exactamente la misma muestra (semilla 42, estratos
collection_name + language) y los mismos chunks que generó el notebook.
"""
from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd

from app.chunking import ESTRATEGIAS, Tokens, hash_contenido, id_estable, nombre_config

STRATA = ["collection_name", "language"]
METADATOS = ["collection_name", "source", "language", "created_at", "url", "title"]


def sha256(texto):
    """Huella SHA-256 de un texto en UTF-8, sin normalizar. None si es nulo."""
    return None if pd.isna(texto) else hashlib.sha256(str(texto).encode("utf-8")).hexdigest()


def universo_de_muestreo(df: pd.DataFrame) -> pd.DataFrame:
    """Con texto, sin copias exactas (queda la de menor document_id), en orden fijo."""
    caracteres = df["text"].fillna("").str.len()
    universo = (
        df.assign(_sha=df["text"].map(sha256))[caracteres > 0]
        .assign(document_id=lambda d: d["document_id"].astype(str))
        .sort_values("document_id")
        .drop_duplicates("_sha")
    )
    for c in STRATA:
        universo[c] = universo[c].fillna("<nulo>")
    return universo


def muestra_estratificada(datos, n, estratos, semilla):
    """Muestra proporcional por estratos (método de los restos mayores)."""
    n = min(n, len(datos))
    estrato = datos[estratos].astype(str).agg(" | ".join, axis=1)
    tamanos = estrato.value_counts().sort_index()
    exacto = tamanos / tamanos.sum() * n
    cuota = np.floor(exacto).astype(int)
    faltan = int(n - cuota.sum())
    cuota.loc[(exacto - cuota).sort_values(ascending=False, kind="stable").index[:faltan]] += 1
    partes = [
        grupo.sample(cuota[clave], random_state=semilla)
        for clave, grupo in datos.groupby(estrato)
        if cuota[clave] > 0
    ]
    return pd.concat(partes).sort_values("document_id")


def huella_ids(document_ids) -> str:
    """Misma huella que el notebook: sha256 del archivo de IDs (uno por línea)."""
    contenido = ("\n".join(document_ids) + "\n").encode("utf-8")
    return hashlib.sha256(contenido).hexdigest()[:16]


def segmentar(muestra: pd.DataFrame, tokenizar, estrategia="struct", max_tokens=256, overlap=32) -> pd.DataFrame:
    """Chunks con offsets verificados, ID estable y metadatos heredados."""
    cid = nombre_config(estrategia, max_tokens, overlap)
    fn = ESTRATEGIAS[estrategia]
    filas = []
    for d in muestra.itertuples(index=False):
        tok = Tokens(d.text, tokenizar)
        meta = {c: getattr(d, c, None) for c in METADATOS}
        spans = fn(d.text, tok, max_tokens, overlap)
        for i, (s, e) in enumerate(spans):
            texto = d.text[s:e]
            assert d.text[s:e] == texto
            filas.append({
                "chunk_id": id_estable(d.document_id, cid, s, e), "estrategia": cid,
                "document_id": d.document_id, "chunk_index": i,
                "char_start": s, "char_end": e, "n_tokens": tok.contar(s, e),
                "text": texto, "content_hash": hash_contenido(texto), **meta,
            })
    return pd.DataFrame(filas)


def deduplicar(chunks: pd.DataFrame) -> pd.DataFrame:
    """Una fila por contenido; `provenance` (JSON) lista todos sus orígenes."""
    chunks = chunks.sort_values(["document_id", "chunk_index"])
    origen: dict[str, list] = {}
    for r in chunks.itertuples():
        origen.setdefault(r.content_hash, []).append({
            "chunk_id": r.chunk_id, "document_id": r.document_id, "url": r.url,
            "collection_name": r.collection_name, "char_start": int(r.char_start),
            "char_end": int(r.char_end), "created_at": str(r.created_at),
        })
    unicos = chunks.drop_duplicates("content_hash").copy()
    unicos["n_sources"] = unicos["content_hash"].map(lambda h: len(origen[h]))
    unicos["provenance"] = unicos["content_hash"].map(lambda h: json.dumps(origen[h], ensure_ascii=False))
    return unicos.reset_index(drop=True)
