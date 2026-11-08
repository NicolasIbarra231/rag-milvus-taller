"""Ingesta a Milvus standalone, en dos modos (literal G).

Modo 1 — reutilizar los vectores del notebook (rápido, mismos vectores de los experimentos):
    python -m scripts.ingest --from-vectors data/vectores.parquet

Modo 2 — desde cero, desde el Parquet del corpus (reproduce muestra, chunks y embeddings):
    python -m scripts.ingest --from-corpus data/documentos_rag.parquet --muestra 500

Solo verificar conteo, dimensión y una búsqueda de control:
    python -m scripts.ingest --verificar

La carga es idempotente: el `id` de cada chunk es un hash estable (documento +
configuración + offsets) y se usa `upsert`. Reejecutar no duplica registros.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from app import milvus_store
from app.config import contrato, settings

LOTE = 256


def huella_vectores(E: np.ndarray) -> str:
    """Misma huella que el notebook (Fase 3): sha256 de la matriz float32."""
    return hashlib.sha256(np.ascontiguousarray(E, dtype=np.float32).tobytes()).hexdigest()[:16]


def cargar_en_milvus(chunks: pd.DataFrame, E: np.ndarray, recrear: bool) -> dict:
    creada = milvus_store.asegurar_coleccion(recrear=recrear)
    t0 = time.perf_counter()
    filas = chunks.to_dict("records")
    insertados = 0
    for i in range(0, len(filas), LOTE):
        registros = [milvus_store.fila_a_registro(f, v) for f, v in zip(filas[i:i + LOTE], E[i:i + LOTE])]
        insertados += milvus_store.upsert(registros)
        print(f"  upsert {min(i + LOTE, len(filas)):,}/{len(filas):,}", flush=True)
    try:
        milvus_store.cliente().flush(settings().coleccion)
    except Exception:  # noqa: BLE001 - Milvus Lite no siempre expone flush
        pass
    return {"coleccion_creada": creada, "upserts": insertados, "segundos": round(time.perf_counter() - t0, 1)}


def verificar(esperados: int | None = None, muestra: tuple[str, np.ndarray] | None = None) -> dict:
    """Conteo, dimensión y búsqueda de control (un vector debe encontrarse a sí mismo)."""
    info = milvus_store.estado()
    resultado = {**info}
    if esperados is not None:
        resultado["esperados"] = esperados
        resultado["conteo_ok"] = info["registros"] == esperados
    dim = info.get("dimension")
    resultado["dimension_ok"] = dim is None or int(dim) == int(contrato()["dimension"])
    if muestra is not None:
        chunk_id, vector = muestra
        top = milvus_store.buscar(vector, 1)
        resultado["busqueda_control_ok"] = bool(top) and top[0]["chunk_id"] == chunk_id
        resultado["busqueda_control_score"] = round(top[0]["score"], 4) if top else None
    return resultado


def desde_vectores(ruta: Path, recrear: bool) -> dict:
    c = contrato()
    df = pd.read_parquet(ruta)
    faltan = {"chunk_id", "embedding", "text", "document_id"} - set(df.columns)
    if faltan:
        sys.exit(f"vectores.parquet no tiene las columnas {sorted(faltan)}")
    contrato_vecino = ruta.with_name("encoder_config.json")
    if contrato_vecino.is_file():
        otro = json.loads(contrato_vecino.read_text(encoding="utf-8"))
        for clave in ("modelo", "dimension", "prefijo_query", "prefijo_passage", "normalizado"):
            if otro.get(clave) != c.get(clave):
                sys.exit(f"El contrato de los vectores ({clave}={otro.get(clave)!r}) no coincide con el de la API "
                         f"({c.get(clave)!r}). No se mezclan vectores de contratos distintos.")
    E = np.vstack(df["embedding"].map(lambda v: np.asarray(v, dtype=np.float32)).to_list())
    if E.shape[1] != int(c["dimension"]):
        sys.exit(f"Los vectores tienen {E.shape[1]} dimensiones; el contrato dice {c['dimension']}.")
    if not df["chunk_id"].is_unique:
        sys.exit("chunk_id no es único en vectores.parquet.")
    normas = np.linalg.norm(E, axis=1)
    if c.get("normalizado") and not np.allclose(normas, 1, atol=1e-3):
        sys.exit("Los vectores no están normalizados y el contrato lo exige.")
    print(f"vectores.parquet: {len(df):,} chunks · {E.shape[1]} dims · huella {huella_vectores(E)}")
    carga = cargar_en_milvus(df.drop(columns=["embedding"]), E, recrear)
    return {"modo": "from-vectors", "archivo": str(ruta), "huella_vectores": huella_vectores(E), **carga,
            "verificacion": verificar(len(df), (df["chunk_id"].iloc[0], E[0]))}


def desde_corpus(ruta: Path, n: int, ids: Path | None, recrear: bool, estrategia: str, max_tokens: int,
                 overlap: int, semilla: int, tokenizer: str | None, guardar: Path | None) -> dict:
    from app.chunking import crear_tokenizer
    from app.corpus import STRATA, deduplicar, huella_ids, muestra_estratificada, segmentar, universo_de_muestreo
    from app.encoder import codificar

    columnas = ["document_id", "text", "collection_name", "title", "url", "source", "language", "created_at"]
    sha_archivo = hashlib.sha256(ruta.read_bytes()).hexdigest()
    df = pd.read_parquet(ruta, columns=columnas)
    print(f"Corpus: {len(df):,} documentos · sha256 {sha_archivo[:16]}…")
    universo = universo_de_muestreo(df)
    if ids:
        lista = [x for x in ids.read_text(encoding="utf-8").splitlines() if x.strip()]
        muestra = universo[universo["document_id"].isin(set(lista))].sort_values("document_id")
        if len(muestra) != len(lista):
            print(f"⚠️ {len(lista) - len(muestra)} IDs del archivo no están en el universo de muestreo.")
    else:
        muestra = muestra_estratificada(universo, n, STRATA, semilla)
    huella = huella_ids(muestra["document_id"].tolist())
    print(f"Muestra: {len(muestra):,} documentos · huella de IDs {huella} (compárela con la del notebook)")

    nombre_tok = tokenizer or contrato()["modelo"]
    chunks = deduplicar(segmentar(muestra, crear_tokenizer(nombre_tok), estrategia, max_tokens, overlap))
    print(f"Chunks únicos: {len(chunks):,} · codificando con {contrato()['modelo']}…")
    t0 = time.perf_counter()
    E = codificar(chunks["text"].tolist(), "passage")
    t_cod = time.perf_counter() - t0
    print(f"Embeddings: {E.shape} en {t_cod:.1f} s · huella {huella_vectores(E)}")
    if guardar:
        guardar.parent.mkdir(parents=True, exist_ok=True)
        chunks.assign(embedding=list(E)).to_parquet(guardar, index=False)
        guardar.with_name("encoder_config.json").write_text(
            json.dumps({k: v for k, v in contrato().items() if not k.startswith("_")}, indent=2, ensure_ascii=False),
            encoding="utf-8")
        print(f"Vectores guardados en {guardar}")
    carga = cargar_en_milvus(chunks, E, recrear)
    return {"modo": "from-corpus", "archivo": str(ruta), "sha256_corpus": sha_archivo, "documentos": len(muestra),
            "huella_ids": huella, "chunks": len(chunks), "huella_vectores": huella_vectores(E),
            "segundos_embeddings": round(t_cod, 1), "tokenizer": nombre_tok, **carga,
            "verificacion": verificar(len(chunks), (chunks["chunk_id"].iloc[0], E[0]))}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    modo = p.add_mutually_exclusive_group(required=True)
    modo.add_argument("--from-vectors", type=Path, help="vectores.parquet generado por el notebook (Fase 3)")
    modo.add_argument("--from-corpus", type=Path, help="Parquet original del corpus (dragons_rag)")
    modo.add_argument("--verificar", action="store_true", help="Solo verificar la colección existente")
    p.add_argument("--muestra", type=int, default=500, help="Tamaño de la muestra estratificada (from-corpus)")
    p.add_argument("--ids", type=Path, help="Archivo de IDs del notebook (muestras/muestra_500_seed42.txt)")
    p.add_argument("--estrategia", default="struct", choices=["struct", "fixed"])
    p.add_argument("--max-tokens", type=int, default=256)
    p.add_argument("--overlap", type=int, default=32)
    p.add_argument("--semilla", type=int, default=42)
    p.add_argument("--tokenizer", help="Por defecto, el del encoder del contrato; 'regex' para no descargar")
    p.add_argument("--guardar-vectores", type=Path, help="Guarda los vectores generados (from-corpus)")
    p.add_argument("--recrear", action="store_true", help="Borra y recrea la colección (reindexación completa)")
    a = p.parse_args(argv)

    print(f"Milvus: {settings().milvus_uri} · colección: {settings().coleccion}")
    if a.verificar:
        resumen = {"modo": "verificar", "verificacion": verificar()}
    elif a.from_vectors:
        if not a.from_vectors.is_file():
            sys.exit(f"No existe {a.from_vectors}. Copie aquí las salidas del notebook (ver data/README.md).")
        resumen = desde_vectores(a.from_vectors, a.recrear)
    else:
        if not a.from_corpus.is_file():
            sys.exit(f"No existe {a.from_corpus}. Copie el Parquet del corpus en data/ (ver README).")
        resumen = desde_corpus(a.from_corpus, a.muestra, a.ids, a.recrear, a.estrategia, a.max_tokens,
                               a.overlap, a.semilla, a.tokenizer, a.guardar_vectores)
    print(json.dumps(resumen, indent=2, ensure_ascii=False, default=str))
    v = resumen["verificacion"]
    return 0 if v.get("conteo_ok", True) and v.get("busqueda_control_ok", True) else 1


if __name__ == "__main__":
    sys.exit(main())
