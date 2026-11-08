"""Fixtures: Milvus Lite en un archivo temporal + encoder falso determinista.

Las pruebas no descargan modelos ni requieren Docker: así corren en CI y dentro
del contenedor (`docker compose exec api pytest -q`). El encoder falso es una
bolsa de palabras con hashing (384 dims, normalizada), suficiente para que
"cambio climático" se parezca más a un texto sobre clima que a uno sobre fútbol.
"""
from __future__ import annotations

import hashlib
import importlib.util
import re

import numpy as np
import pytest

# pymilvus lee MILVUS_URI al importarse y exige forma http://; las pruebas usan
# un archivo .db de Milvus Lite, así que se importa antes de cambiar la variable.
import pymilvus  # noqa: F401,E402

DIM = 384


def codificar_falso(textos, tipo, prefijo=True, batch=64):
    salida = np.zeros((len(textos), DIM), dtype=np.float32)
    for i, t in enumerate(textos):
        for palabra in re.findall(r"\w+", t.lower()):
            h = int(hashlib.md5(palabra.encode()).hexdigest(), 16)
            salida[i, h % DIM] += 1.0
        n = np.linalg.norm(salida[i])
        salida[i] = salida[i] / n if n else salida[i]
    return salida


def _limpiar_caches():
    from app import config, milvus_store

    config.settings.cache_clear()
    config.contrato.cache_clear()
    milvus_store.cliente.cache_clear()


@pytest.fixture
def entorno(tmp_path, monkeypatch):
    """Variables de entorno aisladas por prueba."""
    monkeypatch.setenv("MILVUS_URI", str(tmp_path / "milvus_test.db"))
    monkeypatch.setenv("MILVUS_COLLECTION", "prueba_rag")
    monkeypatch.setenv("ENCODER_CONFIG", str(tmp_path / "no_existe.json"))
    monkeypatch.setenv("RUN_LOG_PATH", str(tmp_path / "logs" / "runs.jsonl"))
    monkeypatch.delenv("EMBEDDING_MODEL", raising=False)
    _limpiar_caches()
    import app.retrieval

    monkeypatch.setattr(app.retrieval, "codificar", codificar_falso)
    yield tmp_path
    _limpiar_caches()


requiere_milvus_lite = pytest.mark.skipif(
    importlib.util.find_spec("milvus_lite") is None,
    reason="milvus-lite no instalado (pip install -r requirements-dev.txt)",
)

DOCUMENTOS = [
    ("url_crawls:a", "url_crawls", "es", "Glaciares en retroceso",
     "El cambio climático acelera el retroceso de los glaciares patagónicos. La temperatura subió 0,76 grados."),
    ("url_crawls:a", "url_crawls", "es", "Glaciares en retroceso",
     "Los glaciares patagónicos pierden masa cada año por el cambio climático y la sequía."),
    ("url_crawls:b", "url_crawls", "es-co", "Sequía en Colombia",
     "La sequía y el cambio climático afectan los embalses de Colombia según el Ideam."),
    ("bluesky_posts:c", "bluesky_posts", "en", None,
     "Climate change is making droughts longer in South America."),
    ("url_crawls:d", "url_crawls", "es", "Fútbol",
     "El equipo ganó el partido de fútbol con dos goles en el estadio."),
]


@pytest.fixture
def coleccion_cargada(entorno):
    from app import milvus_store
    from app.chunking import hash_contenido

    milvus_store.asegurar_coleccion(recrear=True)
    textos = [d[4] for d in DOCUMENTOS]
    E = codificar_falso(textos, "passage")
    registros = []
    for i, ((doc, col, lang, titulo, texto), v) in enumerate(zip(DOCUMENTOS, E)):
        fila = {"chunk_id": f"chunk{i:02d}", "document_id": doc, "text": texto, "title": titulo,
                "url": f"https://ejemplo.org/{i}", "collection_name": col, "source": "prueba", "language": lang,
                "created_at": "2026-10-01", "chunk_index": i, "char_start": 0, "char_end": len(texto),
                "n_tokens": len(texto.split()), "content_hash": hash_contenido(texto),
                "estrategia": "struct256o32", "provenance": "[]"}
        registros.append(milvus_store.fila_a_registro(fila, v))
    milvus_store.upsert(registros)
    return registros
