"""Integración con Milvus (Lite): esquema, upsert idempotente, búsqueda y filtros."""
import pytest

from tests.conftest import codificar_falso, requiere_milvus_lite

pytestmark = requiere_milvus_lite


def test_conteo_e_idempotencia(coleccion_cargada):
    from app import milvus_store

    assert milvus_store.contar() == 5
    milvus_store.upsert(coleccion_cargada)  # reindexar no duplica
    assert milvus_store.contar() == 5
    assert milvus_store.estado()["dimension"] in (384, "384")


def test_busqueda_y_metadatos(coleccion_cargada):
    from app import milvus_store

    q = codificar_falso(["cambio climático glaciares"], "query")[0]
    top = milvus_store.buscar(q, 3)
    assert top[0]["document_id"] == "url_crawls:a"
    assert {"url", "title", "char_start", "char_end", "provenance"} <= set(top[0])


def test_filtros(coleccion_cargada):
    from app import milvus_store

    q = codificar_falso(["cambio climático sequía"], "query")[0]
    solo_bluesky = milvus_store.buscar(q, 5, milvus_store.construir_filtro(["bluesky_posts"]))
    assert {r["collection_name"] for r in solo_bluesky} == {"bluesky_posts"}
    espanol = milvus_store.buscar(q, 5, milvus_store.construir_filtro(idioma="es"))
    assert all(r["language"].startswith("es") for r in espanol)


def test_filtros_rechazan_inyeccion():
    from app import milvus_store

    with pytest.raises(ValueError):
        milvus_store.construir_filtro(['x" or 1==1 or "'])
    with pytest.raises(ValueError):
        milvus_store.construir_filtro(idioma='es" or ')


def test_recuperacion_diversificada(coleccion_cargada):
    from app.retrieval import recuperar

    conv, _ = recuperar("cambio climático glaciares patagónicos", k=2, estrategia="convencional")
    div, met = recuperar("cambio climático glaciares patagónicos", k=2, estrategia="diversificada")
    assert [r["document_id"] for r in conv] == ["url_crawls:a", "url_crawls:a"]
    assert len({r["document_id"] for r in div}) == 2 and met["documentos_unicos"] == 2
