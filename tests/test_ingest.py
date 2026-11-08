"""Ingesta: los dos modos producen la misma colección (reproducibilidad).

--from-corpus (muestra -> chunks -> embeddings) guarda vectores.parquet; luego
--from-vectors carga ese archivo en una colección nueva. Ambos deben dar el mismo
conteo y la misma huella de vectores, y reejecutar no debe duplicar.
"""
import json

import pandas as pd

from tests.conftest import codificar_falso, requiere_milvus_lite

pytestmark = requiere_milvus_lite


def _corpus(ruta):
    filas = []
    for i in range(40):
        col = "bluesky_posts" if i % 3 == 0 else "url_crawls"
        texto = " ".join(f"Frase {j} del documento {i} sobre el clima en Colombia." for j in range(1 if col == "bluesky_posts" else 60))
        filas.append({"document_id": f"{col}:{i:03d}", "text": texto, "collection_name": col,
                      "title": None if col == "bluesky_posts" else f"Noticia {i}", "url": f"https://x.org/{i}",
                      "source": "prueba", "language": "es" if i % 2 else "en", "created_at": "2026-10-01"})
    filas.append(dict(filas[1], document_id="url_crawls:999"))  # copia exacta: debe excluirse del universo
    pd.DataFrame(filas).to_parquet(ruta, index=False)


def test_dos_modos_misma_coleccion(entorno, monkeypatch, capsys):
    import app.encoder
    from app import config, milvus_store
    from scripts import ingest

    monkeypatch.setattr(app.encoder, "codificar", codificar_falso)
    corpus, vectores = entorno / "corpus.parquet", entorno / "salida" / "vectores.parquet"
    _corpus(corpus)

    assert ingest.main(["--from-corpus", str(corpus), "--muestra", "12", "--tokenizer", "regex",
                        "--guardar-vectores", str(vectores), "--recrear"]) == 0
    salida = capsys.readouterr().out
    resumen_a = json.loads(salida[salida.index("{\n"):])
    assert resumen_a["documentos"] == 12 and resumen_a["verificacion"]["conteo_ok"]
    assert resumen_a["verificacion"]["busqueda_control_ok"]
    assert (vectores.parent / "encoder_config.json").is_file()

    # Reejecutar el mismo modo no duplica (upsert por id estable)
    assert ingest.main(["--from-corpus", str(corpus), "--muestra", "12", "--tokenizer", "regex"]) == 0
    capsys.readouterr()
    assert milvus_store.contar() == resumen_a["chunks"]

    # Modo 2 en otra colección: mismos vectores, mismo conteo
    monkeypatch.setenv("MILVUS_COLLECTION", "prueba_desde_vectores")
    config.settings.cache_clear()
    assert ingest.main(["--from-vectors", str(vectores)]) == 0
    salida = capsys.readouterr().out
    resumen_b = json.loads(salida[salida.index("{\n"):])
    assert resumen_b["huella_vectores"] == resumen_a["huella_vectores"]
    assert resumen_b["verificacion"]["registros"] == resumen_a["chunks"]


def test_from_vectors_rechaza_otro_contrato(entorno, tmp_path):
    import pytest

    from scripts import ingest

    ruta = tmp_path / "vectores.parquet"
    pd.DataFrame({"chunk_id": ["a"], "document_id": ["d"], "text": ["t"],
                  "embedding": [list(codificar_falso(["t"], "passage")[0])]}).to_parquet(ruta)
    (tmp_path / "encoder_config.json").write_text(json.dumps({"modelo": "BAAI/bge-m3", "dimension": 1024}))
    with pytest.raises(SystemExit, match="no coincide"):
        ingest.main(["--from-vectors", str(ruta)])
