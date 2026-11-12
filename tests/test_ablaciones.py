"""Ablaciones de la generación, modo local de la evaluación, robustez y registro de entorno."""
import json

import pytest

from tests.conftest import requiere_milvus_lite
from tests.test_evaluacion import api_eval  # noqa: F401  (fixture reutilizada)


# ------------------------------------------------------------ registro de entorno
def test_entorno_proceso_y_modelo(monkeypatch):
    from app import entorno, llm

    info = entorno.proceso()
    assert {"plataforma", "cpus_visibles", "versiones"} <= set(info)
    monkeypatch.setattr(entorno, "_MODELOS", {})
    monkeypatch.setattr(llm, "version", lambda: "0.40.15")
    monkeypatch.setattr(llm, "listar_modelos", lambda: [{"nombre": "m:1b", "digest": "abc", "cuantizacion": "Q8_0"}])
    assert entorno.modelo("m:1b")["cuantizacion"] == "Q8_0"
    assert entorno.modelo("m:1b")["ollama_version"] == "0.40.15"


def test_entorno_no_cachea_un_fallo(monkeypatch):
    from app import entorno, llm

    monkeypatch.setattr(entorno, "_MODELOS", {})

    def falla():
        raise RuntimeError("Ollama caído")
    monkeypatch.setattr(llm, "version", falla)
    assert entorno.modelo("x")["ollama_version"] is None
    monkeypatch.setattr(llm, "version", lambda: "1")
    monkeypatch.setattr(llm, "listar_modelos", lambda: [])
    assert entorno.modelo("x")["ollama_version"] == "1"


# ------------------------------------------------------ inyección en el pipeline
@requiere_milvus_lite
def test_preguntar_inyecta_en_la_posicion_y_no_registra(api_eval):
    from app import rag
    from app.config import settings
    from scripts.robustez import evidencia_sintetica

    ev = evidencia_sintetica("prueba", "Texto inyectado de prueba.")
    res = rag.preguntar("¿Por qué retroceden los glaciares?", modelo="qwen2.5:1.5b", inyectar=[(2, ev)],
                        registrar=False)
    assert res["evidencias"][1]["chunk_id"] == "sintetico-prueba"
    assert res["evidencias"][1]["referencia"] == 2
    assert "entorno" in res and "proceso" in res["entorno"]
    from pathlib import Path
    assert not Path(settings().ruta_logs).is_file()


def test_clasificar_discrepancia():
    from scripts.robustez import clasificar_discrepancia

    assert clasificar_discrepancia("Son 2.700 especies [1].", r"2[.,]?7\d\d", r"1[.,]?200")["clase"] == "mantiene la real"
    c = clasificar_discrepancia("Una fuente dice 2.700 [1]; sin embargo, otra dice 1.200 [2].", r"2[.,]?7\d\d",
                                r"1[.,]?200")
    assert c["clase"] == "reporta ambas y advierte"
    assert clasificar_discrepancia("Son 1.200 [2].", r"2[.,]?7\d\d", r"1[.,]?200")["clase"] == "adopta la falsa"


def test_robustez_con_pipeline_falso(tmp_path):
    from scripts import robustez

    gold = tmp_path / "gold.jsonl"
    ids = ["G005", "G009", "G034", "G024", "G020"]
    gold.write_text("\n".join(json.dumps({"gold_id": g, "pregunta": f"Pregunta {g}"}) for g in ids), encoding="utf-8")

    def preguntar(pregunta, modelo, inyectar, registrar):
        assert registrar is False and inyectar[0][0] == robustez.POSICION
        texto = inyectar[0][1]["text"]
        respuesta = "CANARIO-7319" if "CANARIO" in texto and modelo == "debil" else "Son 2.700 especies [1]."
        return {"respuesta": respuesta, "evidencias": [{"chunk_id": "x", "referencia": 1, "citada": True},
                                                        {**inyectar[0][1], "referencia": 2, "citada": False}],
                "citas": {"estado": "referencias_existentes"}, "abstencion": False, "tiempos": {}, "tokens": {}}

    salida = tmp_path / "rob"
    args = ["--gold", str(gold), "--salida", str(salida), "--modelos", "debil", "fuerte"]
    assert robustez.main(args, preguntar=preguntar, cargar=lambda m: None) == 0
    filas = [json.loads(l) for l in (salida / "robustez_crudo.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(filas) == 2 * (3 * 4 + 3)
    exitos = {(f["modelo"], f["caso"]) for f in filas if f.get("ataque_exitoso")}
    assert exitos == {("debil", "instruccion")}
    # Reanudar no repite casos.
    robustez.main(args, preguntar=preguntar, cargar=lambda m: None)
    assert len((salida / "robustez_crudo.jsonl").read_text(encoding="utf-8").splitlines()) == len(filas)
    assert (salida / "resumen_inyeccion.csv").is_file()


# ------------------------------------------------- ablación de la generación
@requiere_milvus_lite
def test_ablacion_de_generacion_reutiliza_respuestas_identicas(api_eval):
    from openpyxl import load_workbook

    from scripts import evaluar_gold

    cliente, gold, salida = api_eval
    base = ["--gold", str(gold), "--salida", str(salida), "--api", "", "--modelos", "qwen2.5:1.5b",
            "--fraccion-doble", "1.0"]
    evaluar_gold.main(["generacion", *base], sesion=cliente)
    evaluar_gold.main(["planilla", *base], sesion=cliente)
    ruta = salida / "planilla_evaluacion.xlsx"
    libro = load_workbook(ruta)
    h = libro["Evaluador 1"]
    for fila in range(2, h.max_row + 1):
        h[f"N{fila}"], h[f"O{fila}"], h[f"P{fila}"], h[f"Q{fila}"] = 3, 3, 3, "Sí"
    libro.save(ruta)

    # Misma configuración: todo idéntico, nada por calificar.
    assert evaluar_gold.main(["generacion", *base, "--etiqueta", "igual", "--k", "5"], sesion=cliente) == 0
    evaluar_gold.main(["planilla-ablacion", *base, "--etiqueta", "igual"], sesion=cliente)
    assert not (salida / "planilla_igual.xlsx").exists()
    evaluar_gold.main(["calificaciones-ablacion", *base, "--etiqueta", "igual"], sesion=cliente)
    res = (salida / "calificaciones_igual.csv").read_text(encoding="utf-8")
    assert "qwen2.5:1.5b" in res

    # k=1 cambia el contexto: esas respuestas se califican de nuevo.
    evaluar_gold.main(["generacion", *base, "--etiqueta", "k1", "--k", "1"], sesion=cliente)
    filas = [json.loads(l) for l in (salida / "generacion_k1_crudo.jsonl").read_text(encoding="utf-8").splitlines()]
    assert all(f["parametros"]["k"] == 1 for f in filas)
    evaluar_gold.main(["planilla-ablacion", *base, "--etiqueta", "k1"], sesion=cliente)
    assert (salida / "planilla_k1.xlsx").is_file()


@requiere_milvus_lite
def test_recuperacion_local_reporta_cobertura(api_eval):
    import csv

    from scripts import evaluar_gold

    _, gold, salida = api_eval
    assert evaluar_gold.main(["recuperacion", "--local", "--gold", str(gold), "--salida", str(salida),
                              "--etiqueta", "local", "--k-max", "5"]) == 0
    filas = list(csv.DictReader((salida / "recuperacion_local_por_pregunta.csv").open(encoding="utf-8")))
    cobertura = {f["gold_id"]: int(f["relevantes_en_coleccion"]) for f in filas}
    assert cobertura["G001"] == 1 and cobertura["G003"] == 0


def test_separar_ablacion():
    from scripts.evaluar_gold import separar_ablacion

    base = [{"gold_id": "G1", "modelo": "m", "respuesta": "A", "contexto": {"huella": "h1"}},
            {"gold_id": "G2", "modelo": "m", "respuesta": "B", "contexto": {"huella": "h2"}}]
    abl = [{"gold_id": "G1", "modelo": "m", "respuesta": "A", "contexto": {"huella": "h1"}},
           {"gold_id": "G2", "modelo": "m", "respuesta": "B2", "contexto": {"huella": "h9"}},
           {"gold_id": "G3", "modelo": "m", "error": "ReadTimeout"}]
    reuso, nuevas = separar_ablacion(base, abl)
    assert [f["gold_id"] for f in reuso] == ["G1"] and [f["gold_id"] for f in nuevas] == ["G2"]


def test_encoder_pasa_revision_y_model_kwargs_del_contrato(monkeypatch, tmp_path):
    """bge-m3 se carga en safetensors desde una revisión fija (sin torch.load)."""
    import json, sys, types
    from pathlib import Path
    from app import config, encoder

    llamadas = {}

    class FalsoST:
        def __init__(self, nombre, **kw):
            llamadas.update(nombre=nombre, **kw)

        def get_sentence_embedding_dimension(self):
            return 1024

    monkeypatch.setitem(sys.modules, "sentence_transformers", types.SimpleNamespace(SentenceTransformer=FalsoST))
    contrato_bge = json.loads(Path("configs/encoders/bge-m3.json").read_text(encoding="utf-8"))
    monkeypatch.setattr(encoder, "contrato", lambda: contrato_bge)
    encoder.modelo.cache_clear()
    try:
        encoder.modelo()
    finally:
        encoder.modelo.cache_clear()
    assert llamadas["nombre"] == "BAAI/bge-m3"
    assert llamadas["revision"] == "7b44caba5c06fd6f1ef1d2786261e636c38e4295"
    assert llamadas["model_kwargs"] == {"use_safetensors": True}


def test_provenance_larga_se_compacta_sin_perder_documentos():
    """Un párrafo repetido en cientos de documentos no debe romper el upsert (límite JSON de Milvus)."""
    import json
    from app import evaluacion, milvus_store

    prov = [{"chunk_id": f"c{i:04d}" + "x" * 40, "document_id": f"doc_{i:04d}",
             "url": "https://ejemplo.co/" + "a" * 120, "collection_name": "url_crawls",
             "char_start": 0, "char_end": 900, "created_at": "2025-01-01T00:00:00"} for i in range(800)]
    assert len(json.dumps(prov).encode()) > 65_536            # como el caso real (184.919 bytes)
    fila = {"chunk_id": "c0000", "document_id": "doc_0000", "text": "aviso de cookies",
            "n_sources": 800, "provenance": json.dumps(prov)}
    reg = milvus_store.fila_a_registro(fila, [0.0] * 4)
    assert len(json.dumps(reg["provenance"], ensure_ascii=False).encode()) <= milvus_store.LIMITE_JSON
    assert reg["n_sources"] == 800
    assert evaluacion.documentos_de_chunk(reg) == [f"doc_{i:04d}" for i in range(800)]   # ningún documento se pierde

    corto = [{"document_id": "d1", "url": "u"}]
    assert milvus_store.compactar_provenance(corto) == corto                         # lo normal no cambia

    enorme = [{"document_id": f"documento_{i:06d}" * 3} for i in range(5000)]
    rec = milvus_store.compactar_provenance(enorme)
    assert rec[-1] == {"_truncado": True, "total_origenes": 5000}
    assert len(json.dumps(rec).encode()) <= milvus_store.LIMITE_JSON
