"""Evaluación con el gold: métricas a mano, contra sklearn, y el flujo completo contra la API."""
import json
import math

import pytest

from app.evaluacion import (abstencion_correcta, documentos_de_chunk, metricas_recuperacion, percentil,
                            relevantes_en_evidencias)
from tests.conftest import requiere_milvus_lite


# ----------------------------------------------------------------- métricas puras
def test_documentos_de_chunk_incluye_provenance():
    chunk = {"document_id": "a", "provenance": json.dumps([{"document_id": "a"}, {"document_id": "b"}])}
    assert documentos_de_chunk(chunk) == ["a", "b"]
    assert documentos_de_chunk({"document_id": "a", "provenance": [{"document_id": "c"}]}) == ["a", "c"]
    assert documentos_de_chunk({"document_id": "a", "provenance": "no-es-json"}) == ["a"]


def test_metricas_a_mano():
    # Relevantes: x (grado 2) e y (grado 1). Ranking: z, x, w, y.
    m = metricas_recuperacion([["z"], ["x"], ["w"], ["y"]], {"x": 2, "y": 1}, ks=(1, 3, 5))
    assert m["rango_primer_relevante"] == 2 and m["rr"] == 0.5
    assert m["hit@1"] == 0 and m["hit@3"] == 1
    assert m["recall@3"] == 0.5 and m["recall@5"] == 1.0
    dcg = 2 / math.log2(3) + 1 / math.log2(5)
    idcg = 2 / math.log2(2) + 1 / math.log2(3)
    assert m["ndcg@5"] == pytest.approx(dcg / idcg)


def test_chunk_compartido_no_supera_el_ideal():
    # Un chunk deduplicado pertenece a los 3 relevantes: recall completo, pero nDCG <= 1.
    m = metricas_recuperacion([["a", "b", "c"]], {"a": 2, "b": 2, "c": 2}, ks=(1, 5))
    assert m["recall@1"] == 1.0
    assert m["ndcg@1"] == pytest.approx(1.0) and m["ndcg@5"] < 1.0


def test_sin_relevantes_en_lista():
    m = metricas_recuperacion([["z"], ["w"]], {"x": 2}, ks=(1, 5))
    assert m["rr"] == 0 and m["rango_primer_relevante"] is None and m["ndcg@5"] == 0


def test_sin_respuesta_no_tiene_metricas():
    with pytest.raises(ValueError):
        metricas_recuperacion([["z"]], {})


def test_ndcg_coincide_con_sklearn():
    sklearn = pytest.importorskip("sklearn.metrics")
    import numpy as np

    rel = {"d1": 2, "d2": 1, "d3": 2}
    ranking = ["d4", "d3", "d5", "d1", "d2", "d6"]
    m = metricas_recuperacion([[d] for d in ranking], rel, ks=(3, 6))
    verdad = np.array([[rel.get(d, 0) for d in ranking]])
    puntajes = np.array([[len(ranking) - i for i in range(len(ranking))]])
    for k in (3, 6):
        assert m[f"ndcg@{k}"] == pytest.approx(sklearn.ndcg_score(verdad, puntajes, k=k))


def test_relevantes_en_evidencias_y_abstencion():
    ev = [{"document_id": "x", "citada": True}, {"document_id": "z", "citada": True},
          {"document_id": "w", "citada": False}]
    r = relevantes_en_evidencias(ev, {"x": 2})
    assert r["contexto_con_relevante"] is True and r["precision_citas"] == 0.5
    assert relevantes_en_evidencias(ev, {})["contexto_con_relevante"] is None
    assert abstencion_correcta("sin_respuesta", True) and not abstencion_correcta("directa", True)
    assert percentil([1, 2, 3, 4], 50) == 2.5 and percentil([], 50) is None


def test_kappa_coincide_con_sklearn():
    sklearn = pytest.importorskip("sklearn.metrics")
    from scripts.evaluar_gold import kappa

    a = [2, 2, 1, 0, 2, 1, 0, 2, 1, 1]
    b = [2, 1, 1, 0, 2, 2, 0, 2, 0, 1]
    assert kappa(a, b) == pytest.approx(sklearn.cohen_kappa_score(a, b))
    assert kappa(a, b, ponderado=True) == pytest.approx(sklearn.cohen_kappa_score(a, b, weights="quadratic"))
    assert kappa([1, 1], [1, 1]) == 1.0
    # Con la escala completa fija, un desacuerdo 0 vs 3 pesa más que con solo las categorías observadas.
    assert kappa([0, 3, 3, 0], [3, 3, 0, 0], ponderado=True, categorias=[0, 1, 2, 3]) == pytest.approx(
        sklearn.cohen_kappa_score([0, 3, 3, 0], [3, 3, 0, 0], weights="quadratic", labels=[0, 1, 2, 3]))


# ----------------------------------------------- flujo completo contra la API real
GOLD = [
    {"gold_id": "G001", "split": "test", "tipo": "directa", "idioma_pregunta": "es",
     "pregunta": "¿Por qué retroceden los glaciares patagónicos?",
     "respuesta_ref": "Por el cambio climático.", "criterio": "Debe mencionar el cambio climático.",
     "relevantes": {"url_crawls:a": 2}, "rol_documentos": "responde", "documentos": []},
    {"gold_id": "G002", "split": "test", "tipo": "comparativa", "idioma_pregunta": "es",
     "pregunta": "¿Qué efectos de la sequía y el cambio climático se mencionan en Colombia y Sudamérica?",
     "respuesta_ref": "Embalses en Colombia; sequías más largas en Sudamérica.", "criterio": "Usa ambas fuentes.",
     "relevantes": {"url_crawls:b": 2, "bluesky_posts:c": 2}, "rol_documentos": "responde", "documentos": []},
    {"gold_id": "G003", "split": "dev", "tipo": "sin_respuesta", "idioma_pregunta": "es",
     "pregunta": "¿Cuál fue el número ganador de la lotería de Bogotá?",
     "respuesta_ref": "No hay información.", "criterio": "Debe abstenerse.",
     "relevantes": {}, "rol_documentos": "más parecido (no responde)", "documentos": []},
]


def _chat_falso(modelo, system, user, opciones):
    if "lotería" in user:
        texto = "No hay evidencia suficiente para responder."
    else:
        texto = "Según las fuentes, el cambio climático es la causa [1]."
    return {"message": {"content": texto}, "prompt_eval_count": 300, "eval_count": 12,
            "eval_duration": 1_000_000_000, "load_duration": 0}


@pytest.fixture
def api_eval(entorno, coleccion_cargada, monkeypatch):
    from fastapi.testclient import TestClient

    import app.api
    from app import llm

    monkeypatch.setattr(app.api, "cargar_encoder", lambda: None)
    monkeypatch.setattr(llm, "version", lambda: "test")
    monkeypatch.setattr(llm, "listar_modelos", lambda: [
        {"nombre": "qwen2.5:1.5b", "tamano_gb": 1.0, "familia": "qwen2", "parametros": "1.5B",
         "cuantizacion": "Q4_K_M", "digest": "abc"}])
    monkeypatch.setattr(llm, "modelos_en_memoria", lambda: [])
    monkeypatch.setattr(llm, "cargar_en_memoria", lambda m: {"modelo": m, "carga_s": 0.0})
    monkeypatch.setattr(llm, "chat", _chat_falso)
    gold = entorno / "gold.jsonl"
    gold.write_text("\n".join(json.dumps(q, ensure_ascii=False) for q in GOLD), encoding="utf-8")
    return TestClient(app.api.app), gold, entorno / "eval"


@requiere_milvus_lite
def test_flujo_completo_y_reanudacion(api_eval):
    from scripts import evaluar_gold

    cliente, gold, salida = api_eval
    base = ["--gold", str(gold), "--salida", str(salida), "--api", "", "--modelos", "qwen2.5:1.5b", "--k-max", "5"]
    assert evaluar_gold.main(["todo", *base], sesion=cliente) == 0

    rec = [json.loads(l) for l in (salida / "recuperacion_crudo.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [r["gold_id"] for r in rec] == ["G001", "G002", "G003"]
    assert rec[0]["metricas"]["hit@5"] == 1.0 and rec[2]["metricas"] is None

    gen = [json.loads(l) for l in (salida / "generacion_crudo.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(gen) == 3 and not any(g.get("error") for g in gen)
    por_id = {g["gold_id"]: g for g in gen}
    assert por_id["G003"]["abstencion_detectada"] and por_id["G003"]["abstencion_correcta_auto"]
    assert por_id["G001"]["contexto_con_relevante"] is True and por_id["G001"]["estado_citas"] == "referencias_existentes"

    # Reanudar no repite lo que ya está hecho.
    assert evaluar_gold.main(["generacion", *base], sesion=cliente) == 0
    assert len((salida / "generacion_crudo.jsonl").read_text(encoding="utf-8").splitlines()) == 3

    resumen = (salida / "resumen_generacion.csv").read_text(encoding="utf-8")
    assert "qwen2.5:1.5b" in resumen and (salida / "resumen_recuperacion.csv").is_file()
    assert (salida / "config_generacion.json").is_file()


@requiere_milvus_lite
def test_planilla_y_calificaciones(api_eval):
    from openpyxl import load_workbook

    from scripts import evaluar_gold

    cliente, gold, salida = api_eval
    base = ["--gold", str(gold), "--salida", str(salida), "--api", "", "--modelos", "qwen2.5:1.5b",
            "--fraccion-doble", "1.0"]
    evaluar_gold.main(["generacion", *base], sesion=cliente)
    evaluar_gold.main(["planilla", *base], sesion=cliente)

    ruta = salida / "planilla_evaluacion.xlsx"
    libro = load_workbook(ruta)
    assert libro.sheetnames == ["Instrucciones", "Evaluador 1", "Evaluador 2"]
    for hoja, notas in (("Evaluador 1", [3, 1, 3]), ("Evaluador 2", [3, 0, 3])):
        h = libro[hoja]
        assert h.max_row == 4 and h["N1"].value == "Corrección (0-3)"
        for fila, nota in zip(range(2, 5), notas):
            h[f"N{fila}"] = nota
            h[f"O{fila}"] = 2
            h[f"P{fila}"] = "NA" if h[f"C{fila}"].value == "sin_respuesta" else 2
            h[f"Q{fila}"] = "Sí"
    libro.save(ruta)

    assert evaluar_gold.main(["calificaciones", *base], sesion=cliente) == 0
    acuerdo = {r["criterio"]: r for r in
               __import__("csv").DictReader((salida / "acuerdo_evaluadores.csv").open(encoding="utf-8"))}
    assert int(acuerdo["Corrección (0-3)"]["pares"]) == 3
    assert float(acuerdo["Corrección (0-3)"]["acuerdo_exacto"]) == pytest.approx(2 / 3, abs=1e-3)
    promedios = (salida / "calificaciones_por_modelo.csv").read_text(encoding="utf-8")
    assert "qwen2.5:1.5b" in promedios


@requiere_milvus_lite
def test_ablacion_de_estrategia_escribe_archivos_aparte(api_eval):
    from scripts import evaluar_gold

    cliente, gold, salida = api_eval
    base = ["--gold", str(gold), "--salida", str(salida), "--api", "", "--k-max", "8"]
    assert evaluar_gold.main(["recuperacion", *base, "--estrategia", "convencional"], sesion=cliente) == 0
    assert (salida / "resumen_recuperacion_convencional.csv").is_file()
    assert (salida / "recuperacion_convencional_por_pregunta.csv").is_file()
    encabezado = (salida / "resumen_recuperacion_convencional.csv").read_text(encoding="utf-8").splitlines()[0]
    assert "recall@8" in encabezado and "recall@10" not in encabezado
