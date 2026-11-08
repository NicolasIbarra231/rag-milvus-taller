"""API: camino feliz, errores tipados y registro JSONL (Ollama simulado)."""
import json

import pytest

from tests.conftest import requiere_milvus_lite

pytestmark = requiere_milvus_lite


@pytest.fixture
def cliente(entorno, monkeypatch):
    from fastapi.testclient import TestClient

    import app.api
    from app import llm

    monkeypatch.setattr(app.api, "cargar_encoder", lambda: None)
    monkeypatch.setattr(llm, "version", lambda: "test")
    monkeypatch.setattr(llm, "listar_modelos", lambda: [
        {"nombre": "qwen2.5:1.5b", "tamano_gb": 1.0, "familia": "qwen2", "parametros": "1.5B",
         "cuantizacion": "Q4_K_M", "digest": "abc"}])
    monkeypatch.setattr(llm, "modelos_en_memoria", lambda: [])
    return TestClient(app.api.app)  # sin `with`: no se ejecuta el lifespan (no descarga el encoder)


def _chat_ok(modelo, system, user, opciones):
    assert "FUENTES DOCUMENTALES" in user and "[1]" in user
    return {"message": {"content": "Los glaciares patagónicos retroceden por el cambio climático [1]."},
            "prompt_eval_count": 500, "eval_count": 20, "eval_duration": 1_000_000_000,
            "load_duration": 100_000_000}


def test_vivo(cliente):
    assert cliente.get("/health/live").json() == {"status": "ok"}


def test_coleccion_vacia_409(cliente):
    r = cliente.post("/ask", json={"query": "¿Qué pasa con el clima?"})
    assert r.status_code == 409 and r.json()["error"] == "coleccion_vacia"


def test_ask_camino_feliz_y_log(cliente, coleccion_cargada, monkeypatch, entorno):
    from app import llm

    monkeypatch.setattr(llm, "chat", _chat_ok)
    r = cliente.post("/ask", json={"query": "cambio climático glaciares", "model": "qwen2.5:1.5b", "k": 3})
    assert r.status_code == 200, r.text
    res = r.json()
    assert res["citas"]["estado"] == "referencias_existentes"
    assert res["evidencias"][0]["citada"] is True and res["evidencias"][0]["url"]
    assert len({e["document_id"] for e in res["evidencias"]}) == len(res["evidencias"])  # diversificada
    assert res["parametros"]["prompt_version"] == "rag_v5" and res["parametros"]["seed"] == 42
    assert res["tokens"]["por_segundo"] == 20.0
    linea = json.loads((entorno / "logs" / "runs.jsonl").read_text(encoding="utf-8").splitlines()[-1])
    assert linea["run_id"] == res["run_id"] and "text" not in linea["evidencias"][0]


def test_salud_completa(cliente, coleccion_cargada):
    r = cliente.get("/health").json()
    assert r["status"] == "ok" and r["componentes"]["milvus"]["registros"] == 5


@pytest.mark.parametrize("error,http,codigo", [
    ("ModeloNoDescargado", 404, "modelo_no_descargado"),
    ("OllamaNoDisponible", 503, "ollama_no_disponible"),
    ("OllamaTimeout", 504, "ollama_timeout"),
])
def test_errores_de_ollama(cliente, coleccion_cargada, monkeypatch, entorno, error, http, codigo):
    from app import llm

    def falla(*_a, **_k):
        clase = getattr(llm, error)
        raise clase("gemma2:2b") if error == "ModeloNoDescargado" else clase("simulado")

    monkeypatch.setattr(llm, "chat", falla)
    r = cliente.post("/ask", json={"query": "cambio climático", "model": "gemma2:2b"})
    assert r.status_code == http and r.json()["error"] == codigo and r.json()["sugerencia"]
    ultima = json.loads((entorno / "logs" / "runs.jsonl").read_text(encoding="utf-8").splitlines()[-1])
    assert ultima["error"] == error  # los errores también quedan registrados


@pytest.mark.parametrize("cuerpo", [
    {"query": "   "},
    {"query": "hola mundo", "k": 0},
    {"query": "hola mundo", "estrategia": "aleatoria"},
    {"query": "hola mundo", "prompt_version": "rag_v99"},
])
def test_validacion_422(cliente, cuerpo):
    assert cliente.post("/ask", json=cuerpo).status_code == 422


def test_filtro_invalido_422(cliente, coleccion_cargada):
    r = cliente.post("/retrieve", json={"query": "clima", "filtros": {"colecciones": ["twitter"]}})
    assert r.status_code == 422 and r.json()["error"] == "parametro_invalido"


def test_sin_evidencia_se_abstiene_sin_llm(cliente, coleccion_cargada, monkeypatch):
    from app import llm

    monkeypatch.setattr(llm, "chat", lambda *a, **k: pytest.fail("no debe llamar al LLM"))
    # Filtro que no deja ningún resultado: idioma húngaro (no existe en los datos de prueba).
    r = cliente.post("/ask", json={"query": "cambio climático", "filtros": {"idioma": "hu"}})
    assert r.status_code == 200 and r.json()["abstencion"] is True


def test_models_reporta_faltantes(cliente):
    r = cliente.get("/models").json()
    assert "llama3.2:3b" in r["faltantes_experimento"]
