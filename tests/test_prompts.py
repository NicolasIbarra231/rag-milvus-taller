"""Presupuesto de contexto, validación de citas y abstención."""
import pytest

from app import prompts

EVIDENCIAS = [{"text": "a" * 3000, "title": "T1", "url": "u1"},
              {"text": "b" * 3000, "title": "T2", "url": "u2"},
              {"text": "c" * 3000, "title": "T3", "url": "u3"}]


def test_contexto_respeta_presupuesto_y_numera():
    contexto, fuentes, truncado = prompts.construir_contexto(EVIDENCIAS, max_caracteres=6000)
    assert len(contexto) <= 6000
    assert [f["referencia"] for f in fuentes] == list(range(1, len(fuentes) + 1))
    assert truncado
    assert contexto.startswith("[1]\nTítulo: T1")


def test_contexto_sin_truncar():
    _, fuentes, truncado = prompts.construir_contexto([{"text": "corto", "title": "", "url": ""}], 6000)
    assert len(fuentes) == 1 and not truncado


@pytest.mark.parametrize("respuesta,estado", [
    ("Los glaciares retroceden [1]. La sequía afecta embalses [2].", "referencias_existentes"),
    ("Los glaciares retroceden.", "respuesta_sin_citas"),
    ("Dato inventado [7].", "referencias_inexistentes"),
    ("No encuentro evidencia suficiente en los documentos.", "abstencion"),
    ("Los documentos no permiten responder: no hay evidencia suficiente.", "abstencion"),
])
def test_estados_de_citas(respuesta, estado):
    fuentes = [{"referencia": 1}, {"referencia": 2}]
    assert prompts.evaluar_citas(respuesta, fuentes)["estado"] == estado


def test_prompts_versionados_y_huella_estable():
    assert {"rag_v2", "rag_v4", "rag_v5"} <= set(prompts.PROMPTS)
    assert prompts.huella_prompt("rag_v5") == prompts.huella_prompt("rag_v5")
    with pytest.raises(ValueError):
        prompts.obtener("rag_v99")
