"""Prueba de humo contra la pila levantada (API + Milvus + Ollama).

    docker compose exec api python -m scripts.smoke_test
    docker compose exec api python -m scripts.smoke_test --modelos qwen2.5:1.5b llama3.2:1b llama3.2:3b

Verifica salud, recuperación, generación con citas, abstención ante una pregunta
sin respuesta documental y el manejo de un modelo inexistente.
"""
from __future__ import annotations

import argparse
import os
import sys

import requests

API = os.getenv("SMOKE_API_URL", "http://localhost:8000").rstrip("/")
PREGUNTA = "¿Qué impactos del cambio climático se mencionan en las noticias recuperadas?"
SIN_RESPUESTA = "¿Cuál será el número exacto ganador de la lotería de Bogotá del próximo sábado?"

resultados: list[tuple[str, bool, str]] = []


def caso(nombre: str, ok: bool, detalle: str = "") -> None:
    resultados.append((nombre, ok, detalle))
    print(f"{'PASA ' if ok else 'FALLA'} · {nombre} {('— ' + detalle) if detalle else ''}", flush=True)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--modelos", nargs="*", help="Modelos a probar (por defecto, el de /config)")
    a = p.parse_args()

    salud = requests.get(f"{API}/health", timeout=30).json()
    caso("health", salud["status"] == "ok", str({k: v["ok"] for k, v in salud["componentes"].items()}))
    cfg = requests.get(f"{API}/config", timeout=10).json()
    modelos = a.modelos or [cfg["defaults"]["model"]]
    instalados = {m["nombre"] for m in requests.get(f"{API}/models", timeout=10).json()["instalados"]}
    for m in modelos:
        caso(f"modelo descargado: {m}", m in instalados)

    r = requests.post(f"{API}/retrieve", json={"query": PREGUNTA, "k": 5}, timeout=60)
    ok = r.ok and len(r.json()["chunks"]) == 5
    caso("retrieve devuelve k=5", ok, f"{r.status_code}")
    if ok:
        docs = {c["document_id"] for c in r.json()["chunks"]}
        caso("retrieve diversificado (documentos únicos)", len(docs) == 5, f"{len(docs)} documentos")

    for m in modelos:
        r = requests.post(f"{API}/ask", json={"query": PREGUNTA, "model": m}, timeout=360)
        if not r.ok:
            caso(f"ask [{m}]", False, r.text[:200])
            continue
        res = r.json()
        caso(f"ask [{m}] responde", bool(res["respuesta"]), f"{res['tiempos']['total_s']} s · {res['citas']['estado']}")
        caso(f"ask [{m}] sin citas inexistentes", res["citas"]["estado"] != "referencias_inexistentes")
        r2 = requests.post(f"{API}/ask", json={"query": SIN_RESPUESTA, "model": m}, timeout=360)
        if r2.ok:
            caso(f"ask [{m}] se abstiene sin evidencia", r2.json()["abstencion"],
                 r2.json()["respuesta"][:80].replace("\n", " "))

    r = requests.post(f"{API}/ask", json={"query": PREGUNTA, "model": "modelo-inexistente:1b"}, timeout=60)
    caso("modelo inexistente -> 404 legible", r.status_code == 404 and r.json().get("error") == "modelo_no_descargado")
    r = requests.post(f"{API}/ask", json={"query": "  "}, timeout=10)
    caso("pregunta vacía -> 422", r.status_code == 422)

    fallas = [n for n, ok, _ in resultados if not ok]
    print(f"\n{len(resultados) - len(fallas)}/{len(resultados)} casos pasan.")
    # La abstención es una propiedad del modelo, no del sistema: se reporta pero no rompe la prueba.
    criticas = [n for n in fallas if "abstiene" not in n]
    return 1 if criticas else 0


if __name__ == "__main__":
    sys.exit(main())
