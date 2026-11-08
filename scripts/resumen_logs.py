"""Resume reports/logs/rag_runs.jsonl por modelo: p50/p95, tokens/s, errores, abstención.

    docker compose exec api python -m scripts.resumen_logs
    docker compose exec api python -m scripts.resumen_logs --excluir-frio --csv reports/resumen_latencias.csv

Las ejecuciones con arranque en frío (carga del modelo > 1 s) se pueden excluir
para no mezclar el costo de cargar el modelo con la latencia de generación.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from app.config import settings


def cargar(ruta: Path) -> pd.DataFrame:
    filas = []
    for linea in ruta.read_text(encoding="utf-8").splitlines():
        if not linea.strip():
            continue
        r = json.loads(linea)
        t, tok = r.get("tiempos") or {}, r.get("tokens") or {}
        filas.append({
            "modelo": r.get("modelo"), "error": r.get("error"),
            "prompt_version": (r.get("parametros") or {}).get("prompt_version"),
            "total_s": t.get("total_s"), "recuperacion_s": t.get("recuperacion_s"),
            "generacion_s": t.get("generacion_s"), "carga_modelo_s": t.get("carga_modelo_s"),
            "tokens_por_s": tok.get("por_segundo"), "tokens_salida": tok.get("salida"),
            "abstencion": r.get("abstencion"), "citas": (r.get("citas") or {}).get("estado"),
        })
    return pd.DataFrame(filas)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ruta", type=Path, default=Path(settings().ruta_logs))
    p.add_argument("--excluir-frio", action="store_true")
    p.add_argument("--csv", type=Path)
    a = p.parse_args()
    df = cargar(a.ruta)
    if df.empty:
        raise SystemExit(f"No hay ejecuciones en {a.ruta}")
    ok = df[df["error"].isna()]
    if a.excluir_frio:
        ok = ok[ok["carga_modelo_s"].fillna(0) <= 1]
    q = lambda c, x: ok.groupby("modelo")[c].quantile(x)  # noqa: E731
    resumen = pd.DataFrame({
        "ejecuciones": df.groupby("modelo").size(),
        "tasa_error": df.groupby("modelo")["error"].apply(lambda s: s.notna().mean()),
        "total_p50_s": q("total_s", .5), "total_p95_s": q("total_s", .95),
        "recuperacion_p50_s": q("recuperacion_s", .5), "recuperacion_p95_s": q("recuperacion_s", .95),
        "generacion_p50_s": q("generacion_s", .5), "generacion_p95_s": q("generacion_s", .95),
        "tokens_por_s_media": ok.groupby("modelo")["tokens_por_s"].mean(),
        "tasa_abstencion": ok.assign(abstencion=ok["abstencion"].astype(float)).groupby("modelo")["abstencion"].mean(),
        "tasa_sin_citas": ok.groupby("modelo")["citas"].apply(lambda s: (s == "respuesta_sin_citas").mean()),
    }).round(3)
    print(resumen.to_string())
    if a.csv:
        a.csv.parent.mkdir(parents=True, exist_ok=True)
        resumen.to_csv(a.csv)
        print(f"\nGuardado en {a.csv}")


if __name__ == "__main__":
    main()
