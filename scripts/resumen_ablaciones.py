"""Tabla y gráfica de las ablaciones, a partir de reports/eval/ y reports/ablaciones/.

    docker compose exec api python -m scripts.resumen_ablaciones

Recuperación (gold, conjunto de prueba, a nivel de documento), una fila por condición:
  base            colección del experimento (vectores del notebook), diversificada
  convencional    misma colección, recuperación convencional
  base_corpus     struct256o32 + e5-small reconstruida desde el corpus (control de reproducibilidad)
  chunk128 / chunk512   tamaño de chunk 128 y 512 tokens (solapamiento 1/8)
  bge_m3          encoder BAAI/bge-m3 (otra colección)
  muestra10k      muestra de 10.000 documentos (struct256o32 + e5-small)
Generación (qwen2.5:1.5b, calificación humana): base, ctx_convencional y k3.

Escribe reports/ablaciones/resumen_ablaciones_recuperacion.csv,
resumen_ablaciones_generacion.csv y graficas/5_ablaciones_recuperacion.png.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

CONDICIONES = [
    ("base", "Base: struct256o32 + e5-small, diversificada, 500 docs", "eval", "recuperacion"),
    ("convencional", "Estrategia de recuperación convencional", "eval", "recuperacion_convencional"),
    ("base_corpus", "Base reconstruida desde el corpus (control)", "ablaciones", "recuperacion_base_corpus"),
    ("chunk128", "Chunks de 128 tokens (solapamiento 16)", "ablaciones", "recuperacion_chunk128"),
    ("chunk512", "Chunks de 512 tokens (solapamiento 64)", "ablaciones", "recuperacion_chunk512"),
    ("bge_m3", "Encoder BAAI/bge-m3", "ablaciones", "recuperacion_bge_m3"),
    ("muestra10k", "Muestra de 10.000 documentos", "ablaciones", "recuperacion_muestra10k"),
]
GENERACION = [("base", "Base (k = 5, diversificada)"), ("ctx_convencional", "Contexto convencional (k = 5)"),
              ("k3", "k = 3, diversificada")]


def tabla_recuperacion(reports: Path, gold: pd.DataFrame) -> pd.DataFrame:
    idioma = gold.set_index("gold_id")["idioma_documento"].astype(str)
    filas = []
    for etiqueta, descripcion, carpeta, prefijo in CONDICIONES:
        ruta = reports / carpeta / f"{prefijo}_por_pregunta.csv"
        if not ruta.is_file():
            continue
        df = pd.read_csv(ruta)
        df = df[df["n_relevantes"] > 0]
        test = df[df["split"] == "test"]
        cruce = df[(df["idioma_pregunta"] == "es") & (df["gold_id"].map(idioma).str.strip() == "en")]
        fila = {"condicion": etiqueta, "descripcion": descripcion,
                "hit@1": test["hit@1"].mean(), "hit@5": test["hit@5"].mean(), "mrr": test["rr"].mean(),
                "ndcg@5": test["ndcg@5"].mean(), "recall@5": test["recall@5"].mean(),
                "hit@5_40_preguntas": df["hit@5"].mean(),
                "es_a_en_hit@5": f"{int(cruce['hit@5'].sum())} de {len(cruce)}"}
        if "relevantes_en_coleccion" in df:
            fila["cobertura_gold"] = f"{int(df['relevantes_en_coleccion'].sum())} de {int(df['n_relevantes'].sum())}"
        if "busqueda_s" in df and df["busqueda_s"].notna().any():
            fila["busqueda_p50_s"] = df["busqueda_s"].median()
        config = reports / carpeta / f"config_{prefijo}.json"
        if config.is_file():
            salud = json.loads(config.read_text(encoding="utf-8")).get("salud") or {}
            fila["chunks_indexados"] = (salud.get("milvus") or {}).get("registros")
        filas.append(fila)
    return pd.DataFrame(filas).round(3)


def tabla_generacion(reports: Path, modelo: str = "qwen2.5:1.5b") -> pd.DataFrame:
    filas = []
    for etiqueta, descripcion in GENERACION:
        if etiqueta == "base":
            cal, lat = reports / "eval" / "calificaciones_por_modelo.csv", reports / "eval" / "resumen_generacion.csv"
        else:
            cal = reports / "eval" / f"calificaciones_{etiqueta}.csv"
            lat = reports / "eval" / f"resumen_generacion_{etiqueta}.csv"
        if not (cal.is_file() and lat.is_file()):
            continue
        c = pd.read_csv(cal)
        c = c[(c["conjunto"] == "test") & (c["modelo"] == modelo)]
        g = pd.read_csv(lat)
        g = g[(g["conjunto"] == "test") & (g["modelo"] == modelo)]
        if c.empty or g.empty:
            continue
        c, g = c.iloc[0], g.iloc[0]
        exactitud = c.get("exactitud", c.get("correctas (=3)"))
        filas.append({
            "condicion": etiqueta, "descripcion": descripcion, "exactitud": exactitud,
            "fidelidad": c.get("fidelidad_media", c.get("Fidelidad (0-3)")),
            "citas": c.get("citas_media", c.get("Citas (0-3)")),
            "contexto_con_relevante": g["contexto_con_relevante"], "total_p50_s": g["total_p50_s"],
            "tokens_entrada_media": g["tokens_entrada_media"],
            "respuestas_recalificadas": c.get("respuestas_nuevas", 0),
        })
    return pd.DataFrame(filas).round(3)


def grafica(tabla: pd.DataFrame, ruta: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  (matplotlib no está instalado: se omite la gráfica)")
        return
    t = tabla.iloc[::-1]
    fig, ax = plt.subplots(figsize=(9, 0.5 * len(t) + 1.8))
    colores = ["#2a78d6" if c == "base" else "#a3a19a" for c in t["condicion"]]
    ax.barh(t["descripcion"], t["hit@5"], color=colores, height=0.55)
    for y, v in enumerate(t["hit@5"]):
        ax.text(v + 0.01, y, f"{v:.2f}".replace(".", ","), va="center", fontsize=9.5)
    ax.set_xlim(0, 1.08)
    ax.set_xlabel("hit@5 en el conjunto de prueba (30 preguntas con documentos relevantes)")
    for lado in ("top", "right"):
        ax.spines[lado].set_visible(False)
    fig.text(0.01, 0.97, "Ablaciones de recuperación con el gold", fontsize=13, fontweight="bold", va="top")
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    ruta.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(ruta, dpi=200)
    plt.close(fig)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--reports", type=Path, default=Path("reports"))
    p.add_argument("--gold", type=Path, default=Path("data/gold/gold_v1.jsonl"))
    a = p.parse_args(argv)
    gold = pd.DataFrame([json.loads(l) for l in a.gold.read_text(encoding="utf-8").splitlines() if l.strip()])
    salida = a.reports / "ablaciones"
    salida.mkdir(parents=True, exist_ok=True)
    rec = tabla_recuperacion(a.reports, gold)
    gen = tabla_generacion(a.reports)
    rec.to_csv(salida / "resumen_ablaciones_recuperacion.csv", index=False, encoding="utf-8")
    gen.to_csv(salida / "resumen_ablaciones_generacion.csv", index=False, encoding="utf-8")
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print("Recuperación (test):\n" + rec.to_string(index=False))
        print("\nGeneración, qwen2.5:1.5b (test):\n" + (gen.to_string(index=False) if len(gen) else "(sin datos aún)"))
    if len(rec):
        grafica(rec, a.reports / "eval" / "graficas" / "5_ablaciones_recuperacion.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
