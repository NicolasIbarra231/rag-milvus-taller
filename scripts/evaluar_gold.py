"""Evaluación con el gold estándar (punto 6 del taller), contra la API desplegada.

Corre dentro del contenedor de la API, con el gold en data/gold/:

    docker compose exec api python -m scripts.evaluar_gold recuperacion   # minutos
    docker compose exec api python -m scripts.evaluar_gold generacion     # horas en CPU; se puede reanudar
    docker compose exec api python -m scripts.evaluar_gold planilla       # Excel para los evaluadores
    docker compose exec api python -m scripts.evaluar_gold resumen        # métricas automáticas
    docker compose exec api python -m scripts.evaluar_gold calificaciones # tras la evaluación humana

Todo queda en reports/eval/. La generación escribe una línea por respuesta apenas
la recibe: si se interrumpe (Ctrl+C, apagón, timeout), al volver a ejecutarla
continúa donde iba y no repite lo ya hecho.

Qué mide cada paso:
- recuperacion: Recall@k, MRR, nDCG@k y hit@k a nivel de documento (app/evaluacion.py).
- generacion: la respuesta de cada modelo a cada pregunta, con evidencias, citas y tiempos.
- resumen: abstención correcta (detector automático), citas, contexto con relevante,
  latencias p50/p95, tokens/s y errores, por modelo y por conjunto (test, dev, todos).
- planilla: Excel con una hoja por evaluador, la rúbrica y listas desplegables.
- calificaciones: promedios humanos por modelo y acuerdo entre evaluadores (kappa).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import random
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

from app.evaluacion import (KS_POR_DEFECTO, abstencion_correcta, documentos_de_chunk,
                            metricas_recuperacion, percentil, relevantes_en_evidencias)

GOLD_POR_DEFECTO = Path("data/gold/gold_v1.jsonl")
SALIDA_POR_DEFECTO = Path("reports/eval")
API_POR_DEFECTO = os.getenv("EVAL_API_URL", "http://localhost:8000")
TIMEOUT_ASK = 420          # la API corta a los 300 s de Ollama; margen para recuperación y red
CONJUNTOS = ("test", "dev", "todos")


# ------------------------------------------------------------------- utilidades
class ErrorApi(Exception):
    def __init__(self, http: int, codigo: str, detalle: str):
        super().__init__(f"{http} {codigo}: {detalle}")
        self.http, self.codigo, self.detalle = http, codigo, detalle


class Api:
    """Cliente mínimo. `sesion` puede ser requests.Session o el TestClient de FastAPI."""

    def __init__(self, base: str, sesion=None):
        self.base = base.rstrip("/")
        self.sesion = sesion or requests.Session()

    def _respuesta(self, r):
        if r.status_code >= 400:
            try:
                cuerpo = r.json()
            except ValueError:
                cuerpo = {"error": "http_error", "detalle": r.text[:300]}
            raise ErrorApi(r.status_code, cuerpo.get("error", "error"), cuerpo.get("detalle", ""))
        return r.json()

    def get(self, ruta: str, timeout: float = 30):
        return self._respuesta(self.sesion.get(self.base + ruta, timeout=timeout))

    def post(self, ruta: str, cuerpo: dict, timeout: float = 60):
        return self._respuesta(self.sesion.post(self.base + ruta, json=cuerpo, timeout=timeout))


def leer_gold(ruta: Path, conjunto: str = "todos") -> list[dict]:
    if not ruta.is_file():
        sys.exit(f"No existe {ruta}. Copie gold_v1.jsonl en data/gold/ (ver README, sección Evaluación).")
    preguntas = [json.loads(l) for l in ruta.read_text(encoding="utf-8").splitlines() if l.strip()]
    faltan = {"gold_id", "split", "tipo", "pregunta", "relevantes"} - set(preguntas[0])
    if faltan:
        sys.exit(f"El gold no tiene las columnas {sorted(faltan)}.")
    if conjunto != "todos":
        preguntas = [p for p in preguntas if p["split"] == conjunto]
    return preguntas


def leer_jsonl(ruta: Path) -> list[dict]:
    if not ruta.is_file():
        return []
    return [json.loads(l) for l in ruta.read_text(encoding="utf-8").splitlines() if l.strip()]


def agregar_jsonl(ruta: Path, fila: dict) -> None:
    ruta.parent.mkdir(parents=True, exist_ok=True)
    with ruta.open("a", encoding="utf-8") as f:
        f.write(json.dumps(fila, ensure_ascii=False, default=str) + "\n")
        f.flush()


def ultima_por_clave(filas: list[dict]) -> dict[tuple, dict]:
    """Última fila por (gold_id, modelo); una exitosa siempre gana a un error."""
    salida: dict[tuple, dict] = {}
    for f in filas:
        clave = (f["gold_id"], f["modelo"])
        previa = salida.get(clave)
        if previa is None or not f.get("error") or previa.get("error"):
            salida[clave] = f
    return salida


def sha256_archivo(ruta: Path) -> str:
    return hashlib.sha256(ruta.read_bytes()).hexdigest()


def guardar_configuracion(api: Api, gold: Path, salida: Path, extra: dict) -> None:
    """Instantánea de lo que se evaluó: sin ella, los resultados no son reproducibles."""
    cfg = api.get("/config")
    salud = api.get("/health", timeout=60)
    try:
        modelos = api.get("/models")["instalados"]
    except ErrorApi:
        modelos = []
    registro = {
        "fecha_utc": datetime.now(timezone.utc).isoformat(),
        "gold": {"archivo": str(gold), "sha256": sha256_archivo(gold)},
        "parametros_rag": cfg["defaults"], "coleccion": cfg["coleccion_milvus"],
        "contrato_encoder": cfg["contrato_encoder"], "salud": salud.get("componentes"),
        "modelos_instalados": modelos, "plataforma": platform.platform(), **extra,
    }
    salida.mkdir(parents=True, exist_ok=True)
    nombre = f"config_{extra.get('paso', 'evaluacion')}.json"
    (salida / nombre).write_text(json.dumps(registro, indent=2, ensure_ascii=False), encoding="utf-8")


# ------------------------------------------------------------------ recuperación
def nombre_recuperacion(estrategia: str | None, etiqueta: str | None = None) -> str:
    if etiqueta:
        return f"recuperacion_{etiqueta}"
    return "recuperacion" if not estrategia else f"recuperacion_{estrategia}"


def nombre_generacion(etiqueta: str | None) -> str:
    return "generacion_crudo.jsonl" if not etiqueta else f"generacion_{etiqueta}_crudo.jsonl"


def cobertura_local():
    """En modo --local: qué document_id del gold existen en la colección configurada."""
    from app import milvus_store
    from app.config import settings

    def presentes(ids: set[str]) -> set[str]:
        if not ids:
            return set()
        lista = ", ".join(json.dumps(i) for i in sorted(ids))
        filas = milvus_store.cliente().query(settings().coleccion, filter=f"document_id in [{lista}]",
                                             output_fields=["document_id"], limit=16384)
        return {f["document_id"] for f in filas}
    return presentes


def correr_recuperacion(api: Api, preguntas: list[dict], k_max: int, salida: Path,
                        estrategia: str | None = None, etiqueta: str | None = None,
                        cobertura=None) -> pd.DataFrame:
    nombre = nombre_recuperacion(estrategia, etiqueta)
    ks = tuple(sorted({k for k in KS_POR_DEFECTO if k <= k_max} | {k_max}))
    crudo, filas = [], []
    for n, q in enumerate(preguntas, 1):
        cuerpo = {"query": q["pregunta"], "k": k_max}
        if estrategia:
            cuerpo["estrategia"] = estrategia
        r = api.post("/retrieve", cuerpo, timeout=120)
        ranking = []
        for i, c in enumerate(r["chunks"], 1):
            docs = documentos_de_chunk(c)
            grado = max((int(q["relevantes"].get(d, 0)) for d in docs), default=0)
            ranking.append({"rango": i, "chunk_id": c.get("chunk_id"), "document_id": c.get("document_id"),
                            "documentos": docs, "score": c.get("score"), "grado": grado,
                            "titulo": c.get("title"), "coleccion": c.get("collection_name")})
        met = (metricas_recuperacion([x["documentos"] for x in ranking], q["relevantes"], ks)
               if q["relevantes"] else None)
        crudo.append({"gold_id": q["gold_id"], "split": q["split"], "tipo": q["tipo"],
                      "pregunta": q["pregunta"], "relevantes": q["relevantes"], "ranking": ranking,
                      "metricas": met, "tiempos": r.get("metricas")})
        fila = {"gold_id": q["gold_id"], "split": q["split"], "tipo": q["tipo"],
                "idioma_pregunta": q.get("idioma_pregunta"), "n_relevantes": len(q["relevantes"]),
                "recuperacion_s": (r.get("metricas") or {}).get("tiempo_recuperacion_s"),
                "busqueda_s": (r.get("metricas") or {}).get("tiempo_busqueda_s"), **(met or {})}
        if cobertura is not None:
            fila["relevantes_en_coleccion"] = len(cobertura(set(q["relevantes"])))
        filas.append(fila)
        marca = "sin relevantes (sin_respuesta)" if met is None else (
            f"primer relevante en rango {met['rango_primer_relevante']}" if met["rango_primer_relevante"]
            else f"ningún relevante en top-{k_max}")
        print(f"  [{n:>2}/{len(preguntas)}] {q['gold_id']} {q['tipo']:<13} {marca}", flush=True)

    salida.mkdir(parents=True, exist_ok=True)
    with (salida / f"{nombre}_crudo.jsonl").open("w", encoding="utf-8") as f:
        for c in crudo:
            f.write(json.dumps(c, ensure_ascii=False, default=str) + "\n")
    df = pd.DataFrame(filas)
    df.to_csv(salida / f"{nombre}_por_pregunta.csv", index=False, encoding="utf-8")
    return df


def resumen_recuperacion(df: pd.DataFrame) -> pd.DataFrame:
    """Promedios por conjunto y por tipo. Excluye sin_respuesta (no tienen relevantes)."""
    con_rel = df[df["n_relevantes"] > 0].copy()
    metricas = [c for c in con_rel.columns if "@" in c] + ["rr"]
    bloques = []
    for conjunto in CONJUNTOS:
        parte = con_rel if conjunto == "todos" else con_rel[con_rel["split"] == conjunto]
        if parte.empty:
            continue
        fila = {"conjunto": conjunto, "tipo": "todas", "n": len(parte), **parte[metricas].mean().to_dict()}
        bloques.append(fila)
        for tipo, g in parte.groupby("tipo"):
            bloques.append({"conjunto": conjunto, "tipo": tipo, "n": len(g), **g[metricas].mean().to_dict()})
    res = pd.DataFrame(bloques).rename(columns={"rr": "mrr"})
    columnas = ["conjunto", "tipo", "n"] + sorted([c for c in res.columns if "@" in c],
                                                  key=lambda c: (c.split("@")[0], int(c.split("@")[1]))) + ["mrr"]
    return res[columnas].round(3)


# -------------------------------------------------------------------- generación
def registro_generacion(q: dict, modelo: str, res: dict) -> dict:
    evidencias = []
    for e in res.get("evidencias", []):
        docs = documentos_de_chunk(e)
        evidencias.append({
            "referencia": e.get("referencia"), "chunk_id": e.get("chunk_id"),
            "document_id": e.get("document_id"), "documentos": docs, "score": e.get("score"),
            "titulo": e.get("title"), "url": e.get("url"), "citada": e.get("citada"),
            "grado": max((int(q["relevantes"].get(d, 0)) for d in docs), default=0),
        })
    return {
        "gold_id": q["gold_id"], "split": q["split"], "tipo": q["tipo"],
        "idioma_pregunta": q.get("idioma_pregunta"), "modelo": modelo,
        "run_id": res.get("run_id"), "fecha_utc": res.get("fecha_utc"),
        "pregunta": q["pregunta"], "respuesta": res.get("respuesta"),
        "abstencion_detectada": bool(res.get("abstencion")),
        "abstencion_esperada": q["tipo"] == "sin_respuesta",
        "abstencion_correcta_auto": abstencion_correcta(q["tipo"], bool(res.get("abstencion"))),
        "estado_citas": (res.get("citas") or {}).get("estado"),
        "citas": (res.get("citas") or {}).get("citas"),
        "citas_inexistentes": (res.get("citas") or {}).get("citas_inexistentes"),
        **relevantes_en_evidencias(res.get("evidencias", []), q["relevantes"]),
        "evidencias": evidencias, "tiempos": res.get("tiempos"), "tokens": res.get("tokens"),
        "contexto": res.get("contexto"), "parametros": res.get("parametros"),
        "advertencias": res.get("advertencias"),
    }


def correr_generacion(api: Api, preguntas: list[dict], modelos: list[str], salida: Path,
                      etiqueta: str | None = None, k: int | None = None, estrategia: str | None = None) -> None:
    ruta = salida / nombre_generacion(etiqueta)
    extra = {c: v for c, v in (("k", k), ("estrategia", estrategia)) if v}
    hechos = {(f["gold_id"], f["modelo"]) for f in leer_jsonl(ruta) if not f.get("error")}
    instalados = {m["nombre"] for m in api.get("/models")["instalados"]}
    faltan = [m for m in modelos if m not in instalados]
    if faltan:
        sys.exit("Faltan modelos en Ollama: " + ", ".join(faltan) + "\nDescárguelos con: "
                 + " && ".join(f"docker compose exec ollama ollama pull {m}" for m in faltan))

    total = len(preguntas) * len(modelos)
    pendientes_total = sum(1 for m in modelos for q in preguntas if (q["gold_id"], m) not in hechos)
    print(f"Generación: {len(preguntas)} preguntas × {len(modelos)} modelos = {total} respuestas "
          f"({total - pendientes_total} ya hechas, {pendientes_total} pendientes).", flush=True)

    # Modelo por modelo: con OLLAMA_MAX_LOADED_MODELS=1, alternar modelos por pregunta
    # obligaría a recargar pesos en cada llamada y contaminaría las latencias.
    for modelo in modelos:
        pendientes = [q for q in preguntas if (q["gold_id"], modelo) not in hechos]
        if not pendientes:
            print(f"\n{modelo}: completo.", flush=True)
            continue
        print(f"\n{modelo}: precalentando…", flush=True)
        try:
            api.post("/models/load", {"model": modelo}, timeout=TIMEOUT_ASK)
        except (ErrorApi, requests.RequestException) as e:
            print(f"  No se pudo precalentar ({e}); se sigue: la primera respuesta marcará arranque en frío.")
        inicio = time.perf_counter()
        for n, q in enumerate(pendientes, 1):
            try:
                res = api.post("/ask", {"query": q["pregunta"], "model": modelo, **extra}, timeout=TIMEOUT_ASK)
                fila = registro_generacion(q, modelo, res)
                estado = f"{fila['tiempos']['total_s']:.1f} s · {fila['estado_citas']}"
            except ErrorApi as e:
                fila = {"gold_id": q["gold_id"], "split": q["split"], "tipo": q["tipo"], "modelo": modelo,
                        "fecha_utc": datetime.now(timezone.utc).isoformat(), "error": e.codigo,
                        "http": e.http, "detalle": e.detalle}
                estado = f"ERROR {e.http} {e.codigo}"
            except requests.RequestException as e:
                fila = {"gold_id": q["gold_id"], "split": q["split"], "tipo": q["tipo"], "modelo": modelo,
                        "fecha_utc": datetime.now(timezone.utc).isoformat(), "error": type(e).__name__,
                        "detalle": str(e)[:300]}
                estado = f"ERROR {type(e).__name__}"
            agregar_jsonl(ruta, fila)
            transcurrido = time.perf_counter() - inicio
            restante = transcurrido / n * (len(pendientes) - n)
            print(f"  [{n:>2}/{len(pendientes)}] {q['gold_id']} {q['tipo']:<13} {estado}"
                  f"  · faltan ~{restante / 60:.0f} min para {modelo}", flush=True)


def resumen_generacion(filas: list[dict]) -> pd.DataFrame:
    """Una fila por modelo y conjunto. `errores`/`tasa_error` = respuestas que quedaron fallidas;
    `intentos`/`tasa_error_intentos` = todas las llamadas, incluidas las que fallaron y se repitieron."""
    ultimas = list(ultima_por_clave(filas).values())
    bloques = []
    for conjunto in CONJUNTOS:
        intentos = defaultdict(list)
        for f in filas:
            if conjunto == "todos" or f["split"] == conjunto:
                intentos[f["modelo"]].append(f)
        parte = [f for f in ultimas if conjunto == "todos" or f["split"] == conjunto]
        por_modelo = defaultdict(list)
        for f in parte:
            por_modelo[f["modelo"]].append(f)
        for modelo, fs in sorted(por_modelo.items()):
            ok = [f for f in fs if not f.get("error")]
            sin_resp = [f for f in ok if f["tipo"] == "sin_respuesta"]
            con_resp = [f for f in ok if f["tipo"] != "sin_respuesta"]
            no_abst = [f for f in con_resp if not f["abstencion_detectada"]]
            calientes = [f for f in ok if (f["tiempos"] or {}).get("carga_modelo_s", 0) <= 1]
            media = lambda xs: (sum(xs) / len(xs)) if xs else None  # noqa: E731
            bloques.append({
                "conjunto": conjunto, "modelo": modelo, "n": len(fs), "errores": len(fs) - len(ok),
                "tasa_error": (len(fs) - len(ok)) / len(fs) if fs else None,
                "intentos": len(intentos[modelo]),
                "errores_en_intentos": sum(1 for f in intentos[modelo] if f.get("error")),
                "tasa_error_intentos": (sum(1 for f in intentos[modelo] if f.get("error")) / len(intentos[modelo])
                                        if intentos[modelo] else None),
                "abstencion_correcta": media([f["abstencion_correcta_auto"] for f in ok]),
                "se_abstiene_sin_respuesta": media([f["abstencion_detectada"] for f in sin_resp]),
                "abstencion_indebida": media([f["abstencion_detectada"] for f in con_resp]),
                "contexto_con_relevante": media([f["contexto_con_relevante"] for f in con_resp]),
                "sin_citas": media([f["estado_citas"] == "respuesta_sin_citas" for f in no_abst]),
                "citas_inexistentes": media([f["estado_citas"] == "referencias_inexistentes" for f in no_abst]),
                "precision_citas": media([f["precision_citas"] for f in no_abst if f["precision_citas"] is not None]),
                "total_p50_s": percentil([f["tiempos"]["total_s"] for f in calientes], 50),
                "total_p95_s": percentil([f["tiempos"]["total_s"] for f in calientes], 95),
                "recuperacion_p50_s": percentil([f["tiempos"]["recuperacion_s"] for f in calientes], 50),
                "recuperacion_p95_s": percentil([f["tiempos"]["recuperacion_s"] for f in calientes], 95),
                "generacion_p50_s": percentil([f["tiempos"]["generacion_s"] for f in calientes], 50),
                "generacion_p95_s": percentil([f["tiempos"]["generacion_s"] for f in calientes], 95),
                "tokens_por_s": media([f["tokens"]["por_segundo"] for f in calientes
                                       if (f["tokens"] or {}).get("por_segundo")]),
                "tokens_entrada_media": media([f["tokens"]["entrada"] for f in calientes
                                               if (f["tokens"] or {}).get("entrada")]),
                "arranques_en_frio_excluidos": len(ok) - len(calientes),
            })
    return pd.DataFrame(bloques).round(3)


# ---------------------------------------------------------------------- planilla
RUBRICA = [
    ("Corrección (0-3)", "3 = cumple por completo el criterio del gold. 2 = correcta en lo esencial, con una "
     "omisión o imprecisión menor. 1 = acierta solo en parte y omite lo esencial (una comparativa que usa un solo "
     "documento; una ambigua que toma una sola interpretación sin advertir la ambigüedad). 0 = incorrecta, "
     "contradice la referencia o se abstiene teniendo la respuesta en las fuentes. En sin_respuesta: 3 si dice que "
     "no hay información y no da ningún dato; 1 si se abstiene pero agrega información o citas ajenas a la "
     "pregunta; 0 si da un dato."),
    ("Fidelidad (0-3)", "3 = todo lo que afirma está en las fuentes recuperadas. 2 = un detalle menor sin respaldo "
     "o levemente distorsionado. 1 = varias afirmaciones sin respaldo, o una distorsión importante (por ejemplo, "
     "convierte una proyección en un hecho). 0 = se basa en información inventada o ajena a las fuentes, aunque "
     "sea cierta en el mundo. NA si se abstiene sin afirmar nada."),
    ("Citas (0-3)", "3 = cita [n] en cada afirmación y cada fuente respalda lo que se le atribuye. 2 = cita, pero "
     "falta la cita en alguna afirmación o una cita es imprecisa. 1 = cita, pero la mayoría de las citas apuntan a "
     "la fuente equivocada o no respaldan lo afirmado. 0 = no cita, o ninguna cita respalda lo afirmado. NA si se "
     "abstiene sin afirmar nada."),
    ("Abstención correcta", "Sí = se abstuvo en una sin_respuesta, o respondió en las demás. No = inventó en una "
     "sin_respuesta, o se abstuvo teniendo la respuesta en las fuentes."),
]


def elegir_doble(preguntas: list[dict], fraccion: float, semilla: int = 42) -> set[str]:
    """Submuestra para el segundo evaluador, estratificada por tipo (al menos una por tipo)."""
    if fraccion <= 0:
        return set()
    rnd = random.Random(semilla)
    por_tipo = defaultdict(list)
    for q in preguntas:
        por_tipo[q["tipo"]].append(q["gold_id"])
    elegidas = set()
    for tipo in sorted(por_tipo):
        ids = sorted(por_tipo[tipo])
        n = max(1, round(len(ids) * fraccion))
        elegidas.update(rnd.sample(ids, min(n, len(ids))))
    return elegidas


def construir_planilla(preguntas: list[dict], filas: list[dict], ruta: Path, fraccion_doble: float) -> dict:
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Font, PatternFill
    from openpyxl.worksheet.datavalidation import DataValidation

    gold = {q["gold_id"]: q for q in preguntas}
    ultimas = [f for f in ultima_por_clave(filas).values() if not f.get("error") and f["gold_id"] in gold]
    if not ultimas:
        sys.exit("No hay respuestas en generacion_crudo.jsonl. Ejecute primero el paso 'generacion'.")
    ultimas.sort(key=lambda f: (f["gold_id"], f["modelo"]))
    doble = elegir_doble([gold[g] for g in sorted({f["gold_id"] for f in ultimas})], fraccion_doble)

    encabezado = ["gold_id", "split", "tipo", "pregunta", "respuesta_ref", "criterio", "modelo", "respuesta",
                  "fuentes_citadas", "estado_citas (auto)", "abstencion_detectada (auto)",
                  "contexto_con_relevante (auto)", "doble_evaluacion",
                  "Corrección (0-3)", "Fidelidad (0-3)", "Citas (0-3)", "Abstención correcta", "Comentario"]
    anchos = [9, 7, 13, 45, 45, 45, 14, 60, 45, 16, 14, 14, 10, 12, 12, 12, 12, 35]

    def fuentes_citadas(f):
        return "\n".join(f"[{e['referencia']}] {e.get('titulo') or '(sin título)'} — {e.get('url') or ''}"
                         for e in f["evidencias"] if e.get("citada")) or "(ninguna)"

    libro = Workbook()
    instr = libro.active
    instr.title = "Instrucciones"
    lineas = [
        ("Planilla de evaluación humana — gold estándar", True),
        ("Cada fila es la respuesta de un modelo a una pregunta del gold. Califique las columnas en amarillo.", False),
        ("Evaluador 1 califica todas las filas. Evaluador 2 califica solo su hoja (submuestra marcada en "
         "'doble_evaluacion'), sin mirar las notas del evaluador 1: así se mide el acuerdo entre evaluadores.", False),
        ("Las columnas '(auto)' son del sistema y sirven de apoyo; no reemplazan su juicio. En particular, el "
         "detector automático de abstención solo reconoce frases exactas.", False),
        ("", False),
        ("Rúbrica (propuesta; ajústela con el responsable del gold antes de empezar)", True),
    ] + [(f"{nombre}: {texto}", False) for nombre, texto in RUBRICA]
    for i, (texto, negrita) in enumerate(lineas, 1):
        celda = instr.cell(row=i, column=1, value=texto)
        celda.font = Font(bold=negrita, size=12 if negrita else 11)
        celda.alignment = Alignment(wrap_text=True, vertical="top")
    instr.column_dimensions["A"].width = 140

    amarillo = PatternFill("solid", fgColor="FFF2CC")
    gris = PatternFill("solid", fgColor="E7E6E6")
    for nombre_hoja, solo_doble in (("Evaluador 1", False), ("Evaluador 2", True)):
        hoja = libro.create_sheet(nombre_hoja)
        hoja.append(encabezado)
        for f in ultimas:
            if solo_doble and f["gold_id"] not in doble:
                continue
            q = gold[f["gold_id"]]
            hoja.append([f["gold_id"], f["split"], f["tipo"], q["pregunta"], q.get("respuesta_ref"),
                         q.get("criterio"), f["modelo"], f["respuesta"], fuentes_citadas(f),
                         f["estado_citas"], "Sí" if f["abstencion_detectada"] else "No",
                         {True: "Sí", False: "No", None: "NA"}[f["contexto_con_relevante"]],
                         "Sí" if f["gold_id"] in doble else "No", None, None, None, None, None])
        n = hoja.max_row
        for col, ancho in enumerate(anchos, 1):
            hoja.column_dimensions[hoja.cell(row=1, column=col).column_letter].width = ancho
        for celda in hoja[1]:
            celda.font = Font(bold=True)
            celda.fill = amarillo if celda.column >= 14 else gris
            celda.alignment = Alignment(wrap_text=True, vertical="center")
        for fila in hoja.iter_rows(min_row=2, max_row=n):
            for celda in fila:
                celda.alignment = Alignment(wrap_text=True, vertical="top")
                if celda.column >= 14:
                    celda.fill = amarillo
        notas = DataValidation(type="list", formula1='"0,1,2,3"', allow_blank=True)
        notas_na = DataValidation(type="list", formula1='"0,1,2,3,NA"', allow_blank=True)
        si_no = DataValidation(type="list", formula1='"Sí,No"', allow_blank=True)
        for dv in (notas, notas_na, si_no):
            hoja.add_data_validation(dv)
        if n >= 2:
            notas.add(f"N2:N{n}")
            notas_na.add(f"O2:P{n}")
            si_no.add(f"Q2:Q{n}")
        hoja.freeze_panes = "E2"
        hoja.auto_filter.ref = hoja.dimensions
    ruta.parent.mkdir(parents=True, exist_ok=True)
    libro.save(ruta)
    return {"filas_evaluador_1": len(ultimas),
            "filas_evaluador_2": sum(1 for f in ultimas if f["gold_id"] in doble),
            "preguntas_doble": sorted(doble)}


# ---------------------------------------------------------------- calificaciones
CRITERIOS = ["Corrección (0-3)", "Fidelidad (0-3)", "Citas (0-3)", "Abstención correcta"]
ESCALA = [0, 1, 2, 3]


def kappa(a: list, b: list, ponderado: bool = False, categorias: list | None = None) -> float | None:
    """Kappa de Cohen; `ponderado` = cuadrático, para escalas ordinales.

    `categorias` fija la escala completa (0-3): así el peso entre 0 y 3 es el máximo aunque
    nadie haya usado el 1 o el 2. Sin ella se usan las categorías observadas (como sklearn).
    """
    pares = [(x, y) for x, y in zip(a, b) if x is not None and y is not None]
    if len(pares) < 2:
        return None
    observadas = sorted({v for p in pares for v in p}, key=str)
    categorias = list(categorias) if categorias else observadas
    if len(categorias) == 1:
        return 1.0
    idx = {c: i for i, c in enumerate(categorias)}
    m = len(categorias)
    peso = (lambda i, j: ((i - j) / (m - 1)) ** 2) if ponderado else (lambda i, j: 0.0 if i == j else 1.0)
    n = len(pares)
    obs = sum(peso(idx[x], idx[y]) for x, y in pares) / n
    pa = [sum(1 for x, _ in pares if x == c) / n for c in categorias]
    pb = [sum(1 for _, y in pares if y == c) / n for c in categorias]
    esp = sum(pa[i] * pb[j] * peso(i, j) for i in range(m) for j in range(m))
    return None if esp == 0 else 1 - obs / esp


def _normalizar_nota(v):
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return None
    s = str(v).strip()
    if s in {"", "NA", "N/A", "nan"}:
        return None
    if s.lower() in {"sí", "si"}:
        return 1
    if s.lower() == "no":
        return 0
    try:
        return int(float(s))
    except ValueError:
        return None


def resumir_calificaciones(ruta: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    e1 = pd.read_excel(ruta, sheet_name="Evaluador 1")
    e2 = pd.read_excel(ruta, sheet_name="Evaluador 2")
    for df in (e1, e2):
        for c in CRITERIOS:
            df[c] = df[c].map(_normalizar_nota)
    calificadas = e1.dropna(subset=["Corrección (0-3)"])
    promedios = []
    for conjunto in CONJUNTOS:
        parte = calificadas if conjunto == "todos" else calificadas[calificadas["split"] == conjunto]
        for modelo, g in parte.groupby("modelo"):
            fila = {"conjunto": conjunto, "modelo": modelo, "n_calificadas": len(g)}
            for c in CRITERIOS:
                fila[c] = g[c].mean()
            fila["correctas (=3)"] = (g["Corrección (0-3)"] == 3).mean()
            promedios.append(fila)
    clave = ["gold_id", "modelo"]
    unidas = e1[clave + CRITERIOS].merge(e2[clave + CRITERIOS], on=clave, suffixes=(" E1", " E2"))
    acuerdo = []
    for c in CRITERIOS:
        a, b = unidas[f"{c} E1"].tolist(), unidas[f"{c} E2"].tolist()
        pares = [(x, y) for x, y in zip(a, b) if x is not None and y is not None and not pd.isna(x) and not pd.isna(y)]
        acuerdo.append({
            "criterio": c, "pares": len(pares),
            "acuerdo_exacto": (sum(1 for x, y in pares if x == y) / len(pares)) if pares else None,
            "kappa": kappa([x for x, _ in pares], [y for _, y in pares]),
            "kappa_ponderado": None if c == "Abstención correcta" else
            kappa([x for x, _ in pares], [y for _, y in pares], ponderado=True, categorias=ESCALA),
        })
    return pd.DataFrame(promedios).round(3), pd.DataFrame(acuerdo).round(3)


# ------------------------------------------------- ablaciones de la generación
def leer_notas(ruta: Path, hoja: str = "Evaluador 1") -> pd.DataFrame:
    df = pd.read_excel(ruta, sheet_name=hoja)
    for c in CRITERIOS:
        df[c] = df[c].map(_normalizar_nota)
    return df


def separar_ablacion(base: list[dict], ablacion: list[dict]) -> tuple[list[dict], list[dict]]:
    """Con temperatura 0 y semilla fija, mismo contexto => misma respuesta.

    Devuelve (reutilizables, por_calificar): una respuesta de la ablación se reutiliza si su
    contexto (huella) y su texto son idénticos a los de la corrida base para la misma
    pregunta y el mismo modelo; si no, hay que calificarla.
    """
    b = {k: f for k, f in ultima_por_clave(base).items() if not f.get("error")}
    reutilizables, por_calificar = [], []
    for clave, f in sorted(ultima_por_clave(ablacion).items()):
        if f.get("error"):
            continue
        previa = b.get(clave)
        igual = (previa is not None and (previa.get("contexto") or {}).get("huella") == (f.get("contexto") or {}).get("huella")
                 and (previa.get("respuesta") or "").strip() == (f.get("respuesta") or "").strip())
        (reutilizables if igual else por_calificar).append(f)
    return reutilizables, por_calificar


def metricas_humanas(df: pd.DataFrame) -> dict:
    """Las cuatro medidas de la regla de decisión sobre un conjunto de filas calificadas."""
    sin = df[df["tipo"] == "sin_respuesta"]
    return {
        "n": len(df),
        "exactitud": (df["Corrección (0-3)"] == 3).mean() if len(df) else None,
        "correccion_media": df["Corrección (0-3)"].mean(),
        "fidelidad_media": df["Fidelidad (0-3)"].mean(),
        "citas_media": df["Citas (0-3)"].mean(),
        "abstencion_correcta_sin_respuesta": sin["Abstención correcta"].mean() if len(sin) else None,
    }


def calificaciones_ablacion(base_planilla: Path, planilla_ablacion: Path, reutilizadas: Path) -> pd.DataFrame:
    base = leer_notas(base_planilla)
    nuevas = leer_notas(planilla_ablacion) if planilla_ablacion.is_file() else base.iloc[0:0]
    reuso = pd.read_csv(reutilizadas) if reutilizadas.is_file() else pd.DataFrame(columns=["gold_id", "modelo"])
    heredadas = base.merge(reuso[["gold_id", "modelo"]], on=["gold_id", "modelo"])
    pendientes = nuevas[nuevas["Corrección (0-3)"].isna()]
    if len(pendientes):
        print(f"  Aviso: {len(pendientes)} respuestas de la ablación aún no tienen nota; se excluyen.")
    todas = pd.concat([heredadas, nuevas.dropna(subset=["Corrección (0-3)"])], ignore_index=True)
    filas = []
    for conjunto in CONJUNTOS:
        parte = todas if conjunto == "todos" else todas[todas["split"] == conjunto]
        for modelo, g in parte.groupby("modelo"):
            filas.append({"conjunto": conjunto, "modelo": modelo, **metricas_humanas(g),
                          "respuestas_nuevas": int(g.index.isin(todas.index[len(heredadas):]).sum())})
    return pd.DataFrame(filas).round(3)


# --------------------------------------------------------------------------- CLI
def main(argv=None, sesion=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("paso", choices=["recuperacion", "generacion", "planilla", "resumen", "calificaciones", "todo",
                                     "planilla-ablacion", "calificaciones-ablacion"])
    p.add_argument("--gold", type=Path, default=GOLD_POR_DEFECTO)
    p.add_argument("--salida", type=Path, default=SALIDA_POR_DEFECTO)
    p.add_argument("--api", default=API_POR_DEFECTO)
    p.add_argument("--conjunto", choices=CONJUNTOS, default="todos", help="Preguntas a evaluar (split del gold)")
    p.add_argument("--modelos", nargs="*", help="Por defecto, los del experimento (LLM_MODELS_EXPERIMENT)")
    p.add_argument("--k-max", type=int, default=10, help="Profundidad de la recuperación evaluada (1-15)")
    p.add_argument("--estrategia", choices=["diversificada", "convencional"],
                   help="Ablación de recuperación; por defecto, la de la API (diversificada)")
    p.add_argument("--fraccion-doble", type=float, default=0.3, help="Submuestra para el segundo evaluador")
    p.add_argument("--planilla", type=Path, help="Planilla calificada (paso calificaciones)")
    p.add_argument("--etiqueta", help="Nombre de la condición de ablación (archivos *_<etiqueta>*)")
    p.add_argument("--k", type=int, help="Generación: k de la API para esta condición (ablación)")
    p.add_argument("--local", action="store_true",
                   help="Sin servidor HTTP: usa la API en este proceso, con la colección y el encoder del entorno "
                        "(MILVUS_COLLECTION, ENCODER_CONFIG). Para ablaciones de chunk, encoder y muestra")
    a = p.parse_args(argv)
    if not 1 <= a.k_max <= 15:
        sys.exit("--k-max debe estar entre 1 y 15 (límite de la API).")

    cobertura = None
    if a.local and sesion is None:
        from fastapi.testclient import TestClient

        import app.api
        sesion, a.api = TestClient(app.api.app), ""
    if a.local:
        cobertura = cobertura_local()
    api = Api(a.api, sesion)
    preguntas = leer_gold(a.gold, a.conjunto)
    pasos = ["recuperacion", "generacion", "planilla", "resumen"] if a.paso == "todo" else [a.paso]

    if any(x in pasos for x in ("recuperacion", "generacion")):
        try:
            salud = api.get("/health", timeout=60)
        except (ErrorApi, requests.RequestException) as e:
            sys.exit(f"La API no responde en {a.api}: {e}")
        registros = (salud.get("componentes", {}).get("milvus") or {}).get("registros")
        print(f"API {a.api} · Milvus con {registros} registros · gold {a.gold} ({len(preguntas)} preguntas, "
              f"conjunto '{a.conjunto}')", flush=True)

    if "recuperacion" in pasos:
        nombre = nombre_recuperacion(a.estrategia, a.etiqueta)
        guardar_configuracion(api, a.gold, a.salida, {"paso": nombre, "k_max": a.k_max,
                                                       "estrategia": a.estrategia or "por defecto",
                                                       "etiqueta": a.etiqueta, "conjunto": a.conjunto})
        print(f"\nRecuperación (top-{a.k_max}, estrategia {a.estrategia or 'por defecto'}"
              f"{', condición ' + a.etiqueta if a.etiqueta else ''}):", flush=True)
        df = correr_recuperacion(api, preguntas, a.k_max, a.salida, a.estrategia, a.etiqueta, cobertura)
        if cobertura is not None:
            total = int(df["n_relevantes"].sum())
            print(f"Cobertura del gold en la colección: {int(df['relevantes_en_coleccion'].sum())} de {total} "
                  "documentos relevantes indexados.")
        res = resumen_recuperacion(df)
        res.to_csv(a.salida / f"resumen_{nombre}.csv", index=False, encoding="utf-8")
        print("\n" + res.to_string(index=False))

    if "generacion" in pasos:
        modelos = a.modelos or api.get("/config")["modelos_experimento"]
        paso = "generacion" if not a.etiqueta else f"generacion_{a.etiqueta}"
        guardar_configuracion(api, a.gold, a.salida, {"paso": paso, "modelos": modelos, "k": a.k,
                                                       "estrategia": a.estrategia, "conjunto": a.conjunto})
        correr_generacion(api, preguntas, modelos, a.salida, a.etiqueta, a.k, a.estrategia)

    if "planilla" in pasos:
        gold_completo = leer_gold(a.gold, "todos")
        filas = [f for f in leer_jsonl(a.salida / "generacion_crudo.jsonl")
                 if a.conjunto == "todos" or f["split"] == a.conjunto]
        info = construir_planilla(gold_completo, filas, a.salida / "planilla_evaluacion.xlsx", a.fraccion_doble)
        print(f"\nPlanilla: {a.salida / 'planilla_evaluacion.xlsx'} · {info['filas_evaluador_1']} filas para el "
              f"evaluador 1, {info['filas_evaluador_2']} para el evaluador 2 "
              f"({len(info['preguntas_doble'])} preguntas en doble evaluación).")

    if "resumen" in pasos:
        filas = leer_jsonl(a.salida / nombre_generacion(a.etiqueta))
        if not filas:
            sys.exit("No hay resultados de generación todavía.")
        res = resumen_generacion(filas)
        sufijo = f"_{a.etiqueta}" if a.etiqueta else ""
        res.to_csv(a.salida / f"resumen_generacion{sufijo}.csv", index=False, encoding="utf-8")
        with pd.option_context("display.width", 250, "display.max_columns", 30):
            print("\nGeneración (detector automático; la calificación humana es la referencia):")
            print(res.to_string(index=False))

    if a.paso == "planilla-ablacion":
        if not a.etiqueta:
            sys.exit("Indique --etiqueta (la condición de la ablación).")
        base = leer_jsonl(a.salida / "generacion_crudo.jsonl")
        abl = leer_jsonl(a.salida / nombre_generacion(a.etiqueta))
        reutilizables, por_calificar = separar_ablacion(base, abl)
        pd.DataFrame([{"gold_id": f["gold_id"], "modelo": f["modelo"]} for f in reutilizables],
                     columns=["gold_id", "modelo"]).to_csv(a.salida / f"reutilizadas_{a.etiqueta}.csv", index=False)
        ruta = a.salida / f"planilla_{a.etiqueta}.xlsx"
        if por_calificar:
            construir_planilla(leer_gold(a.gold, "todos"), por_calificar, ruta, 0)
        print(f"Ablación '{a.etiqueta}': {len(reutilizables)} respuestas idénticas a la base (heredan su nota) y "
              f"{len(por_calificar)} por calificar" + (f" en {ruta}" if por_calificar else "") + ".")

    if a.paso == "calificaciones-ablacion":
        if not a.etiqueta:
            sys.exit("Indique --etiqueta (la condición de la ablación).")
        res = calificaciones_ablacion(a.planilla or a.salida / "planilla_evaluacion.xlsx",
                                      a.salida / f"planilla_{a.etiqueta}.xlsx",
                                      a.salida / f"reutilizadas_{a.etiqueta}.csv")
        res.to_csv(a.salida / f"calificaciones_{a.etiqueta}.csv", index=False, encoding="utf-8")
        print(f"\nCalificación humana, condición '{a.etiqueta}':\n" + res.to_string(index=False))

    if a.paso == "calificaciones":
        ruta = a.planilla or a.salida / "planilla_evaluacion.xlsx"
        if not ruta.is_file():
            sys.exit(f"No existe {ruta}.")
        promedios, acuerdo = resumir_calificaciones(ruta)
        promedios.to_csv(a.salida / "calificaciones_por_modelo.csv", index=False, encoding="utf-8")
        acuerdo.to_csv(a.salida / "acuerdo_evaluadores.csv", index=False, encoding="utf-8")
        print("\nCalificación humana por modelo (evaluador 1):\n" + promedios.to_string(index=False))
        print("\nAcuerdo entre evaluadores:\n" + acuerdo.to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
