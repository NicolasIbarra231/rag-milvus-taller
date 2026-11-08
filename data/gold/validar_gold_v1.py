import json
import re
import sys
from collections import Counter, defaultdict

import pandas as pd

carpeta, ruta_vectores = sys.argv[1], sys.argv[2]
v = pd.read_parquet(ruta_vectores)
norm = lambda s: " ".join(str(s).split())
texto_chunk = {c: norm(t) for c, t in zip(v["chunk_id"], v["text"])}
doc_de_chunk = dict(zip(v["chunk_id"], v["document_id"]))
docs_corpus = set(v["document_id"])
procedencia = {c: str(p) for c, p in zip(v["chunk_id"], v["provenance"])}

csv = pd.read_csv(f"{carpeta}/gold_v1.csv", encoding="utf-8")
jsonl = [json.loads(l) for l in open(f"{carpeta}/gold_v1.jsonl", encoding="utf-8")]
problemas = []
aviso = lambda m: problemas.append(m)

if len(csv) != len(jsonl):
    aviso(f"CSV ({len(csv)}) y JSONL ({len(jsonl)}) tienen distinto número de filas")
if list(csv["gold_id"]) != [r["gold_id"] for r in jsonl]:
    aviso("El orden o los IDs no coinciden entre CSV y JSONL")
if csv["gold_id"].duplicated().any():
    aviso(f"IDs duplicados: {csv.loc[csv.gold_id.duplicated(), 'gold_id'].tolist()}")

for col in ["pregunta", "respuesta_ref", "criterio"]:
    vacios = csv[csv[col].isna() | csv[col].astype(str).str.strip().isin(["", "nan", "None"])]
    if len(vacios):
        aviso(f"{col} vacío en {vacios.gold_id.tolist()}")
    marcadores = csv[csv[col].astype(str).str.contains(r"\(sin cambios\)|\(nueva|antes:|nan$", regex=True)]
    if len(marcadores):
        aviso(f"{col} con texto de la hoja de revisión en {marcadores.gold_id.tolist()}")

if csv["pregunta"].str.lower().duplicated().any():
    aviso(f"Preguntas repetidas: {csv.loc[csv.pregunta.str.lower().duplicated(), 'gold_id'].tolist()}")

for r in jsonl:
    gid, tipo, rel = r["gold_id"], r["tipo"], r["relevantes"]
    if tipo not in {"directa", "temporal", "comparativa", "ambigua", "sin_respuesta"}:
        aviso(f"{gid}: tipo desconocido {tipo}")
    if tipo == "sin_respuesta" and rel:
        aviso(f"{gid}: sin_respuesta con documentos relevantes")
    if tipo != "sin_respuesta" and not rel:
        aviso(f"{gid}: sin documentos relevantes")
    for doc, grado in rel.items():
        if doc not in docs_corpus:
            aviso(f"{gid}: documento relevante inexistente {doc}")
        if grado not in (1, 2):
            aviso(f"{gid}: grado inválido {grado}")
    if tipo in {"directa", "temporal"} and 2 not in rel.values():
        aviso(f"{gid}: directa/temporal sin ningún documento de grado 2")
    if tipo == "comparativa" and sum(1 for g in rel.values() if g == 2) < 2:
        aviso(f"{gid}: comparativa con menos de 2 documentos de grado 2")
    if tipo == "ambigua" and len(rel) < 2:
        aviso(f"{gid}: ambigua con menos de 2 documentos")
    for d in r["documentos"]:
        c = d["chunk_id"]
        if c not in texto_chunk:
            aviso(f"{gid}: chunk inexistente {c}")
            continue
        if doc_de_chunk[c] != d["document_id"] and d["document_id"] not in procedencia[c]:
            aviso(f"{gid}: el chunk {c[:8]} no pertenece a {d['document_id']}")
        if tipo != "sin_respuesta" and d["document_id"] not in rel:
            aviso(f"{gid}: documento de evidencia {d['document_id'][-20:]} no está en relevantes")
        for e in d["evidencias"]:
            if norm(e) not in texto_chunk[c]:
                aviso(f"{gid}: evidencia NO textual en el chunk ({e[:60]}…)")
    p = r["pregunta"]
    if not p.endswith("?"):
        aviso(f"{gid}: la pregunta no termina en «?»")
    if r["idioma_pregunta"] == "es" and "¿" not in p:
        aviso(f"{gid}: pregunta en español sin «¿»")
    if r["idioma_pregunta"] == "en" and re.search(r"[¿áéíóúñ]", p):
        aviso(f"{gid}: pregunta marcada en inglés con caracteres del español")
    if p[:1].islower() or (p.startswith("¿") and p[1:2].islower()):
        aviso(f"{gid}: la pregunta empieza en minúscula")
    if "  " in p or " ," in p or " ?" in p:
        aviso(f"{gid}: espacios raros en la pregunta")

docs_por_split = defaultdict(set)
for r in jsonl:
    for d in r["documentos"]:
        docs_por_split[r["split"]].add(d["document_id"])
fuga = docs_por_split["dev"] & docs_por_split["test"]
if fuga:
    aviso(f"Fuga: {len(fuga)} documentos de evidencia aparecen en dev y en test")

print("=" * 70)
print("PROBLEMAS ENCONTRADOS:", len(problemas))
for m in problemas:
    print("  -", m)
print("=" * 70)
print("Total:", len(csv))
print("Split:", csv.split.value_counts().to_dict())
print("Tipo x split:\n", pd.crosstab(csv.tipo, csv.split, margins=True))
print("Colección:", Counter(c for c in csv.coleccion).most_common())
print("Idioma pregunta:", csv.idioma_pregunta.value_counts().to_dict())
print("Idioma documento:", csv.idioma_documento.value_counts().to_dict())
anios = Counter(f[:4] for fs in csv.fecha_documento for f in fs.split(" | "))
print("Años de los documentos de evidencia:", dict(sorted(anios.items())))
print("Origen:", csv.origen_borrador.value_counts().to_dict(), "| Versión:", csv.version_elegida.value_counts().to_dict())
