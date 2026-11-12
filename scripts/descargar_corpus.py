"""Descarga el corpus del taller con la librería del curso (open_corpus_co_es).

Mismo procedimiento que la celda de carga del notebook (Parte 1): busca el corpus en el
catálogo, lo descarga, verifica el SHA-256 publicado y lo copia a data/.

data/ se monta como solo lectura en la API, así que se descarga dentro del contenedor y
luego se copia al equipo:

    docker compose exec api python -m scripts.descargar_corpus --salida /tmp/documentos_rag.parquet
    docker cp rag-api:/tmp/documentos_rag.parquet data/documentos_rag.parquet

El Parquet NO se publica en GitHub (.gitignore lo excluye).
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import shutil
import subprocess
import sys
from pathlib import Path

NOMBRES = ("dragons_rag", "dragons_rag_2026_10_19")   # el catálogo actual usa el primero


def sha256_archivo(ruta: Path) -> str:
    h = hashlib.sha256()
    with ruta.open("rb") as f:
        while bloque := f.read(1 << 20):
            h.update(bloque)
    return h.hexdigest()


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--salida", type=Path, default=Path("/tmp/documentos_rag.parquet"))
    a = p.parse_args(argv)

    try:
        occ = importlib.import_module("open_corpus_co_es.downloader")
    except ImportError:
        print("Instalando open_corpus_co_es (no viene en la imagen)…", flush=True)
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--upgrade", "--no-cache-dir",
                        "open_corpus_co_es"], check=True)
        occ = importlib.import_module("open_corpus_co_es.downloader")

    nombre = next((n for n in NOMBRES if n in occ.CATALOG), None)
    if nombre is None:
        parecidos = [n for n in occ.CATALOG if "rag" in n.lower() or "dragon" in n.lower()]
        sys.exit(f"El corpus no está en el catálogo. Entradas parecidas: {parecidos}")
    ficha = occ.CATALOG[nombre]
    print(f"Catálogo: {nombre} · versión {ficha.get('fecha_version')} · {ficha.get('documentos', 0):,} documentos",
          flush=True)
    occ.download_corpus(nombre)
    origen = sorted(Path(occ.get_corpus_path(nombre)).rglob("*.parquet"))[0]
    sha = sha256_archivo(origen)
    publicado = ficha.get("sha256")
    if publicado and sha != publicado:
        sys.exit(f"El SHA-256 descargado ({sha[:16]}…) NO coincide con el publicado ({publicado[:16]}…).")
    a.salida.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(origen, a.salida)
    print(f"Corpus copiado a {a.salida} · sha256 {sha[:16]}… "
          f"{'(coincide con el publicado)' if publicado else '(sin hash publicado para comparar)'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
