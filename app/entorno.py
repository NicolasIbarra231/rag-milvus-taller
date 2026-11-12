"""Registro de entorno: versiones y recursos visibles desde el proceso de la API.

Cada ejecución guarda en el log la versión de Ollama, el digest y la cuantización del
modelo usado y los recursos del contenedor (CPU y límite de memoria). El hardware del
equipo anfitrión (CPU, RAM total, GPU) no es visible desde Docker: lo registra
scripts/registrar_entorno.ps1 en reports/entorno/.
"""
from __future__ import annotations

import functools
import os
import platform
import sys
from pathlib import Path


def _leer(ruta: str) -> str | None:
    try:
        return Path(ruta).read_text(encoding="utf-8").strip()
    except OSError:
        return None


@functools.lru_cache(maxsize=1)
def proceso() -> dict:
    """Plataforma, versiones de librerías y límites de cgroup del contenedor."""
    versiones = {"python": sys.version.split()[0]}
    for paquete in ("torch", "sentence_transformers", "pymilvus", "fastapi"):
        try:
            versiones[paquete] = __import__(paquete).__version__
        except Exception:  # noqa: BLE001 - una librería ausente no debe romper el registro
            versiones[paquete] = None
    cpu_max = _leer("/sys/fs/cgroup/cpu.max")          # "400000 100000" = 4 CPU
    mem_max = _leer("/sys/fs/cgroup/memory.max")       # bytes o "max"
    cpus_limite = None
    if cpu_max and cpu_max.split()[0] != "max":
        cuota, periodo = cpu_max.split()[:2]
        cpus_limite = round(int(cuota) / int(periodo), 2)
    return {
        "plataforma": platform.platform(),
        "cpus_visibles": os.cpu_count(),
        "cpus_limite_contenedor": cpus_limite,
        "memoria_limite_contenedor_gb": (round(int(mem_max) / 1024 ** 3, 2)
                                         if mem_max and mem_max.isdigit() else None),
        "versiones": versiones,
    }


_MODELOS: dict[str, dict] = {}


def modelo(nombre: str) -> dict:
    """Versión de Ollama, digest, cuantización y tamaño del modelo.

    Se consulta una vez por modelo; si Ollama no responde, no se guarda en caché
    y se reintenta en la siguiente ejecución.
    """
    if nombre in _MODELOS:
        return _MODELOS[nombre]
    from app import llm

    info: dict = {"ollama_version": None}
    try:
        info["ollama_version"] = llm.version()
        for m in llm.listar_modelos():
            if m.get("nombre") == nombre:
                info.update({k: m.get(k) for k in ("digest", "cuantizacion", "parametros", "familia", "tamano_gb")})
                break
        _MODELOS[nombre] = info
    except Exception:  # noqa: BLE001 - el registro nunca debe tumbar una respuesta
        pass
    return info


def para_registro(nombre_modelo: str) -> dict:
    return {"modelo": modelo(nombre_modelo), "proceso": proceso()}
