"""Cliente mínimo de Ollama con errores tipados (para mensajes útiles en la UI)."""
from __future__ import annotations

import requests

from app.config import settings


class ErrorOllama(RuntimeError):
    codigo = "ollama_error"
    http = 502
    sugerencia = "Revise `docker compose logs ollama`."


class OllamaNoDisponible(ErrorOllama):
    codigo = "ollama_no_disponible"
    http = 503
    sugerencia = "Verifique que el contenedor esté arriba: `docker compose ps ollama`."


class ModeloNoDescargado(ErrorOllama):
    codigo = "modelo_no_descargado"
    http = 404

    def __init__(self, modelo: str):
        super().__init__(f"El modelo '{modelo}' no está descargado en Ollama.")
        self.sugerencia = f"Descárguelo con: docker compose exec ollama ollama pull {modelo}"


class OllamaTimeout(ErrorOllama):
    codigo = "ollama_timeout"
    http = 504
    sugerencia = ("El modelo tardó demasiado (arranque en frío o falta de memoria). "
                  "Reintente, use un modelo más pequeño o aumente OLLAMA_TIMEOUT_SECONDS.")


def _url(ruta: str) -> str:
    return settings().ollama_url.rstrip("/") + ruta


def _peticion(metodo: str, ruta: str, timeout: float, **kwargs) -> dict:
    try:
        r = requests.request(metodo, _url(ruta), timeout=timeout, **kwargs)
    except requests.Timeout as e:
        raise OllamaTimeout(f"Ollama no respondió en {timeout:.0f} s.") from e
    except requests.ConnectionError as e:
        raise OllamaNoDisponible(f"No hay conexión con Ollama en {settings().ollama_url}.") from e
    if r.status_code == 404:
        modelo = (kwargs.get("json") or {}).get("model", "?")
        raise ModeloNoDescargado(modelo)
    if r.status_code >= 400:
        try:
            detalle = r.json().get("error", r.text)
        except ValueError:
            detalle = r.text
        raise ErrorOllama(f"Ollama respondió {r.status_code}: {detalle}")
    return r.json() if r.content else {}


def version() -> str:
    return _peticion("GET", "/api/version", timeout=5).get("version", "?")


def listar_modelos() -> list[dict]:
    """Modelos descargados (equivalente a `ollama list`)."""
    modelos = _peticion("GET", "/api/tags", timeout=10).get("models", [])
    salida = []
    for m in modelos:
        d = m.get("details") or {}
        salida.append({
            "nombre": m.get("name") or m.get("model"),
            "tamano_gb": round((m.get("size") or 0) / 1e9, 2),
            "familia": d.get("family"),
            "parametros": d.get("parameter_size"),
            "cuantizacion": d.get("quantization_level"),
            "digest": (m.get("digest") or "")[:12],
        })
    return sorted(salida, key=lambda x: x["nombre"] or "")


def modelos_en_memoria() -> list[dict]:
    """Modelos cargados en RAM/VRAM ahora mismo (equivalente a `ollama ps`)."""
    modelos = _peticion("GET", "/api/ps", timeout=10).get("models", [])
    return [{
        "nombre": m.get("name"),
        "ram_gb": round((m.get("size") or 0) / 1e9, 2),
        "vram_gb": round((m.get("size_vram") or 0) / 1e9, 2),
        "expira": m.get("expires_at"),
    } for m in modelos]


def cargar_en_memoria(modelo: str) -> dict:
    """Precalienta el modelo (sin prompt) para separar arranque en frío de la medición."""
    r = _peticion("POST", "/api/generate", timeout=settings().ollama_timeout,
                  json={"model": modelo, "keep_alive": settings().ollama_keep_alive})
    return {"modelo": modelo, "carga_s": round((r.get("load_duration") or 0) / 1e9, 3)}


def descargar_de_memoria(modelo: str) -> dict:
    """Libera RAM/VRAM (keep_alive=0). No borra el modelo del disco."""
    _peticion("POST", "/api/generate", timeout=60, json={"model": modelo, "keep_alive": 0})
    return {"modelo": modelo, "estado": "descargado_de_memoria"}


def chat(modelo: str, system: str, user: str, opciones: dict) -> dict:
    s = settings()
    datos = _peticion("POST", "/api/chat", timeout=s.ollama_timeout, json={
        "model": modelo,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        "stream": False,
        "options": opciones,
        "keep_alive": s.ollama_keep_alive,
    })
    return datos
