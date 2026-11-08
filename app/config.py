"""Configuración central del sistema RAG.

Todo parámetro que cambie entre experimentos sale de variables de entorno (.env),
con los valores por defecto del experimento principal del notebook (Fase 5):
encoder multilingual-e5-small, chunks struct256o32, recuperación diversificada
(30 candidatos -> k=5), prompt rag_v5, temperatura 0, semilla 42, num_ctx 4096,
num_predict 250 y presupuesto de contexto de 6.000 caracteres.

El *contrato de codificación* (encoder_config.json, generado en la Fase 3) se
carga aquí. La ingesta y la API lo leen del mismo archivo, de modo que preguntas
y documentos se codifican siempre igual.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path


def _env(nombre: str, defecto: str) -> str:
    valor = os.getenv(nombre)
    return defecto if valor is None or valor.strip() == "" else valor.strip()


def _env_int(nombre: str, defecto: int) -> int:
    return int(_env(nombre, str(defecto)))


def _env_float(nombre: str, defecto: float) -> float:
    return float(_env(nombre, str(defecto)))


# Contrato por defecto: idéntico al encoder_config.json del notebook (Fase 3).
CONTRATO_POR_DEFECTO = {
    "modelo": "intfloat/multilingual-e5-small",
    "revision": None,
    "dimension": 384,
    "max_tokens": 512,
    "prefijo_query": "query: ",
    "prefijo_passage": "passage: ",
    "normalizado": True,
    "metrica_milvus": "COSINE",
    "estrategia_chunks": "struct256o32",
}


@dataclass(frozen=True)
class Settings:
    # --- Milvus ---
    milvus_uri: str = field(default_factory=lambda: _env("MILVUS_URI", "http://milvus:19530"))
    milvus_token: str = field(default_factory=lambda: _env("MILVUS_TOKEN", ""))
    coleccion: str = field(default_factory=lambda: _env("MILVUS_COLLECTION", "rag_struct256_e5"))
    tipo_indice: str = field(default_factory=lambda: _env("MILVUS_INDEX_TYPE", "AUTOINDEX"))

    # --- Encoder / contrato ---
    ruta_contrato: str = field(default_factory=lambda: _env("ENCODER_CONFIG", "data/encoder_config.json"))
    dispositivo_encoder: str = field(default_factory=lambda: _env("EMBEDDING_DEVICE", "cpu"))

    # --- Ollama ---
    ollama_url: str = field(default_factory=lambda: _env("OLLAMA_URL", "http://ollama:11434"))
    modelo_por_defecto: str = field(default_factory=lambda: _env("LLM_MODEL", "qwen2.5:1.5b"))
    modelos_experimento: tuple[str, ...] = field(
        default_factory=lambda: tuple(
            m.strip() for m in _env("LLM_MODELS_EXPERIMENT", "qwen2.5:1.5b,llama3.2:1b,llama3.2:3b").split(",") if m.strip()
        )
    )
    ollama_timeout: float = field(default_factory=lambda: _env_float("OLLAMA_TIMEOUT_SECONDS", 300))
    ollama_keep_alive: str = field(default_factory=lambda: _env("OLLAMA_KEEP_ALIVE", "10m"))

    # --- Parámetros RAG por defecto (experimento principal) ---
    prompt_por_defecto: str = field(default_factory=lambda: _env("PROMPT_VERSION", "rag_v5"))
    estrategia_por_defecto: str = field(default_factory=lambda: _env("RETRIEVAL_STRATEGY", "diversificada"))
    k_por_defecto: int = field(default_factory=lambda: _env_int("TOP_K", 5))
    candidatos: int = field(default_factory=lambda: _env_int("RETRIEVAL_CANDIDATES", 30))
    max_contexto_chars: int = field(default_factory=lambda: _env_int("MAX_CONTEXT_CHARS", 6000))
    temperatura: float = field(default_factory=lambda: _env_float("LLM_TEMPERATURE", 0.0))
    semilla: int = field(default_factory=lambda: _env_int("LLM_SEED", 42))
    num_ctx: int = field(default_factory=lambda: _env_int("LLM_NUM_CTX", 4096))
    num_predict: int = field(default_factory=lambda: _env_int("LLM_NUM_PREDICT", 250))

    # --- Trazabilidad ---
    ruta_logs: str = field(default_factory=lambda: _env("RUN_LOG_PATH", "reports/logs/rag_runs.jsonl"))
    nivel_log: str = field(default_factory=lambda: _env("LOG_LEVEL", "INFO"))


@lru_cache(maxsize=1)
def settings() -> Settings:
    return Settings()


@lru_cache(maxsize=1)
def contrato() -> dict:
    """Carga el contrato de codificación; si no existe, usa el del notebook."""
    ruta = Path(settings().ruta_contrato)
    datos = dict(CONTRATO_POR_DEFECTO)
    if ruta.is_file():
        datos.update(json.loads(ruta.read_text(encoding="utf-8")))
        datos["_origen"] = str(ruta)
    else:
        datos["_origen"] = "valores por defecto (no se encontró encoder_config.json)"
    # Permite forzar el encoder por entorno, solo para ablaciones (otra colección).
    if os.getenv("EMBEDDING_MODEL"):
        datos["modelo"] = os.environ["EMBEDDING_MODEL"]
    return datos
