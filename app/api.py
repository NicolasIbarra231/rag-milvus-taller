"""API FastAPI del asistente RAG (literal F).

Endpoints
---------
GET  /health          vivo + estado de Milvus, Ollama y encoder
GET  /health/live     solo "el proceso responde" (healthcheck de Docker)
GET  /config          parámetros por defecto, prompts y contrato del encoder
GET  /models          modelos descargados y en memoria; faltantes del experimento
POST /models/load     precalienta un modelo en RAM/VRAM
POST /models/unload   libera RAM/VRAM de un modelo (no lo borra del disco)
POST /retrieve        solo recuperación (sin LLM)
POST /ask             recuperación + generación + validación de citas

Todos los errores salen con la misma forma: {"error", "detalle", "sugerencia"}.
"""
from __future__ import annotations

import logging
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator

from app import llm, milvus_store, prompts
from app.config import contrato, settings
from app.encoder import ErrorContrato, modelo as cargar_encoder
from app.rag import preguntar, registrar_ejecucion
from app.retrieval import ESTRATEGIAS, recuperar

logging.basicConfig(level=settings().nivel_log,
                    format="%(asctime)s %(levelname)s %(name)s %(message)s")
log = logging.getLogger("api")


# ---------------------------------------------------------------- errores
class ErrorAPI(Exception):
    def __init__(self, http: int, codigo: str, detalle: str, sugerencia: str = ""):
        self.http, self.codigo, self.detalle, self.sugerencia = http, codigo, detalle, sugerencia


def _cuerpo_error(codigo: str, detalle: str, sugerencia: str = "") -> dict:
    return {"error": codigo, "detalle": detalle, "sugerencia": sugerencia}


@asynccontextmanager
async def ciclo_de_vida(_app: FastAPI):
    # Carga el encoder al arrancar para que la primera pregunta no pague la descarga.
    try:
        cargar_encoder()
        log.info("Encoder listo: %s", contrato()["modelo"])
    except Exception as e:  # noqa: BLE001 - /health lo reportará
        log.error("No se pudo cargar el encoder: %s", e)
    yield


app = FastAPI(title="Asistente RAG · Milvus + LLM locales", version="1.0.0", lifespan=ciclo_de_vida)


@app.exception_handler(ErrorAPI)
async def _manejar_api(_: Request, e: ErrorAPI):
    return JSONResponse(status_code=e.http, content=_cuerpo_error(e.codigo, e.detalle, e.sugerencia))


@app.exception_handler(llm.ErrorOllama)
async def _manejar_ollama(_: Request, e: llm.ErrorOllama):
    return JSONResponse(status_code=e.http, content=_cuerpo_error(e.codigo, str(e), e.sugerencia))


@app.exception_handler(milvus_store.ErrorMilvus)
async def _manejar_milvus(_: Request, e: milvus_store.ErrorMilvus):
    return JSONResponse(status_code=503, content=_cuerpo_error(
        "milvus_no_disponible", str(e), "Verifique `docker compose ps milvus` y `docker compose logs milvus`."))


@app.exception_handler(ErrorContrato)
async def _manejar_contrato(_: Request, e: ErrorContrato):
    return JSONResponse(status_code=500, content=_cuerpo_error("contrato_encoder", str(e)))


@app.exception_handler(ValueError)
async def _manejar_valor(_: Request, e: ValueError):
    return JSONResponse(status_code=422, content=_cuerpo_error("parametro_invalido", str(e)))


@app.middleware("http")
async def _registrar_peticion(request: Request, call_next):
    t0 = time.perf_counter()
    respuesta = await call_next(request)
    log.info("%s %s -> %s (%.3fs)", request.method, request.url.path, respuesta.status_code,
             time.perf_counter() - t0)
    return respuesta


# ---------------------------------------------------------------- modelos de datos
class Filtros(BaseModel):
    colecciones: list[str] | None = Field(default=None, description="bluesky_posts y/o url_crawls")
    idioma: str | None = Field(default=None, description="Código de dos letras: es, en, hu...")


class PeticionRetrieve(BaseModel):
    query: str = Field(min_length=3, max_length=1000)
    k: int = Field(default_factory=lambda: settings().k_por_defecto, ge=1, le=15)
    estrategia: str = Field(default_factory=lambda: settings().estrategia_por_defecto)
    filtros: Filtros | None = None

    @field_validator("query")
    @classmethod
    def _sin_vacios(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("La pregunta está vacía.")
        return v.strip()

    @field_validator("estrategia")
    @classmethod
    def _estrategia(cls, v: str) -> str:
        if v not in ESTRATEGIAS:
            raise ValueError(f"estrategia debe ser una de {ESTRATEGIAS}")
        return v


class PeticionAsk(PeticionRetrieve):
    model: str = Field(default_factory=lambda: settings().modelo_por_defecto, min_length=1)
    prompt_version: str = Field(default_factory=lambda: settings().prompt_por_defecto)
    temperature: float | None = Field(default=None, ge=0, le=1)
    num_predict: int | None = Field(default=None, ge=16, le=2048)

    @field_validator("prompt_version")
    @classmethod
    def _prompt(cls, v: str) -> str:
        prompts.obtener(v)
        return v


class PeticionModelo(BaseModel):
    model: str = Field(min_length=1)


def _filtro(p: PeticionRetrieve) -> str:
    if not p.filtros:
        return ""
    return milvus_store.construir_filtro(p.filtros.colecciones, p.filtros.idioma)


def _exigir_coleccion() -> None:
    info = milvus_store.estado()
    if not info["existe"] or info["registros"] == 0:
        raise ErrorAPI(409, "coleccion_vacia",
                       f"La colección '{info['coleccion']}' no existe o está vacía.",
                       "Ejecute la ingesta: docker compose exec api python -m scripts.ingest --from-vectors "
                       "data/vectores.parquet")


# ---------------------------------------------------------------- endpoints
@app.get("/health/live")
def vivo():
    return {"status": "ok"}


@app.get("/health")
def salud():
    componentes = {}
    try:
        componentes["milvus"] = {"ok": True, **milvus_store.estado()}
    except milvus_store.ErrorMilvus as e:
        componentes["milvus"] = {"ok": False, "detalle": str(e)}
    try:
        componentes["ollama"] = {"ok": True, "version": llm.version(),
                                 "modelos": len(llm.listar_modelos())}
    except llm.ErrorOllama as e:
        componentes["ollama"] = {"ok": False, "detalle": str(e)}
    try:
        cargar_encoder()
        componentes["encoder"] = {"ok": True, "modelo": contrato()["modelo"], "dimension": contrato()["dimension"]}
    except Exception as e:  # noqa: BLE001
        componentes["encoder"] = {"ok": False, "detalle": str(e)}
    listo = all(c["ok"] for c in componentes.values()) and componentes["milvus"].get("registros", 0) > 0
    return {"status": "ok" if listo else "degradado", "componentes": componentes,
            "fecha_utc": datetime.now(timezone.utc).isoformat()}


@app.get("/config")
def configuracion():
    s = settings()
    return {
        "defaults": {
            "model": s.modelo_por_defecto, "k": s.k_por_defecto, "estrategia": s.estrategia_por_defecto,
            "prompt_version": s.prompt_por_defecto, "candidatos": s.candidatos,
            "max_contexto_chars": s.max_contexto_chars, "temperature": s.temperatura,
            "seed": s.semilla, "num_ctx": s.num_ctx, "num_predict": s.num_predict,
        },
        "estrategias": list(ESTRATEGIAS),
        "prompts": {v: {"descripcion": p["descripcion"], "huella": prompts.huella_prompt(v)}
                    for v, p in prompts.PROMPTS.items()},
        "modelos_experimento": list(s.modelos_experimento),
        "colecciones": sorted(milvus_store.COLECCIONES_VALIDAS),
        "coleccion_milvus": s.coleccion,
        "contrato_encoder": contrato(),
    }


@app.get("/models")
def modelos():
    instalados = llm.listar_modelos()
    try:
        en_memoria = llm.modelos_en_memoria()
    except llm.ErrorOllama:
        en_memoria = []
    nombres = {m["nombre"] for m in instalados}
    return {
        "instalados": instalados,
        "en_memoria": en_memoria,
        "experimento": list(settings().modelos_experimento),
        "faltantes_experimento": [m for m in settings().modelos_experimento if m not in nombres],
    }


@app.post("/models/load")
def cargar_modelo(p: PeticionModelo):
    return llm.cargar_en_memoria(p.model)


@app.post("/models/unload")
def descargar_modelo(p: PeticionModelo):
    return llm.descargar_de_memoria(p.model)


@app.post("/retrieve")
def endpoint_retrieve(p: PeticionRetrieve):
    _exigir_coleccion()
    evidencias, metricas = recuperar(p.query, k=p.k, estrategia=p.estrategia, filtro=_filtro(p))
    for i, e in enumerate(evidencias, 1):
        e["referencia"] = i
    return {"query": p.query, "chunks": evidencias, "metricas": metricas}


@app.post("/ask")
def endpoint_ask(p: PeticionAsk):
    _exigir_coleccion()
    try:
        return preguntar(p.query, modelo=p.model, k=p.k, estrategia=p.estrategia,
                         prompt_version=p.prompt_version, filtro=_filtro(p),
                         temperatura=p.temperature, num_predict=p.num_predict)
    except Exception as e:
        # Los errores también se registran: alimentan la "tasa de errores" (literal 6).
        registrar_ejecucion({
            "run_id": uuid.uuid4().hex[:12], "fecha_utc": datetime.now(timezone.utc).isoformat(),
            "pregunta": p.query, "modelo": p.model, "error": type(e).__name__, "detalle": str(e),
            "parametros": {"k": p.k, "estrategia": p.estrategia, "prompt_version": p.prompt_version},
        })
        raise
