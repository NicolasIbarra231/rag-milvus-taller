"""Acceso a Milvus standalone: esquema explícito, upsert idempotente y búsqueda.

En el notebook (Colab) se usó Milvus Lite con creación rápida y campos
dinámicos. Aquí el esquema es explícito para que los campos, sus longitudes y
la métrica queden documentados y versionados junto con el código.
"""
from __future__ import annotations

import json
import re
from functools import lru_cache

from app.config import contrato, settings

CAMPOS_SALIDA = [
    "document_id", "text", "title", "url", "collection_name", "source", "language",
    "created_at", "chunk_index", "char_start", "char_end", "n_tokens", "n_sources",
    "content_hash", "estrategia", "provenance",
]

# Longitudes máximas (bytes UTF-8) de los VARCHAR.
LIMITES = {
    "id": 64, "document_id": 512, "text": 16384, "title": 2048, "url": 4096,
    "collection_name": 64, "source": 64, "language": 32, "created_at": 64,
    "content_hash": 64, "estrategia": 32,
}
COLECCIONES_VALIDAS = {"bluesky_posts", "url_crawls"}


class ErrorMilvus(RuntimeError):
    """Milvus no responde o la operación falló."""


@lru_cache(maxsize=1)
def cliente():
    from pymilvus import MilvusClient

    s = settings()
    try:
        return MilvusClient(uri=s.milvus_uri, token=s.milvus_token or None)
    except Exception as e:  # noqa: BLE001 - se reporta como error de servicio
        raise ErrorMilvus(f"No se pudo conectar a Milvus en {s.milvus_uri}: {e}") from e


def truncar(valor, limite: int) -> str:
    """Convierte a str y recorta por bytes UTF-8 sin partir caracteres."""
    if valor is None:
        return ""
    try:
        import math

        if isinstance(valor, float) and math.isnan(valor):
            return ""
    except TypeError:
        pass
    texto = str(valor)
    codificado = texto.encode("utf-8")
    if len(codificado) <= limite:
        return texto
    return codificado[:limite].decode("utf-8", errors="ignore")


def crear_esquema(dim: int):
    from pymilvus import DataType, MilvusClient

    esquema = MilvusClient.create_schema(auto_id=False, enable_dynamic_field=False)
    esquema.add_field("id", DataType.VARCHAR, is_primary=True, max_length=LIMITES["id"])
    esquema.add_field("vector", DataType.FLOAT_VECTOR, dim=dim)
    for campo in ["document_id", "text", "title", "url", "collection_name", "source",
                  "language", "created_at", "content_hash", "estrategia"]:
        esquema.add_field(campo, DataType.VARCHAR, max_length=LIMITES[campo])
    for campo in ["chunk_index", "char_start", "char_end", "n_tokens", "n_sources"]:
        esquema.add_field(campo, DataType.INT64)
    esquema.add_field("provenance", DataType.JSON)
    return esquema


def asegurar_coleccion(recrear: bool = False) -> bool:
    """Crea la colección si no existe. Devuelve True si la creó."""
    s, c = settings(), cliente()
    if recrear and c.has_collection(s.coleccion):
        c.drop_collection(s.coleccion)
    if c.has_collection(s.coleccion):
        return False
    dim = int(contrato()["dimension"])
    indices = c.prepare_index_params()
    indices.add_index(field_name="vector", index_type=s.tipo_indice, metric_type=contrato()["metrica_milvus"])
    c.create_collection(collection_name=s.coleccion, schema=crear_esquema(dim), index_params=indices)
    return True


def fila_a_registro(fila: dict, vector) -> dict:
    """Normaliza una fila (chunk + metadatos) al esquema de la colección."""
    prov = fila.get("provenance") or "[]"
    if isinstance(prov, str):
        prov = json.loads(prov)
    registro = {"id": truncar(fila["chunk_id"], LIMITES["id"]), "vector": [float(x) for x in vector]}
    for campo in ["document_id", "text", "title", "url", "collection_name", "source",
                  "language", "created_at", "content_hash", "estrategia"]:
        registro[campo] = truncar(fila.get(campo), LIMITES[campo])
    for campo in ["chunk_index", "char_start", "char_end", "n_tokens"]:
        registro[campo] = int(fila.get(campo) or 0)
    registro["n_sources"] = int(fila.get("n_sources") or len(prov) or 1)
    registro["provenance"] = prov
    return registro


def upsert(registros: list[dict]) -> int:
    """Upsert por clave primaria estable: reejecutar la ingesta no duplica."""
    if not registros:
        return 0
    res = cliente().upsert(collection_name=settings().coleccion, data=registros)
    return int(res.get("upsert_count", len(registros))) if isinstance(res, dict) else len(registros)


def contar() -> int:
    s, c = settings(), cliente()
    if not c.has_collection(s.coleccion):
        return 0
    try:
        filas = c.query(collection_name=s.coleccion, filter="", output_fields=["count(*)"])
        return int(filas[0]["count(*)"])
    except Exception:  # noqa: BLE001 - algunas versiones solo exponen stats
        return int(c.get_collection_stats(s.coleccion)["row_count"])


def estado() -> dict:
    """Resumen para /health: existencia, conteo y dimensión de la colección."""
    s, c = settings(), cliente()
    try:
        existe = c.has_collection(s.coleccion)
    except Exception as e:  # noqa: BLE001
        raise ErrorMilvus(f"Milvus no responde: {e}") from e
    info = {"coleccion": s.coleccion, "existe": existe, "registros": 0}
    if existe:
        info["registros"] = contar()
        for campo in c.describe_collection(s.coleccion).get("fields", []):
            if campo.get("name") == "vector":
                info["dimension"] = campo.get("params", {}).get("dim")
    return info


def construir_filtro(colecciones: list[str] | None = None, idioma: str | None = None) -> str:
    """Expresión de filtro segura (lista blanca), para no inyectar expresiones."""
    partes = []
    if colecciones:
        invalidas = set(colecciones) - COLECCIONES_VALIDAS
        if invalidas:
            raise ValueError(f"Colecciones no válidas: {sorted(invalidas)}")
        lista = ", ".join(f'"{x}"' for x in sorted(set(colecciones)))
        partes.append(f"collection_name in [{lista}]")
    if idioma:
        if not re.fullmatch(r"[a-z]{2}", idioma):
            raise ValueError("idioma debe ser un código de dos letras, p. ej. 'es' o 'en'")
        partes.append(f'language like "{idioma}%"')  # 'es' incluye es-co, es-mx...
    return " and ".join(partes)


def buscar(vector, limite: int, filtro: str = "") -> list[dict]:
    s = settings()
    try:
        resultados = cliente().search(
            collection_name=s.coleccion,
            data=[[float(x) for x in vector]],
            limit=limite,
            filter=filtro,
            output_fields=CAMPOS_SALIDA,
            search_params={"metric_type": contrato()["metrica_milvus"]},
        )
    except Exception as e:  # noqa: BLE001
        raise ErrorMilvus(f"Falló la búsqueda en Milvus: {e}") from e
    salida = []
    for hit in resultados[0]:
        entidad = dict(hit["entity"]) if "entity" in hit else {}
        salida.append({"chunk_id": hit["id"], "score": float(hit["distance"]), **entidad})
    return salida
