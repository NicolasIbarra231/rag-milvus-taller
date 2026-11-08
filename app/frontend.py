"""Interfaz web (Streamlit) del asistente RAG — literal F.

Muestra pregunta, LLM, respuesta, tiempos, evidencias y enlaces; maneja estados
de carga y errores, y lista dinámicamente los modelos descargados en Ollama.
La interfaz SOLO habla con la API: no toca Milvus ni Ollama directamente.
"""
from __future__ import annotations

import json
import os

import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000").rstrip("/")
TIMEOUT_ASK = float(os.getenv("FRONTEND_TIMEOUT_SECONDS", "330"))

st.set_page_config(page_title="Asistente RAG · Milvus", page_icon="🔎", layout="wide")


# ------------------------------------------------------------------ cliente API
class FalloAPI(Exception):
    def __init__(self, titulo: str, detalle: str = "", sugerencia: str = ""):
        super().__init__(titulo)
        self.titulo, self.detalle, self.sugerencia = titulo, detalle, sugerencia


def llamar(metodo: str, ruta: str, timeout: float = 15, **kwargs) -> dict:
    try:
        r = requests.request(metodo, API_URL + ruta, timeout=timeout, **kwargs)
    except requests.ConnectionError:
        raise FalloAPI("No hay conexión con la API", f"No se pudo contactar {API_URL}.",
                       "Revise `docker compose ps api` y `docker compose logs api`.")
    except requests.Timeout:
        raise FalloAPI("La API tardó demasiado", f"Se superaron {timeout:.0f} s de espera.",
                       "El modelo puede estar cargándose en frío; reintente o use un modelo más pequeño.")
    if r.status_code >= 400:
        try:
            cuerpo = r.json()
        except ValueError:
            cuerpo = {"detalle": r.text}
        if isinstance(cuerpo.get("detail"), list):  # errores de validación de FastAPI
            detalle = "; ".join(f"{'.'.join(map(str, d.get('loc', [])))}: {d.get('msg')}" for d in cuerpo["detail"])
            raise FalloAPI("Parámetros inválidos", detalle)
        raise FalloAPI(cuerpo.get("error", f"Error {r.status_code}"), cuerpo.get("detalle", ""),
                       cuerpo.get("sugerencia", ""))
    return r.json()


def mostrar_fallo(e: FalloAPI) -> None:
    st.error(f"**{e.titulo}**\n\n{e.detalle}")
    if e.sugerencia:
        st.info(f"💡 {e.sugerencia}")


@st.cache_data(ttl=15, show_spinner=False)
def obtener(ruta: str) -> dict:
    return llamar("GET", ruta)


def obtener_seguro(ruta: str) -> tuple[dict | None, FalloAPI | None]:
    try:
        return obtener(ruta), None
    except FalloAPI as e:
        return None, e


# ------------------------------------------------------------------ render
ESTADO_CITAS = {
    "referencias_existentes": ("success", "Citas presentes y existentes (verifique que cada fuente respalde su afirmación)"),
    "abstencion": ("info", "El modelo se abstuvo: no encontró evidencia suficiente"),
    "respuesta_sin_citas": ("warning", "Respuesta sin citas [n]: no es verificable"),
    "referencias_inexistentes": ("error", "La respuesta cita fuentes que no existen"),
}


def mostrar_evidencias(evidencias: list[dict]) -> None:
    if not evidencias:
        st.caption("No se recuperaron evidencias.")
        return
    for ev in evidencias:
        marca = "✅ citada" if ev.get("citada") else "· no citada"
        titulo = ev.get("title") or ev.get("document_id")
        with st.expander(f"[{ev.get('referencia')}] {marca} — {titulo[:110]} · score {ev.get('score', 0):.3f}"):
            if ev.get("url"):
                st.markdown(f"🔗 [{ev['url'][:100]}]({ev['url']})")
            st.caption(
                f"Colección: `{ev.get('collection_name')}` · Idioma: `{ev.get('language')}` · "
                f"Fecha: `{ev.get('created_at')}` · Documento: `{ev.get('document_id')}` · "
                f"Offsets: {ev.get('char_start')}–{ev.get('char_end')} · Tokens: {ev.get('n_tokens')}"
            )
            st.write(ev.get("text", ""))
            prov = ev.get("provenance") or []
            if isinstance(prov, list) and len(prov) > 1:
                st.caption(f"Este fragmento aparece en {len(prov)} documentos (deduplicado):")
                for p in prov[1:6]:
                    st.markdown(f"- [{p.get('document_id')}]({p.get('url')})")


def mostrar_resultado(res: dict, compacto: bool = False) -> None:
    t, tok = res["tiempos"], res["tokens"]
    if not compacto:
        st.markdown(f"#### ❓ {res['pregunta']}")
    st.markdown(f"**🤖 LLM:** `{res['modelo']}` · prompt `{res['parametros']['prompt_version']}` · "
                f"k={res['parametros']['k']} · {res['parametros']['estrategia']}")
    c = st.columns(4 if not compacto else 2)
    c[0].metric("Tiempo total", f"{t['total_s']:.2f} s")
    c[1].metric("Generación", f"{t['generacion_s']:.2f} s")
    if not compacto:
        c[2].metric("Recuperación", f"{t['recuperacion_s']:.3f} s")
        c[3].metric("Tokens/s", tok["por_segundo"] if tok["por_segundo"] else "—")

    tipo, texto = ESTADO_CITAS.get(res["citas"]["estado"], ("info", res["citas"]["estado"]))
    getattr(st, tipo)(texto)
    st.markdown("**Respuesta**")
    st.markdown(res["respuesta"] or "_(respuesta vacía)_")
    for adv in res.get("advertencias", []):
        st.warning(adv)

    if not compacto:
        st.markdown(f"**Evidencias** ({len(res['evidencias'])} fragmentos, "
                    f"{res['contexto']['documentos_unicos']} documentos únicos)")
        mostrar_evidencias(res["evidencias"])
        with st.expander("Trazabilidad de la ejecución"):
            st.json({k: res[k] for k in ("run_id", "parametros", "tiempos", "tokens", "contexto", "encoder", "coleccion")})


# ------------------------------------------------------------------ barra lateral
st.title("🔎 Asistente RAG · Milvus + LLM locales")
st.caption("Corpus de prensa digital y Bluesky. Las respuestas deben contrastarse con las fuentes originales.")

with st.sidebar:
    st.header("Estado del sistema")
    if st.button("↻ Actualizar estado y modelos", use_container_width=True):
        obtener.clear()
    salud, err_salud = obtener_seguro("/health")
    if err_salud:
        mostrar_fallo(err_salud)
        st.stop()
    for nombre, comp in salud["componentes"].items():
        if comp["ok"]:
            extra = f"{comp.get('registros', '')} chunks" if nombre == "milvus" else comp.get("version", comp.get("modelo", ""))
            st.markdown(f"✅ **{nombre}** {extra}")
        else:
            st.markdown(f"⚠️ **{nombre}**: {comp.get('detalle', '')}")
    if salud["componentes"].get("milvus", {}).get("registros", 0) == 0:
        st.warning("La colección está vacía. Ejecute la ingesta (ver README).")

    config, err_config = obtener_seguro("/config")
    if err_config:
        mostrar_fallo(err_config)
        st.stop()
    defaults = config["defaults"]

    st.header("Modelo")
    modelos, err_modelos = obtener_seguro("/models")
    if err_modelos:
        mostrar_fallo(err_modelos)
        st.stop()
    nombres = [m["nombre"] for m in modelos["instalados"]]
    if not nombres:
        st.error("No hay modelos descargados en Ollama.")
        st.code("docker compose exec ollama ollama pull qwen2.5:1.5b", language="bash")
        st.stop()
    experimento = set(config["modelos_experimento"])
    indice = nombres.index(defaults["model"]) if defaults["model"] in nombres else 0
    modelo = st.selectbox("LLM local", nombres, index=indice,
                          format_func=lambda n: f"★ {n}" if n in experimento else n,
                          help="Lista dinámica: modelos descargados en Ollama. ★ = usado en el experimento.")
    info = next(m for m in modelos["instalados"] if m["nombre"] == modelo)
    st.caption(f"{info['parametros']} · {info['cuantizacion']} · {info['tamano_gb']} GB · digest {info['digest']}")
    if modelos["faltantes_experimento"]:
        with st.expander("Modelos del experimento sin descargar"):
            for m in modelos["faltantes_experimento"]:
                st.code(f"docker compose exec ollama ollama pull {m}", language="bash")

    col1, col2 = st.columns(2)
    if col1.button("Precalentar", use_container_width=True, help="Carga el modelo en memoria (evita medir el arranque en frío)."):
        with st.spinner(f"Cargando {modelo}…"):
            try:
                r = llamar("POST", "/models/load", timeout=TIMEOUT_ASK, json={"model": modelo})
                st.success(f"Cargado en {r['carga_s']} s")
            except FalloAPI as e:
                mostrar_fallo(e)
        obtener.clear()
    if col2.button("Liberar", use_container_width=True, help="Saca el modelo de RAM/VRAM sin borrarlo del disco."):
        try:
            llamar("POST", "/models/unload", timeout=60, json={"model": modelo})
            st.success("Memoria liberada")
        except FalloAPI as e:
            mostrar_fallo(e)
        obtener.clear()
    if modelos and modelos["en_memoria"]:
        st.caption("En memoria: " + ", ".join(f"{m['nombre']} ({m['vram_gb']} GB VRAM)" for m in modelos["en_memoria"]))

    st.header("Recuperación")
    k = st.slider("Top-k (fragmentos)", 1, 15, int(defaults["k"]))
    estrategia = st.radio("Estrategia", config["estrategias"],
                          index=config["estrategias"].index(defaults["estrategia"]),
                          help="Diversificada: máximo un fragmento por documento (experimento principal).")
    with st.expander("Filtros"):
        colecciones = st.multiselect("Colecciones", config["colecciones"], default=[])
        idioma = st.text_input("Idioma (prefijo de 2 letras)", value="", max_chars=2,
                               help="'es' incluye es-co, es-mx, etc.").strip().lower()
    with st.expander("Generación avanzada"):
        versiones = list(config["prompts"])
        prompt_version = st.selectbox("Versión de prompt", versiones,
                                      index=versiones.index(defaults["prompt_version"]),
                                      format_func=lambda v: f"{v} — {config['prompts'][v]['descripcion']}")
        temperatura = st.slider("Temperatura", 0.0, 1.0, float(defaults["temperature"]), 0.05)
        num_predict = st.number_input("Máx. tokens de salida", 16, 2048, int(defaults["num_predict"]), 10)
    if (k, estrategia, prompt_version, temperatura, num_predict) != (
            defaults["k"], defaults["estrategia"], defaults["prompt_version"], defaults["temperature"], defaults["num_predict"]):
        st.info("Configuración distinta a la del experimento principal.")


def cuerpo(pregunta: str, modelo_: str) -> dict:
    filtros = {}
    if colecciones:
        filtros["colecciones"] = colecciones
    if idioma:
        filtros["idioma"] = idioma
    return {"query": pregunta, "model": modelo_, "k": k, "estrategia": estrategia,
            "prompt_version": prompt_version, "temperature": temperatura,
            "num_predict": int(num_predict), "filtros": filtros or None}


# ------------------------------------------------------------------ pestañas
st.session_state.setdefault("historial", [])
tab_preguntar, tab_comparar, tab_recuperar = st.tabs(["💬 Preguntar", "⚖️ Comparar modelos", "📚 Solo recuperación"])

with tab_preguntar:
    with st.form("form_preguntar"):
        pregunta = st.text_area("Pregunta", placeholder="¿Qué impactos del cambio climático se mencionan en las noticias?",
                                height=90)
        enviar = st.form_submit_button("Preguntar", type="primary")
    fallo_actual = False
    if enviar:
        if len(pregunta.strip()) < 3:
            st.warning("Escriba una pregunta de al menos 3 caracteres.")
        else:
            with st.status(f"Consultando con {modelo}…", expanded=True) as estado:
                st.write("1. Codificando la pregunta y buscando en Milvus…")
                st.write(f"2. Generando la respuesta con **{modelo}** (puede tardar si el modelo arranca en frío)…")
                try:
                    res = llamar("POST", "/ask", timeout=TIMEOUT_ASK, json=cuerpo(pregunta.strip(), modelo))
                    st.session_state["historial"].insert(0, res)
                    estado.update(label=f"Listo en {res['tiempos']['total_s']:.2f} s", state="complete", expanded=False)
                except FalloAPI as e:
                    estado.update(label="La consulta falló", state="error", expanded=True)
                    mostrar_fallo(e)
                    fallo_actual = True
    if st.session_state["historial"]:
        if fallo_actual:
            st.caption("Abajo: la última respuesta exitosa de la sesión (no corresponde a la consulta que falló).")
        mostrar_resultado(st.session_state["historial"][0])
        if len(st.session_state["historial"]) > 1:
            with st.expander(f"Historial de la sesión ({len(st.session_state['historial']) - 1} anteriores)"):
                for r in st.session_state["historial"][1:]:
                    st.markdown(f"- **{r['modelo']}** · {r['tiempos']['total_s']:.2f} s · {r['citas']['estado']} — {r['pregunta']}")
        st.download_button("Descargar historial (JSON)", json.dumps(st.session_state["historial"], ensure_ascii=False, indent=2),
                           file_name="historial_rag.json", mime="application/json")

with tab_comparar:
    st.caption("La misma pregunta con varios LLM. La evidencia recuperada es idéntica para todos "
               "(compare la *huella de contexto*): solo cambia el modelo, como en el experimento principal.")
    por_defecto = [m for m in nombres if m in experimento] or nombres[:1]
    elegidos = st.multiselect("Modelos a comparar", nombres, default=por_defecto)
    pregunta_cmp = st.text_area("Pregunta a comparar", key="pregunta_cmp", height=80)
    if st.button("Comparar", type="primary", disabled=not elegidos):
        if len(pregunta_cmp.strip()) < 3:
            st.warning("Escriba una pregunta de al menos 3 caracteres.")
        else:
            progreso = st.progress(0.0, text="Iniciando…")
            resultados = []
            for i, m in enumerate(elegidos):
                progreso.progress(i / len(elegidos), text=f"Generando con {m} ({i + 1}/{len(elegidos)})…")
                try:
                    resultados.append(llamar("POST", "/ask", timeout=TIMEOUT_ASK, json=cuerpo(pregunta_cmp.strip(), m)))
                except FalloAPI as e:
                    resultados.append({"modelo": m, "fallo": e})
            progreso.progress(1.0, text="Comparación terminada")
            st.session_state["comparacion"] = resultados
    resultados = st.session_state.get("comparacion", [])
    if resultados:
        columnas = st.columns(len(resultados))
        for col, res in zip(columnas, resultados):
            with col:
                if "fallo" in res:
                    st.markdown(f"**{res['modelo']}**")
                    mostrar_fallo(res["fallo"])
                else:
                    mostrar_resultado(res, compacto=True)
                    st.caption(f"Huella de contexto: `{res['contexto']['huella']}`")
        validos = [r for r in resultados if "fallo" not in r]
        if len({r["contexto"]["huella"] for r in validos}) > 1:
            st.warning("Los modelos recibieron contextos distintos: la comparación no es justa.")
        if validos:
            st.markdown("**Evidencias compartidas**")
            mostrar_evidencias(validos[0]["evidencias"])

with tab_recuperar:
    st.caption("Inspecciona qué recupera Milvus sin llamar a ningún LLM (útil para depurar y para el gold estándar).")
    pregunta_rec = st.text_input("Consulta", key="pregunta_rec")
    if st.button("Buscar") and len(pregunta_rec.strip()) >= 3:
        try:
            c = cuerpo(pregunta_rec.strip(), modelo)
            r = llamar("POST", "/retrieve", timeout=60,
                       json={x: c[x] for x in ("query", "k", "estrategia", "filtros")})
            m = r["metricas"]
            st.caption(f"{m['recuperados']} fragmentos · {m['documentos_unicos']} documentos · "
                       f"{m['tiempo_recuperacion_s']:.3f} s (codificación {m['tiempo_codificacion_s']:.3f} s, "
                       f"búsqueda {m['tiempo_busqueda_s']:.3f} s)")
            mostrar_evidencias(r["chunks"])
        except FalloAPI as e:
            mostrar_fallo(e)
