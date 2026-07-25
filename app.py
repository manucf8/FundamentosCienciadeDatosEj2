"""
Dashboard Inteligente del Sector Agro (Colombia)
--------------------------------------------------
Streamlit + Groq API (Llama 3.3 70B u otros modelos disponibles en Groq)

El dashboard calcula KPIs, EDA y gráficas sobre los datos agro, y arma
automáticamente un resumen estadístico del subconjunto filtrado que se
envía como contexto al modelo LLM para que el usuario pueda "chatear"
con sus propios datos y recibir interpretaciones ancladas en las cifras
reales (no inventadas).

El usuario ingresa su propia API Key de Groq desde la barra lateral.
"""

from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import streamlit as st
from groq import Groq

# ----------------------------------------------------------------------------
# CONFIGURACIÓN GENERAL
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="Dashboard Inteligente Agro Colombia",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded",
)
sns.set_theme(style="whitegrid")

MODELOS_DISPONIBLES = {
    "Llama 3.3 70B Versatile (Groq)": "llama-3.3-70b-versatile",
    "GPT-OSS 120B (Groq)": "openai/gpt-oss-120b",
    "Qwen 3.6 27B (Groq)": "qwen/qwen3.6-27b",
}

VARIABLES_NUMERICAS = ["Area_Hectareas", "Produccion_Anual_Ton", "Precio_Venta_Por_Ton_COP", "Rendimiento_Ton_Ha"]

NOMBRES_LEGIBLES = {
    "Area_Hectareas": "Área (hectáreas)",
    "Produccion_Anual_Ton": "Producción anual (Ton)",
    "Precio_Venta_Por_Ton_COP": "Precio de venta (COP/Ton)",
    "Rendimiento_Ton_Ha": "Rendimiento (Ton/Ha)",
}

SYSTEM_TEMPLATE = """Eres un analista experto en agroindustria colombiana.
Tu tarea es explicar e interpretar resultados estadísticos de un dashboard de fincas agrícolas,
en español, de forma clara y útil para un usuario de negocio (no necesariamente técnico).

Reglas importantes:
- Basa TODAS tus respuestas únicamente en el resumen de datos que se te entrega a continuación.
- Si el usuario pregunta algo que no se puede responder con este resumen, dilo explícitamente
  y sugiere qué otro corte o filtro permitiría responderlo, en vez de inventar cifras.
- Da explicaciones orientadas a decisiones: qué implica el patrón, qué riesgos u oportunidades sugiere.
- Sé conciso: usa viñetas cuando ayude a la claridad.

## Resumen estadístico de los datos filtrados actualmente en el dashboard

{contexto}
"""

# ----------------------------------------------------------------------------
# CARGA Y PREPARACIÓN DE DATOS
# ----------------------------------------------------------------------------
@st.cache_data
def cargar_datos(fuente):
    df = pd.read_csv(fuente)

    columnas_esperadas = {
        "ID_Finca", "Departamento", "Tipo_Cultivo", "Area_Hectareas", "Produccion_Anual_Ton",
        "Sistema_Riego_Tecnificado", "Nivel_Tecnificacion", "Precio_Venta_Por_Ton_COP",
        "Tipo_Suelo", "Fecha_Ultima_Auditoria",
    }
    faltantes = columnas_esperadas - set(df.columns)
    if faltantes:
        raise ValueError(f"Faltan columnas esperadas en el CSV: {faltantes}")

    if df["Sistema_Riego_Tecnificado"].dtype == object:
        df["Sistema_Riego_Tecnificado"] = df["Sistema_Riego_Tecnificado"].astype(str).str.lower().map(
            {"true": True, "false": False}
        )

    df["Fecha_Ultima_Auditoria"] = pd.to_datetime(df["Fecha_Ultima_Auditoria"], errors="coerce")
    df["Rendimiento_Ton_Ha"] = df["Produccion_Anual_Ton"] / df["Area_Hectareas"].replace(0, np.nan)

    orden_tec = ["Bajo", "Medio", "Alto", "Muy Alto"]
    if set(df["Nivel_Tecnificacion"].unique()).issubset(set(orden_tec)):
        df["Nivel_Tecnificacion"] = pd.Categorical(df["Nivel_Tecnificacion"], categories=orden_tec, ordered=True)

    return df


def formato_pct(parte, total):
    return f"{(parte / total * 100):.1f}%" if total else "0%"


def construir_contexto(df: pd.DataFrame) -> str:
    """Genera un resumen estadístico compacto en texto plano para dar contexto al LLM."""
    if df.empty:
        return "No hay registros disponibles con los filtros actuales."

    n = len(df)
    lineas = []
    lineas.append(f"- Registros (fincas) analizados: {n}")
    lineas.append(f"- Área total: {df['Area_Hectareas'].sum():,.1f} ha | Producción total: {df['Produccion_Anual_Ton'].sum():,.1f} ton")
    lineas.append(f"- Rendimiento promedio: {df['Rendimiento_Ton_Ha'].mean():.2f} ton/ha (mín {df['Rendimiento_Ton_Ha'].min():.2f}, máx {df['Rendimiento_Ton_Ha'].max():.2f})")
    lineas.append(f"- Precio de venta promedio: {df['Precio_Venta_Por_Ton_COP'].mean():,.0f} COP/ton (mín {df['Precio_Venta_Por_Ton_COP'].min():,.0f}, máx {df['Precio_Venta_Por_Ton_COP'].max():,.0f})")

    lineas.append("\n### Por departamento (rendimiento y precio promedio)")
    dep = df.groupby("Departamento").agg(
        rendimiento=("Rendimiento_Ton_Ha", "mean"),
        precio=("Precio_Venta_Por_Ton_COP", "mean"),
        n=("ID_Finca", "count"),
    ).sort_values("rendimiento", ascending=False)
    for depto, row in dep.iterrows():
        lineas.append(f"- {depto}: rendimiento {row['rendimiento']:.2f} ton/ha, precio {row['precio']:,.0f} COP/ton, n={int(row['n'])}")

    lineas.append("\n### Por tipo de cultivo (rendimiento y precio promedio)")
    cult = df.groupby("Tipo_Cultivo").agg(
        rendimiento=("Rendimiento_Ton_Ha", "mean"),
        precio=("Precio_Venta_Por_Ton_COP", "mean"),
        n=("ID_Finca", "count"),
    ).sort_values("rendimiento", ascending=False)
    for cultivo, row in cult.iterrows():
        lineas.append(f"- {cultivo}: rendimiento {row['rendimiento']:.2f} ton/ha, precio {row['precio']:,.0f} COP/ton, n={int(row['n'])}")

    lineas.append("\n### Por nivel de tecnificación (rendimiento promedio)")
    tec = df.groupby("Nivel_Tecnificacion", observed=True)["Rendimiento_Ton_Ha"].mean().sort_values(ascending=False)
    for nivel, val in tec.items():
        lineas.append(f"- {nivel}: {val:.2f} ton/ha")

    lineas.append("\n### Riego tecnificado vs. no tecnificado")
    riego = df.groupby("Sistema_Riego_Tecnificado")["Rendimiento_Ton_Ha"].mean()
    con_riego = riego.get(True, float("nan"))
    sin_riego = riego.get(False, float("nan"))
    lineas.append(f"- Con riego tecnificado: {con_riego:.2f} ton/ha | Sin riego tecnificado: {sin_riego:.2f} ton/ha")

    lineas.append("\n### Por tipo de suelo (rendimiento promedio)")
    suelo = df.groupby("Tipo_Suelo")["Rendimiento_Ton_Ha"].mean().sort_values(ascending=False)
    for tipo, val in suelo.items():
        lineas.append(f"- {tipo}: {val:.2f} ton/ha")

    lineas.append("\n### Correlaciones entre variables numéricas")
    corr = df[["Area_Hectareas", "Produccion_Anual_Ton", "Precio_Venta_Por_Ton_COP", "Rendimiento_Ton_Ha"]].corr()
    lineas.append(f"- Área vs. Producción: {corr.loc['Area_Hectareas', 'Produccion_Anual_Ton']:.2f}")
    lineas.append(f"- Rendimiento vs. Precio: {corr.loc['Rendimiento_Ton_Ha', 'Precio_Venta_Por_Ton_COP']:.2f}")
    lineas.append(f"- Área vs. Rendimiento: {corr.loc['Area_Hectareas', 'Rendimiento_Ton_Ha']:.2f}")

    if df["Fecha_Ultima_Auditoria"].notna().any():
        lineas.append(f"\n### Auditorías\n- Rango de fechas de última auditoría: {df['Fecha_Ultima_Auditoria'].min().date()} a {df['Fecha_Ultima_Auditoria'].max().date()}")

    return "\n".join(lineas)


# ----------------------------------------------------------------------------
# SIDEBAR: CARGA DE ARCHIVO
# ----------------------------------------------------------------------------
st.sidebar.title("⚙️ Configuración")
st.sidebar.markdown("#### 📂 Datos")

archivo_subido = st.sidebar.file_uploader("Sube tu CSV de fincas agro", type=["csv"])

error_carga = None
if archivo_subido is not None:
    try:
        df_raw = cargar_datos(archivo_subido)
        st.sidebar.success(f"Archivo **{archivo_subido.name}** cargado ✅")
    except Exception as e:
        error_carga = str(e)
        df_raw = cargar_datos("agro_colombia.csv")
        st.sidebar.error(f"Error al leer el archivo: {error_carga}\nSe usará el dataset de ejemplo.")
else:
    df_raw = cargar_datos("agro_colombia.csv")
    st.sidebar.info("Usando el dataset de ejemplo incluido (`agro_colombia.csv`).")

st.sidebar.markdown("---")
st.sidebar.markdown("#### 🔍 Filtros")

deptos_disp = sorted(df_raw["Departamento"].dropna().unique().tolist())
cultivos_disp = sorted(df_raw["Tipo_Cultivo"].dropna().unique().tolist())
suelos_disp = sorted(df_raw["Tipo_Suelo"].dropna().unique().tolist())

deptos_sel = st.sidebar.multiselect("Departamento", deptos_disp, default=deptos_disp)
cultivos_sel = st.sidebar.multiselect("Tipo de cultivo", cultivos_disp, default=cultivos_disp)
suelos_sel = st.sidebar.multiselect("Tipo de suelo", suelos_disp, default=suelos_disp)
riego_sel = st.sidebar.radio("Riego tecnificado", ["Todas", "Con riego", "Sin riego"], index=0)

df = df_raw[
    df_raw["Departamento"].isin(deptos_sel)
    & df_raw["Tipo_Cultivo"].isin(cultivos_sel)
    & df_raw["Tipo_Suelo"].isin(suelos_sel)
].copy()
if riego_sel == "Con riego":
    df = df[df["Sistema_Riego_Tecnificado"] == True]
elif riego_sel == "Sin riego":
    df = df[df["Sistema_Riego_Tecnificado"] == False]

st.sidebar.caption(f"Registros tras filtros: **{len(df)}** de {len(df_raw)}")

st.sidebar.markdown("---")
st.sidebar.markdown("#### 🤖 Asistente IA (Groq)")
api_key = st.sidebar.text_input(
    "API Key de Groq", type="password", placeholder="gsk_...",
    help="Tu clave nunca se guarda ni se envía a ningún lugar distinto de Groq.",
)
modelo_label = st.sidebar.selectbox("Modelo", list(MODELOS_DISPONIBLES.keys()), index=0)
modelo_id = MODELOS_DISPONIBLES[modelo_label]
temperatura = st.sidebar.slider("Creatividad (temperature)", 0.0, 1.5, 0.4, 0.1)

if st.sidebar.button("🗑️ Borrar historial de chat"):
    st.session_state.chat_agro = []
    st.rerun()

st.sidebar.caption(
    "⚠️ Nota: Groq retira `llama-3.3-70b-versatile` el 16 de agosto de 2026. "
    "Si deja de funcionar, cambia a `openai/gpt-oss-120b` o `qwen/qwen3.6-27b`."
)

if df.empty:
    st.warning("No hay registros con los filtros seleccionados. Ajusta los filtros en la barra lateral.")
    st.stop()

contexto_actual = construir_contexto(df)

# ----------------------------------------------------------------------------
# ENCABEZADO Y KPIs
# ----------------------------------------------------------------------------
st.title("🌱 Dashboard Inteligente del Sector Agro — Colombia")
st.markdown(
    "Explora indicadores de producción agrícola y **chatea con un modelo Llama (vía Groq)** "
    "para que interprete los resultados en lenguaje natural, anclado en las cifras reales del dashboard."
)

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Fincas", len(df))
k2.metric("Área total (ha)", f"{df['Area_Hectareas'].sum():,.0f}")
k3.metric("Producción total (ton)", f"{df['Produccion_Anual_Ton'].sum():,.0f}")
k4.metric("Rendimiento prom. (ton/ha)", f"{df['Rendimiento_Ton_Ha'].mean():.2f}")
k5.metric("Precio prom. (COP/ton)", f"{df['Precio_Venta_Por_Ton_COP'].mean():,.0f}")

st.markdown("---")

tabs = st.tabs(["📂 Datos & EDA", "📊 Gráficas", "🤖 Asistente IA — Interpretación con Llama"])

# ----------------------------------------------------------------------------
# TAB 1: DATOS & EDA
# ----------------------------------------------------------------------------
with tabs[0]:
    st.subheader("📂 Vista previa de datos")
    if error_carga:
        st.error(f"El archivo subido no pudo procesarse: {error_carga}")
    st.dataframe(df.head(10), use_container_width=True)

    st.markdown("#### Estadística descriptiva")
    st.dataframe(df[VARIABLES_NUMERICAS].describe().T.style.format("{:.2f}"), use_container_width=True)

    st.markdown("#### Calidad de datos")
    c1, c2 = st.columns(2)
    with c1:
        nulos = df_raw.isnull().sum()
        if nulos.sum() == 0:
            st.success("No se detectaron valores nulos.")
        else:
            st.dataframe(nulos[nulos > 0].rename("Nulos"))
    with c2:
        dup = df_raw.duplicated(subset=["ID_Finca"]).sum()
        st.success("Sin IDs de finca duplicados.") if dup == 0 else st.warning(f"{dup} IDs duplicados.")

    st.markdown("#### Matriz de correlación")
    corr = df[VARIABLES_NUMERICAS].corr()
    fig_corr, ax_corr = plt.subplots(figsize=(6, 4.5))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax_corr,
                xticklabels=[NOMBRES_LEGIBLES[c] for c in corr.columns],
                yticklabels=[NOMBRES_LEGIBLES[c] for c in corr.columns])
    st.pyplot(fig_corr)
    plt.close(fig_corr)

    st.download_button(
        "⬇️ Descargar datos filtrados (CSV)",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="agro_colombia_filtrado.csv",
        mime="text/csv",
    )

# ----------------------------------------------------------------------------
# TAB 2: GRÁFICAS
# ----------------------------------------------------------------------------
with tabs[1]:
    st.subheader("📊 Visualizaciones")

    st.markdown("#### Rendimiento (ton/ha) por departamento")
    orden_dep = df.groupby("Departamento")["Rendimiento_Ton_Ha"].mean().sort_values(ascending=False).index
    fig1 = px.box(df, x="Departamento", y="Rendimiento_Ton_Ha", category_orders={"Departamento": list(orden_dep)},
                  labels={"Rendimiento_Ton_Ha": "Rendimiento (ton/ha)"})
    st.plotly_chart(fig1, use_container_width=True)

    st.markdown("#### Producción total por tipo de cultivo")
    prod_cultivo = df.groupby("Tipo_Cultivo")["Produccion_Anual_Ton"].sum().sort_values(ascending=False).reset_index()
    fig2 = px.bar(prod_cultivo, x="Tipo_Cultivo", y="Produccion_Anual_Ton",
                  labels={"Produccion_Anual_Ton": "Producción total (ton)", "Tipo_Cultivo": "Cultivo"})
    st.plotly_chart(fig2, use_container_width=True)

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("#### Precio de venta por nivel de tecnificación")
        fig3, ax3 = plt.subplots(figsize=(6, 4.5))
        orden_tec_presente = [t for t in ["Bajo", "Medio", "Alto", "Muy Alto"] if t in df["Nivel_Tecnificacion"].unique()]
        sns.boxplot(data=df, x="Nivel_Tecnificacion", y="Precio_Venta_Por_Ton_COP", order=orden_tec_presente, palette="crest", ax=ax3)
        ax3.set_xlabel("Nivel de tecnificación")
        ax3.set_ylabel("Precio (COP/ton)")
        st.pyplot(fig3)
        plt.close(fig3)

    with col_b:
        st.markdown("#### Rendimiento: con vs. sin riego tecnificado")
        fig4, ax4 = plt.subplots(figsize=(6, 4.5))
        sns.barplot(data=df, x="Sistema_Riego_Tecnificado", y="Rendimiento_Ton_Ha", palette="Set2", ax=ax4, errorbar="sd")
        ax4.set_xticklabels(["Sin riego", "Con riego"])
        ax4.set_ylabel("Rendimiento (ton/ha)")
        ax4.set_xlabel("")
        st.pyplot(fig4)
        plt.close(fig4)

    st.markdown("#### Relación Área vs. Producción (por cultivo)")
    fig5 = px.scatter(df, x="Area_Hectareas", y="Produccion_Anual_Ton", color="Tipo_Cultivo",
                       size="Precio_Venta_Por_Ton_COP", hover_data=["Departamento", "Tipo_Suelo"],
                       labels={"Area_Hectareas": "Área (ha)", "Produccion_Anual_Ton": "Producción (ton)"})
    st.plotly_chart(fig5, use_container_width=True)

    st.markdown("#### Rendimiento promedio por tipo de suelo")
    suelo_rend = df.groupby("Tipo_Suelo")["Rendimiento_Ton_Ha"].mean().sort_values(ascending=False).reset_index()
    fig6 = px.bar(suelo_rend, x="Tipo_Suelo", y="Rendimiento_Ton_Ha",
                  labels={"Rendimiento_Ton_Ha": "Rendimiento (ton/ha)", "Tipo_Suelo": "Tipo de suelo"})
    st.plotly_chart(fig6, use_container_width=True)

# ----------------------------------------------------------------------------
# TAB 3: ASISTENTE IA
# ----------------------------------------------------------------------------
with tabs[2]:
    st.subheader("🤖 Asistente IA — Interpretación con Llama (Groq)")
    st.caption(
        "El asistente recibe automáticamente un resumen estadístico de los datos filtrados "
        "(no el CSV completo) y responde basándose en esas cifras."
    )

    with st.expander("📋 Ver el resumen de datos que recibe el modelo (contexto actual)"):
        st.text(contexto_actual)

    if "chat_agro" not in st.session_state:
        st.session_state.chat_agro = []

    col_btn1, col_btn2 = st.columns([1, 1])
    interpretacion_auto = col_btn1.button("🔮 Generar interpretación automática de los datos filtrados")
    col_btn2.caption("O simplemente escribe tu propia pregunta abajo.")

    for msg in st.session_state.chat_agro:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    def responder(pregunta_usuario: str):
        if not api_key:
            st.error("⚠️ Debes ingresar tu API Key de Groq en la barra lateral antes de chatear.")
            return

        st.session_state.chat_agro.append({"role": "user", "content": pregunta_usuario})
        with st.chat_message("user"):
            st.markdown(pregunta_usuario)

        mensajes_api = [{"role": "system", "content": SYSTEM_TEMPLATE.format(contexto=contexto_actual)}]
        mensajes_api += [{"role": m["role"], "content": m["content"]} for m in st.session_state.chat_agro]

        with st.chat_message("assistant"):
            placeholder = st.empty()
            respuesta_completa = ""
            try:
                client = Groq(api_key=api_key)
                stream = client.chat.completions.create(
                    model=modelo_id,
                    messages=mensajes_api,
                    temperature=temperatura,
                    max_tokens=1200,
                    stream=True,
                )
                for chunk in stream:
                    delta = chunk.choices[0].delta.content or ""
                    respuesta_completa += delta
                    placeholder.markdown(respuesta_completa + "▌")
                placeholder.markdown(respuesta_completa)
            except Exception as e:
                respuesta_completa = (
                    f"❌ Error al llamar a la API de Groq:\n\n`{e}`\n\n"
                    "Verifica tu API Key, el modelo seleccionado y tu conexión a internet."
                )
                placeholder.error(respuesta_completa)

        st.session_state.chat_agro.append({"role": "assistant", "content": respuesta_completa})

    if interpretacion_auto:
        responder(
            "Genera una interpretación ejecutiva de los datos filtrados actualmente: "
            "menciona los departamentos y cultivos más y menos eficientes en rendimiento, "
            "el efecto del riego tecnificado y del tipo de suelo, y 2-3 recomendaciones accionables."
        )

    pregunta = st.chat_input("Pregunta algo sobre estos datos (ej. '¿qué departamento tiene mejor rendimiento?')")
    if pregunta:
        responder(pregunta)

st.markdown("---")
st.caption("Dashboard construido con Streamlit · seaborn · plotly · Groq (Llama) — Ejercicio de estadística / ciencia de datos.")
