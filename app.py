"""
Dashboard de Monitoreo Ambiental Urbano
----------------------------------------
Ejercicio de estadística / ciencia de datos.
Autor: Generado con Claude

Estructura:
1. Carga de archivo
2. EDA (Análisis Exploratorio de Datos)
3. Storytelling por variable
4. Gráficas (seaborn, plotly, pyplot)
5. Reporte / Conclusiones
"""

import io
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ----------------------------------------------------------------------------
# CONFIGURACIÓN GENERAL
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="Dashboard Monitoreo Ambiental",
    page_icon="🌎",
    layout="wide",
    initial_sidebar_state="expanded",
)

sns.set_theme(style="whitegrid")

ORDEN_ICA = [
    "Buena",
    "Moderada",
    "Dañina para grupos sensibles",
    "Dañina",
    "Muy Dañina",
    "Peligrosa",
]

COLOR_ICA = {
    "Buena": "#2ecc71",
    "Moderada": "#f1c40f",
    "Dañina para grupos sensibles": "#e67e22",
    "Dañina": "#e74c3c",
    "Muy Dañina": "#8e44ad",
    "Peligrosa": "#7f0000",
}

VARIABLES_NUMERICAS = [
    "PM2_5_Ug_m3",
    "Temperatura_C",
    "Humedad_Relativa_Pct",
    "Nivel_Ruido_dB",
]

NOMBRES_LEGIBLES = {
    "PM2_5_Ug_m3": "PM2.5 (µg/m³)",
    "Temperatura_C": "Temperatura (°C)",
    "Humedad_Relativa_Pct": "Humedad relativa (%)",
    "Nivel_Ruido_dB": "Nivel de ruido (dB)",
}


# ----------------------------------------------------------------------------
# CARGA Y PREPARACIÓN DE DATOS
# ----------------------------------------------------------------------------
@st.cache_data
def cargar_datos(fuente):
    df = pd.read_csv(fuente)

    columnas_esperadas = {
        "ID_Sensor", "Ciudad", "Tipo_Zona", "PM2_5_Ug_m3", "Temperatura_C",
        "Humedad_Relativa_Pct", "Presencia_Lluvia", "Nivel_Ruido_dB",
        "Indice_Calidad_Aire_ICA", "Hora_Lectura",
    }
    faltantes = columnas_esperadas - set(df.columns)
    if faltantes:
        raise ValueError(f"Faltan columnas esperadas en el CSV: {faltantes}")

    # Normalizar tipo booleano (por si llega como texto "True"/"False")
    if df["Presencia_Lluvia"].dtype == object:
        df["Presencia_Lluvia"] = df["Presencia_Lluvia"].astype(str).str.lower().map(
            {"true": True, "false": False}
        )

    # Hora -> franja horaria
    df["Hora_dt"] = pd.to_datetime(df["Hora_Lectura"], format="%H:%M", errors="coerce")
    df["Hora_num"] = df["Hora_dt"].dt.hour + df["Hora_dt"].dt.minute / 60

    def franja(h):
        if pd.isna(h):
            return "Sin dato"
        if h < 6:
            return "Madrugada (00-06)"
        elif h < 12:
            return "Mañana (06-12)"
        elif h < 18:
            return "Tarde (12-18)"
        else:
            return "Noche (18-24)"

    df["Franja_Horaria"] = df["Hora_num"].apply(franja)

    # Orden categórico del ICA (si todos los valores existen en el catálogo)
    categorias_presentes = set(df["Indice_Calidad_Aire_ICA"].dropna().unique())
    if categorias_presentes.issubset(set(ORDEN_ICA)):
        df["Indice_Calidad_Aire_ICA"] = pd.Categorical(
            df["Indice_Calidad_Aire_ICA"], categories=ORDEN_ICA, ordered=True
        )

    return df


def formato_pct(parte, total):
    return f"{(parte / total * 100):.1f}%" if total else "0%"


# ----------------------------------------------------------------------------
# SIDEBAR: CARGA + FILTROS
# ----------------------------------------------------------------------------
st.sidebar.title("⚙️ Configuración")
st.sidebar.markdown("#### 📂 1. Cargar archivo")

archivo_subido = st.sidebar.file_uploader(
    "Sube tu CSV de monitoreo ambiental", type=["csv"]
)

error_carga = None
if archivo_subido is not None:
    try:
        df_raw = cargar_datos(archivo_subido)
        st.sidebar.success(f"Archivo **{archivo_subido.name}** cargado correctamente ✅")
    except Exception as e:
        error_carga = str(e)
        df_raw = cargar_datos("monitoreo_ambiental.csv")
        st.sidebar.error(f"Error al leer el archivo: {error_carga}\nSe usará el dataset de ejemplo.")
else:
    df_raw = cargar_datos("monitoreo_ambiental.csv")
    st.sidebar.info("Usando el dataset de ejemplo incluido (`monitoreo_ambiental.csv`).")

st.sidebar.markdown("---")
st.sidebar.markdown("#### 🔍 Filtros")

ciudades_disp = sorted(df_raw["Ciudad"].dropna().unique().tolist())
zonas_disp = sorted(df_raw["Tipo_Zona"].dropna().unique().tolist())

ciudades_sel = st.sidebar.multiselect("Ciudad", ciudades_disp, default=ciudades_disp)
zonas_sel = st.sidebar.multiselect("Tipo de zona", zonas_disp, default=zonas_disp)
lluvia_sel = st.sidebar.radio("Condición de lluvia", ["Todas", "Con lluvia", "Sin lluvia"], index=0)

df = df_raw[df_raw["Ciudad"].isin(ciudades_sel) & df_raw["Tipo_Zona"].isin(zonas_sel)].copy()
if lluvia_sel == "Con lluvia":
    df = df[df["Presencia_Lluvia"] == True]
elif lluvia_sel == "Sin lluvia":
    df = df[df["Presencia_Lluvia"] == False]

st.sidebar.markdown("---")
st.sidebar.caption(f"Registros tras filtros: **{len(df)}** de {len(df_raw)}")

if df.empty:
    st.warning("No hay registros con los filtros seleccionados. Ajusta los filtros en la barra lateral.")
    st.stop()

# ----------------------------------------------------------------------------
# ENCABEZADO
# ----------------------------------------------------------------------------
st.title("🌎 Dashboard de Monitoreo Ambiental Urbano")
st.markdown(
    "Análisis exploratorio, storytelling con datos y reporte de conclusiones sobre "
    "**calidad del aire (PM2.5 e ICA)**, **ruido**, **temperatura**, **humedad** y **lluvia** "
    "en distintas zonas urbanas."
)

col_a, col_b, col_c, col_d = st.columns(4)
col_a.metric("Registros analizados", len(df))
col_b.metric("PM2.5 promedio (µg/m³)", f"{df['PM2_5_Ug_m3'].mean():.1f}")
col_c.metric("Ruido promedio (dB)", f"{df['Nivel_Ruido_dB'].mean():.1f}")
col_d.metric("% con lluvia", formato_pct((df["Presencia_Lluvia"] == True).sum(), len(df)))

st.markdown("---")

tabs = st.tabs([
    "📂 1. Carga de Datos",
    "🔎 2. EDA",
    "📖 3. Storytelling",
    "📊 4. Gráficas",
    "📝 5. Reporte y Conclusiones",
])

# ----------------------------------------------------------------------------
# TAB 1: CARGA DE DATOS
# ----------------------------------------------------------------------------
with tabs[0]:
    st.subheader("📂 Carga y vista previa de datos")

    if error_carga:
        st.error(f"El archivo subido no pudo procesarse: {error_carga}")

    st.markdown("**Vista previa (primeras 10 filas, con filtros aplicados):**")
    st.dataframe(df.head(10), use_container_width=True)

    c1, c2, c3 = st.columns(3)
    c1.metric("Filas", df.shape[0])
    c2.metric("Columnas", df.shape[0] and df_raw.shape[1])
    c3.metric("Sensores únicos", df["ID_Sensor"].nunique())

    st.markdown("**Tipos de dato por columna:**")
    tipos_df = pd.DataFrame({
        "Columna": df_raw.columns,
        "Tipo": [str(t) for t in df_raw.dtypes],
        "Nulos": df_raw.isnull().sum().values,
        "% Nulos": (df_raw.isnull().sum().values / len(df_raw) * 100).round(2),
    })
    st.dataframe(tipos_df, use_container_width=True, hide_index=True)

    st.download_button(
        "⬇️ Descargar datos filtrados (CSV)",
        data=df.drop(columns=["Hora_dt", "Hora_num"]).to_csv(index=False).encode("utf-8"),
        file_name="monitoreo_ambiental_filtrado.csv",
        mime="text/csv",
    )

# ----------------------------------------------------------------------------
# TAB 2: EDA
# ----------------------------------------------------------------------------
with tabs[1]:
    st.subheader("🔎 Análisis Exploratorio de Datos (EDA)")

    st.markdown("#### Estadística descriptiva — variables numéricas")
    st.dataframe(df[VARIABLES_NUMERICAS].describe().T.style.format("{:.2f}"), use_container_width=True)

    st.markdown("#### Distribución de variables categóricas")
    cat_c1, cat_c2, cat_c3 = st.columns(3)
    with cat_c1:
        st.markdown("**Ciudad**")
        st.dataframe(df["Ciudad"].value_counts().rename("Conteo"), use_container_width=True)
    with cat_c2:
        st.markdown("**Tipo de zona**")
        st.dataframe(df["Tipo_Zona"].value_counts().rename("Conteo"), use_container_width=True)
    with cat_c3:
        st.markdown("**Índice de Calidad del Aire (ICA)**")
        orden_presente = [c for c in ORDEN_ICA if c in df["Indice_Calidad_Aire_ICA"].unique()]
        conteo_ica = df["Indice_Calidad_Aire_ICA"].value_counts().reindex(orden_presente)
        st.dataframe(conteo_ica.rename("Conteo"), use_container_width=True)

    st.markdown("#### Valores nulos / calidad de datos")
    nulos = df_raw.isnull().sum()
    if nulos.sum() == 0:
        st.success("No se detectaron valores nulos en el dataset original.")
    else:
        st.dataframe(nulos[nulos > 0].rename("Nulos"), use_container_width=True)

    duplicados = df_raw.duplicated(subset=["ID_Sensor"]).sum()
    if duplicados == 0:
        st.success("No hay IDs de sensor duplicados.")
    else:
        st.warning(f"Se encontraron {duplicados} IDs de sensor duplicados.")

    st.markdown("#### Matriz de correlación (variables numéricas)")
    corr = df[VARIABLES_NUMERICAS].corr()
    fig_corr, ax_corr = plt.subplots(figsize=(6, 4.5))
    sns.heatmap(
        corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax_corr,
        xticklabels=[NOMBRES_LEGIBLES[c] for c in corr.columns],
        yticklabels=[NOMBRES_LEGIBLES[c] for c in corr.columns],
    )
    st.pyplot(fig_corr, use_container_width=False)
    plt.close(fig_corr)

    st.markdown("#### Detección de valores atípicos (outliers) — regla IQR")
    outlier_rows = []
    for col in VARIABLES_NUMERICAS:
        q1, q3 = df[col].quantile([0.25, 0.75])
        iqr = q3 - q1
        lim_inf, lim_sup = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        n_out = ((df[col] < lim_inf) | (df[col] > lim_sup)).sum()
        outlier_rows.append({
            "Variable": NOMBRES_LEGIBLES[col],
            "Límite inferior": round(lim_inf, 2),
            "Límite superior": round(lim_sup, 2),
            "N° outliers": n_out,
            "% outliers": formato_pct(n_out, len(df)),
        })
    st.dataframe(pd.DataFrame(outlier_rows), use_container_width=True, hide_index=True)

# ----------------------------------------------------------------------------
# TAB 3: STORYTELLING
# ----------------------------------------------------------------------------
with tabs[2]:
    st.subheader("📖 Storytelling por variable")
    st.caption("Insights generados dinámicamente a partir de los datos filtrados actualmente.")

    # --- Historia 1: PM2.5 por ciudad ---
    pm_ciudad = df.groupby("Ciudad")["PM2_5_Ug_m3"].mean().sort_values(ascending=False)
    ciudad_top = pm_ciudad.index[0]
    ciudad_bottom = pm_ciudad.index[-1]
    diff_pct = (pm_ciudad.iloc[0] - pm_ciudad.iloc[-1]) / pm_ciudad.iloc[-1] * 100 if pm_ciudad.iloc[-1] else 0

    st.markdown("### 🌫️ PM2.5 y la geografía de la contaminación")
    st.markdown(
        f"- **{ciudad_top}** presenta el promedio más alto de PM2.5 con **{pm_ciudad.iloc[0]:.1f} µg/m³**, "
        f"un **{diff_pct:.0f}% más** que **{ciudad_bottom}**, la ciudad con el nivel promedio más bajo "
        f"({pm_ciudad.iloc[-1]:.1f} µg/m³).\n"
        f"- Esto sugiere que la ubicación geográfica y las dinámicas urbanas locales (tráfico, industria, "
        f"altura) inciden directamente en la exposición a material particulado."
    )

    # --- Historia 2: Zona vs PM2.5 / Ruido ---
    zona_stats = df.groupby("Tipo_Zona")[["PM2_5_Ug_m3", "Nivel_Ruido_dB"]].mean().sort_values(
        "PM2_5_Ug_m3", ascending=False
    )
    zona_top_pm = zona_stats.index[0]
    zona_top_ruido = df.groupby("Tipo_Zona")["Nivel_Ruido_dB"].mean().idxmax()

    st.markdown("### 🏭 El tipo de zona importa")
    st.markdown(
        f"- La zona **{zona_top_pm}** concentra el promedio más alto de PM2.5 "
        f"({zona_stats.loc[zona_top_pm, 'PM2_5_Ug_m3']:.1f} µg/m³).\n"
        f"- La zona con mayor nivel de ruido promedio es **{zona_top_ruido}** "
        f"({df.groupby('Tipo_Zona')['Nivel_Ruido_dB'].mean().max():.1f} dB).\n"
        f"- Contaminación y ruido no siempre golpean el mismo tipo de zona: vale la pena revisar "
        f"políticas diferenciadas por uso del suelo en lugar de una regla única para toda la ciudad."
    )

    # --- Historia 3: Lluvia y PM2.5 ---
    pm_lluvia = df.groupby("Presencia_Lluvia")["PM2_5_Ug_m3"].mean()
    if True in pm_lluvia.index and False in pm_lluvia.index:
        con_lluvia = pm_lluvia.get(True, np.nan)
        sin_lluvia = pm_lluvia.get(False, np.nan)
        diferencia = sin_lluvia - con_lluvia
        direccion = "menor" if diferencia > 0 else "mayor"
        st.markdown("### 🌧️ ¿La lluvia limpia el aire?")
        st.markdown(
            f"- El PM2.5 promedio **con lluvia** es de **{con_lluvia:.1f} µg/m³**, frente a "
            f"**{sin_lluvia:.1f} µg/m³** sin lluvia.\n"
            f"- Es decir, con lluvia el PM2.5 es en promedio **{abs(diferencia):.1f} µg/m³ {direccion}** "
            f"que sin lluvia, un patrón consistente con el efecto de \"lavado atmosférico\" de partículas."
        )

    # --- Historia 4: ICA por ciudad ---
    st.markdown("### 🚦 Riesgo para la salud según el ICA")
    if isinstance(df["Indice_Calidad_Aire_ICA"].dtype, pd.CategoricalDtype):
        peores_categorias = ["Dañina", "Muy Dañina", "Peligrosa"]
        df["ICA_riesgo_alto"] = df["Indice_Calidad_Aire_ICA"].isin(peores_categorias)
        riesgo_ciudad = df.groupby("Ciudad")["ICA_riesgo_alto"].mean().sort_values(ascending=False)
        ciudad_riesgo = riesgo_ciudad.index[0]
        st.markdown(
            f"- En **{ciudad_riesgo}**, el **{riesgo_ciudad.iloc[0]*100:.0f}%** de las lecturas caen en "
            f"categorías de riesgo alto (Dañina, Muy Dañina o Peligrosa), la proporción más alta entre las ciudades analizadas.\n"
            f"- Esto tiene implicancias directas de salud pública: grupos sensibles (niños, adultos mayores, "
            f"personas con afecciones respiratorias) están más expuestos en esa ciudad."
        )

    # --- Historia 5: Patrón horario ---
    st.markdown("### 🕐 ¿Existen horas críticas del día?")
    franja_stats = df.groupby("Franja_Horaria")[["PM2_5_Ug_m3", "Nivel_Ruido_dB"]].mean()
    franja_top_pm = franja_stats["PM2_5_Ug_m3"].idxmax()
    franja_top_ruido = franja_stats["Nivel_Ruido_dB"].idxmax()
    st.markdown(
        f"- La franja horaria con mayor PM2.5 promedio es **{franja_top_pm}** "
        f"({franja_stats.loc[franja_top_pm, 'PM2_5_Ug_m3']:.1f} µg/m³).\n"
        f"- La franja con más ruido promedio es **{franja_top_ruido}** "
        f"({franja_stats.loc[franja_top_ruido, 'Nivel_Ruido_dB']:.1f} dB).\n"
        f"- Monitorear por franja horaria (en vez de solo promedios diarios) permite dirigir mejor "
        f"alertas tempranas y medidas de mitigación (p. ej. restricciones de tráfico en horas pico)."
    )

    # --- Historia 6: Correlaciones relevantes ---
    st.markdown("### 🔗 Relaciones entre variables")
    corr_pm_temp = df["PM2_5_Ug_m3"].corr(df["Temperatura_C"])
    corr_pm_hum = df["PM2_5_Ug_m3"].corr(df["Humedad_Relativa_Pct"])
    corr_pm_ruido = df["PM2_5_Ug_m3"].corr(df["Nivel_Ruido_dB"])
    st.markdown(
        f"- Correlación PM2.5 – Temperatura: **{corr_pm_temp:.2f}**\n"
        f"- Correlación PM2.5 – Humedad relativa: **{corr_pm_hum:.2f}**\n"
        f"- Correlación PM2.5 – Ruido: **{corr_pm_ruido:.2f}**\n\n"
        f"En general estas correlaciones son {'débiles' if max(abs(corr_pm_temp), abs(corr_pm_hum), abs(corr_pm_ruido)) < 0.3 else 'moderadas o fuertes'}, "
        f"lo que sugiere que el PM2.5 en esta muestra está más determinado por la ubicación (ciudad/zona) "
        f"que por las condiciones climáticas puntuales del momento de la lectura."
    )

# ----------------------------------------------------------------------------
# TAB 4: GRÁFICAS
# ----------------------------------------------------------------------------
with tabs[3]:
    st.subheader("📊 Visualizaciones")
    sub_seaborn, sub_plotly, sub_pyplot = st.tabs(["🎨 Seaborn", "⚡ Plotly (interactivo)", "📐 Matplotlib / Pyplot"])

    # ---------------- SEABORN ----------------
    with sub_seaborn:
        st.markdown("#### Boxplot de PM2.5 por ciudad")
        fig1, ax1 = plt.subplots(figsize=(8, 4.5))
        orden_ciudades_pm = df.groupby("Ciudad")["PM2_5_Ug_m3"].median().sort_values(ascending=False).index
        sns.boxplot(data=df, x="Ciudad", y="PM2_5_Ug_m3", order=orden_ciudades_pm, palette="Set2", ax=ax1)
        ax1.set_xlabel("Ciudad")
        ax1.set_ylabel("PM2.5 (µg/m³)")
        ax1.tick_params(axis="x", rotation=20)
        st.pyplot(fig1)
        plt.close(fig1)

        st.markdown("#### Distribución de PM2.5 (histograma + densidad)")
        fig2, ax2 = plt.subplots(figsize=(8, 4.5))
        sns.histplot(data=df, x="PM2_5_Ug_m3", kde=True, color="#3498db", ax=ax2)
        ax2.set_xlabel("PM2.5 (µg/m³)")
        st.pyplot(fig2)
        plt.close(fig2)

        st.markdown("#### Ruido promedio por tipo de zona")
        fig3, ax3 = plt.subplots(figsize=(8, 4.5))
        orden_zonas_ruido = df.groupby("Tipo_Zona")["Nivel_Ruido_dB"].mean().sort_values(ascending=False).index
        sns.barplot(data=df, x="Tipo_Zona", y="Nivel_Ruido_dB", order=orden_zonas_ruido, palette="viridis", ax=ax3, errorbar="sd")
        ax3.set_xlabel("Tipo de zona")
        ax3.set_ylabel("Nivel de ruido (dB)")
        ax3.tick_params(axis="x", rotation=20)
        st.pyplot(fig3)
        plt.close(fig3)

        st.markdown("#### Relación Temperatura vs PM2.5, coloreado por lluvia")
        fig4, ax4 = plt.subplots(figsize=(8, 4.5))
        sns.scatterplot(
            data=df, x="Temperatura_C", y="PM2_5_Ug_m3", hue="Presencia_Lluvia",
            palette={True: "#2980b9", False: "#e67e22"}, ax=ax4, alpha=0.7,
        )
        ax4.set_xlabel("Temperatura (°C)")
        ax4.set_ylabel("PM2.5 (µg/m³)")
        st.pyplot(fig4)
        plt.close(fig4)

    # ---------------- PLOTLY ----------------
    with sub_plotly:
        st.markdown("#### PM2.5 promedio por ciudad y tipo de zona")
        pivot_pm = df.groupby(["Ciudad", "Tipo_Zona"])["PM2_5_Ug_m3"].mean().reset_index()
        fig5 = px.bar(
            pivot_pm, x="Ciudad", y="PM2_5_Ug_m3", color="Tipo_Zona", barmode="group",
            labels={"PM2_5_Ug_m3": "PM2.5 (µg/m³)"}, title=None,
        )
        st.plotly_chart(fig5, use_container_width=True)

        st.markdown("#### Distribución del Índice de Calidad del Aire (ICA)")
        conteo_ica_total = df["Indice_Calidad_Aire_ICA"].value_counts().reindex(
            [c for c in ORDEN_ICA if c in df["Indice_Calidad_Aire_ICA"].unique()]
        ).reset_index()
        conteo_ica_total.columns = ["ICA", "Conteo"]
        fig6 = px.pie(
            conteo_ica_total, names="ICA", values="Conteo", color="ICA",
            color_discrete_map=COLOR_ICA, hole=0.35,
        )
        st.plotly_chart(fig6, use_container_width=True)

        st.markdown("#### Dispersión interactiva: Humedad vs PM2.5")
        fig7 = px.scatter(
            df, x="Humedad_Relativa_Pct", y="PM2_5_Ug_m3", color="Ciudad",
            size="Nivel_Ruido_dB", hover_data=["Tipo_Zona", "Indice_Calidad_Aire_ICA"],
            labels={"Humedad_Relativa_Pct": "Humedad relativa (%)", "PM2_5_Ug_m3": "PM2.5 (µg/m³)"},
        )
        st.plotly_chart(fig7, use_container_width=True)

        st.markdown("#### PM2.5 promedio por franja horaria")
        franja_orden = ["Madrugada (00-06)", "Mañana (06-12)", "Tarde (12-18)", "Noche (18-24)"]
        pm_franja = df.groupby("Franja_Horaria")["PM2_5_Ug_m3"].mean().reindex(
            [f for f in franja_orden if f in df["Franja_Horaria"].unique()]
        ).reset_index()
        fig8 = px.line(pm_franja, x="Franja_Horaria", y="PM2_5_Ug_m3", markers=True,
                        labels={"PM2_5_Ug_m3": "PM2.5 (µg/m³)", "Franja_Horaria": "Franja horaria"})
        st.plotly_chart(fig8, use_container_width=True)

    # ---------------- PYPLOT (matplotlib puro) ----------------
    with sub_pyplot:
        st.markdown("#### Serie de PM2.5 a lo largo del día (todas las lecturas)")
        df_hora = df.dropna(subset=["Hora_num"]).sort_values("Hora_num")
        fig9, ax9 = plt.subplots(figsize=(9, 4.5))
        ax9.scatter(df_hora["Hora_num"], df_hora["PM2_5_Ug_m3"], alpha=0.5, color="#c0392b", s=20)
        ax9.set_xlabel("Hora del día")
        ax9.set_ylabel("PM2.5 (µg/m³)")
        ax9.set_xlim(0, 24)
        st.pyplot(fig9)
        plt.close(fig9)

        st.markdown("#### Comparación de medias: PM2.5 por ciudad (barras de error)")
        stats_ciudad = df.groupby("Ciudad")["PM2_5_Ug_m3"].agg(["mean", "std"]).sort_values("mean", ascending=False)
        fig10, ax10 = plt.subplots(figsize=(8, 4.5))
        ax10.bar(stats_ciudad.index, stats_ciudad["mean"], yerr=stats_ciudad["std"], capsize=5, color="#16a085")
        ax10.set_ylabel("PM2.5 promedio (µg/m³)")
        ax10.tick_params(axis="x", rotation=20)
        st.pyplot(fig10)
        plt.close(fig10)

        st.markdown("#### Proporción de lecturas con lluvia por ciudad")
        prop_lluvia = df.groupby("Ciudad")["Presencia_Lluvia"].mean().sort_values(ascending=False)
        fig11, ax11 = plt.subplots(figsize=(8, 4.5))
        ax11.barh(prop_lluvia.index, prop_lluvia.values * 100, color="#2980b9")
        ax11.set_xlabel("% de lecturas con lluvia")
        st.pyplot(fig11)
        plt.close(fig11)

# ----------------------------------------------------------------------------
# TAB 5: REPORTE Y CONCLUSIONES
# ----------------------------------------------------------------------------
with tabs[4]:
    st.subheader("📝 Reporte automático de conclusiones")
    st.caption("Generado dinámicamente según los filtros aplicados en la barra lateral.")

    pm_ciudad = df.groupby("Ciudad")["PM2_5_Ug_m3"].mean().sort_values(ascending=False)
    zona_stats = df.groupby("Tipo_Zona")["PM2_5_Ug_m3"].mean().sort_values(ascending=False)
    ruido_zona = df.groupby("Tipo_Zona")["Nivel_Ruido_dB"].mean().sort_values(ascending=False)
    pm_lluvia = df.groupby("Presencia_Lluvia")["PM2_5_Ug_m3"].mean()

    ica_riesgo_pct = np.nan
    ciudad_riesgo_txt = ""
    if isinstance(df["Indice_Calidad_Aire_ICA"].dtype, pd.CategoricalDtype):
        peores = ["Dañina", "Muy Dañina", "Peligrosa"]
        riesgo_total = df["Indice_Calidad_Aire_ICA"].isin(peores).mean() * 100
        riesgo_ciudad = df.groupby("Ciudad").apply(
            lambda g: g["Indice_Calidad_Aire_ICA"].isin(peores).mean() * 100
        ).sort_values(ascending=False)
        ica_riesgo_pct = riesgo_total
        ciudad_riesgo_txt = f"{riesgo_ciudad.index[0]} ({riesgo_ciudad.iloc[0]:.0f}%)"

    fecha_reporte = datetime.now().strftime("%Y-%m-%d %H:%M")

    reporte_md = f"""# Reporte de Monitoreo Ambiental Urbano

**Fecha de generación:** {fecha_reporte}
**Registros analizados:** {len(df)} (de {len(df_raw)} totales, tras filtros aplicados)
**Ciudades incluidas:** {", ".join(ciudades_sel)}
**Zonas incluidas:** {", ".join(zonas_sel)}

## Resumen ejecutivo

- El nivel promedio de PM2.5 en la muestra analizada es de **{df['PM2_5_Ug_m3'].mean():.1f} µg/m³**
  (mínimo: {df['PM2_5_Ug_m3'].min():.1f}, máximo: {df['PM2_5_Ug_m3'].max():.1f}).
- **{pm_ciudad.index[0]}** es la ciudad con mayor contaminación promedio por PM2.5
  ({pm_ciudad.iloc[0]:.1f} µg/m³), mientras que **{pm_ciudad.index[-1]}** presenta el nivel más bajo
  ({pm_ciudad.iloc[-1]:.1f} µg/m³).
- La zona tipo **{zona_stats.index[0]}** concentra el mayor promedio de PM2.5
  ({zona_stats.iloc[0]:.1f} µg/m³), y la zona **{ruido_zona.index[0]}** presenta el mayor nivel de
  ruido promedio ({ruido_zona.iloc[0]:.1f} dB).
- El {ica_riesgo_pct:.0f}% del total de lecturas corresponde a niveles de riesgo alto para la salud
  (Dañina, Muy Dañina o Peligrosa){f", concentrándose especialmente en {ciudad_riesgo_txt}" if ciudad_riesgo_txt else ""}.
- El nivel de ruido promedio general es de **{df['Nivel_Ruido_dB'].mean():.1f} dB**.

## Hallazgos por variable

### PM2.5 (material particulado)
Ranking de ciudades por PM2.5 promedio (µg/m³):
{chr(10).join(f"- {c}: {v:.1f}" for c, v in pm_ciudad.items())}

### Ruido
Ranking de zonas por ruido promedio (dB):
{chr(10).join(f"- {z}: {v:.1f}" for z, v in ruido_zona.items())}

### Efecto de la lluvia sobre PM2.5
- Con lluvia: {pm_lluvia.get(True, float('nan')):.1f} µg/m³
- Sin lluvia: {pm_lluvia.get(False, float('nan')):.1f} µg/m³

## Recomendaciones sugeridas

1. Priorizar monitoreo y medidas de mitigación en **{pm_ciudad.index[0]}** y en zonas tipo
   **{zona_stats.index[0]}**, donde se concentra la mayor exposición a PM2.5.
2. Evaluar políticas de reducción de ruido específicas para zonas **{ruido_zona.index[0]}**.
3. Reforzar alertas tempranas en las franjas horarias y condiciones climáticas asociadas a mayores
   picos de contaminación (ver pestaña de Storytelling).
4. Dar seguimiento particular a grupos sensibles en zonas con alta proporción de lecturas en
   categorías ICA de riesgo alto.

---
*Reporte generado automáticamente a partir de los datos filtrados en el dashboard.*
"""

    st.markdown(reporte_md)

    st.download_button(
        "⬇️ Descargar reporte (Markdown)",
        data=reporte_md.encode("utf-8"),
        file_name="reporte_monitoreo_ambiental.md",
        mime="text/markdown",
    )

st.markdown("---")
st.caption("Dashboard construido con Streamlit · seaborn · plotly · matplotlib — Ejercicio de estadística / ciencia de datos.")
