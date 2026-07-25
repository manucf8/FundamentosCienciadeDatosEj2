"""
Chatbot de Cultura General e Historia Mundial
-----------------------------------------------
Streamlit + Groq API (Llama 3.3 70B Versatile)

El usuario ingresa su propia API Key de Groq desde la barra lateral.
"""

import streamlit as st
from groq import Groq

# ----------------------------------------------------------------------------
# CONFIGURACIÓN GENERAL
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="Chatbot de Cultura General e Historia",
    page_icon="📚",
    layout="centered",
)

MODELOS_DISPONIBLES = {
    "Llama 3.3 70B Versatile (Groq)": "llama-3.3-70b-versatile",
    "GPT-OSS 120B (Groq)": "openai/gpt-oss-120b",
    "Qwen 3.6 27B (Groq)": "qwen/qwen3.6-27b",
}

SYSTEM_PROMPT = (
    "Eres un asistente experto en cultura general e historia mundial. "
    "Respondes de forma clara, precisa y educativa a preguntas sobre historia, "
    "geografía, arte, ciencia, literatura, política y sociedad a lo largo del tiempo. "
    "Cuando sea relevante, das contexto histórico (fechas, lugares, personajes clave) "
    "y señalas si un dato es controvertido o tiene distintas interpretaciones según la fuente. "
    "Si no sabes algo con certeza, lo dices explícitamente en vez de inventar datos. "
    "Responde en español, de manera concisa pero completa."
)

# ----------------------------------------------------------------------------
# SIDEBAR: API KEY Y CONFIGURACIÓN
# ----------------------------------------------------------------------------
st.sidebar.title("⚙️ Configuración")

api_key = st.sidebar.text_input(
    "API Key de Groq",
    type="password",
    placeholder="gsk_...",
    help="Tu clave nunca se guarda ni se envía a ningún lugar distinto de Groq.",
)

modelo_label = st.sidebar.selectbox("Modelo", list(MODELOS_DISPONIBLES.keys()), index=0)
modelo_id = MODELOS_DISPONIBLES[modelo_label]

st.sidebar.markdown("---")
temperatura = st.sidebar.slider("Creatividad (temperature)", 0.0, 1.5, 0.6, 0.1)
max_tokens = st.sidebar.slider("Longitud máxima de respuesta (tokens)", 256, 2048, 1024, 128)

st.sidebar.markdown("---")
if st.sidebar.button("🗑️ Borrar historial de chat"):
    st.session_state.mensajes = []
    st.rerun()

st.sidebar.caption(
    "⚠️ Nota: Groq anunció que `llama-3.3-70b-versatile` se retira el **16 de agosto de 2026**. "
    "Si la app deja de responder con ese modelo después de esa fecha, cambia a `openai/gpt-oss-120b` "
    "o `qwen/qwen3.6-27b` en el selector de arriba."
)

# ----------------------------------------------------------------------------
# ENCABEZADO
# ----------------------------------------------------------------------------
st.title("📚 Chatbot de Cultura General e Historia Mundial")
st.markdown(
    "Pregunta lo que quieras sobre historia, geografía, arte, ciencia, política o sociedad. "
    "Ingresa tu API Key de Groq en la barra lateral para empezar."
)

# ----------------------------------------------------------------------------
# ESTADO DE LA CONVERSACIÓN
# ----------------------------------------------------------------------------
if "mensajes" not in st.session_state:
    st.session_state.mensajes = []

# Mostrar historial existente
for msg in st.session_state.mensajes:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ----------------------------------------------------------------------------
# ENTRADA DEL USUARIO
# ----------------------------------------------------------------------------
pregunta = st.chat_input("Escribe tu pregunta de cultura general o historia...")

if pregunta:
    if not api_key:
        st.error("⚠️ Debes ingresar tu API Key de Groq en la barra lateral antes de chatear.")
        st.stop()

    # Mostrar mensaje del usuario
    st.session_state.mensajes.append({"role": "user", "content": pregunta})
    with st.chat_message("user"):
        st.markdown(pregunta)

    # Construir historial para la API (system + turnos previos)
    mensajes_api = [{"role": "system", "content": SYSTEM_PROMPT}] + [
        {"role": m["role"], "content": m["content"]} for m in st.session_state.mensajes
    ]

    # Llamar a Groq y mostrar respuesta en streaming
    with st.chat_message("assistant"):
        placeholder = st.empty()
        respuesta_completa = ""
        try:
            client = Groq(api_key=api_key)
            stream = client.chat.completions.create(
                model=modelo_id,
                messages=mensajes_api,
                temperature=temperatura,
                max_tokens=max_tokens,
                stream=True,
            )
            for chunk in stream:
                delta = chunk.choices[0].delta.content or ""
                respuesta_completa += delta
                placeholder.markdown(respuesta_completa + "▌")
            placeholder.markdown(respuesta_completa)
        except Exception as e:
            respuesta_completa = (
                f"❌ Ocurrió un error al llamar a la API de Groq:\n\n`{e}`\n\n"
                "Verifica que tu API Key sea válida, que el modelo seleccionado siga disponible, "
                "y que tengas conexión a internet."
            )
            placeholder.error(respuesta_completa)

    st.session_state.mensajes.append({"role": "assistant", "content": respuesta_completa})

st.markdown("---")
st.caption("Powered by Groq · Llama 3.3 70B Versatile · Streamlit")
