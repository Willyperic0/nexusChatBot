import os
import streamlit as st
import chatbot  # usa ChatRuntime de tu chatbot.py

# Reduce logs de TF si está instalado (no estorba si no lo está)
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

st.set_page_config(page_title="Poke-Chatbot", page_icon="🎓", layout="centered")

st.markdown(
    """
    <div style="display: flex; align-items: center;">
        <img src="https://img.icons8.com/ios-filled/50/000000/school.png" width="30" style="margin-right: 10px;">
        <h1 style="margin-bottom: 0;">Chatbot</h1>
    </div>
    """,
    unsafe_allow_html=True
)

@st.cache_resource(show_spinner=True)
def get_runtime():
    return chatbot.ChatRuntime()

rt = get_runtime()

# Estado inicial
if "messages" not in st.session_state:
    st.session_state.messages = []
if "first_message" not in st.session_state:
    st.session_state.first_message = True

# Mostrar historial
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Mensaje inicial del asistente
if st.session_state.first_message:
    initial = "Hola, ¿cómo puedo ayudarte?"
    with st.chat_message("assistant"):
        st.markdown(initial)
    st.session_state.messages.append({"role": "assistant", "content": initial})
    st.session_state.first_message = False

# Capturar mensaje del usuario
if prompt := st.chat_input("¿Cómo puedo ayudarte?"):
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    try:
        res = rt.get_response(prompt, nn_threshold=0.5)
    except Exception as e:
        res = f"⚠️ Ocurrió un error procesando tu mensaje: `{e}`"

    with st.chat_message("assistant"):
        st.markdown(res)
    st.session_state.messages.append({"role": "assistant", "content": res})
