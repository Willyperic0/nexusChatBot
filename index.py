import streamlit as st
from chatbot import predict_class, get_response, intents
st.markdown(
    """
    <div style="display: flex; align-items: center;">
        <img src="https://img.icons8.com/ios-filled/50/000000/school.png" width="30" style="margin-right: 10px;">
        <h1 style="margin-bottom: 0;">Poke-Chatbot</h1>
    </div>
    """,
    unsafe_allow_html=True
)


# Inicializar el estado de los mensajes y el primer mensaje
if "messages" not in st.session_state:
    st.session_state.messages = []
if "first_message" not in st.session_state:
    st.session_state.first_message = True

# Mostrar mensajes anteriores
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Mensaje inicial del asistente
if st.session_state.first_message:
    with st.chat_message("assistant"):
        st.markdown("Hola, ¿Cómo puedo ayudarte?")
    st.session_state.messages.append({"role": "assistant", "content": "Hola, ¿cómo puedo ayudarte?"})
    st.session_state.first_message = False  # Corregir el nombre de la variable

# Capturar el mensaje del usuario
if prompt := st.chat_input("¿Cómo puedo ayudarte?"):
    # Mostrar y guardar el mensaje del usuario
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    insts = predict_class(prompt)
    res = get_response(insts, intents)

    # Mostrar y guardar el mensaje del asistente (que repite el del usuario)
    with st.chat_message("assistant"):
        st.markdown(res)
    st.session_state.messages.append({"role": "assistant", "content": prompt})