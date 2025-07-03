import streamlit as st
import requests  # To connect to your backend

st.set_page_config(page_title="Banking Chatbot", page_icon="🤖")
st.title("🏦 Smart Banking Assistant")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User input
if prompt := st.chat_input("Ask about balance, transactions, or loans"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Get bot response from backend
    try:
        response = requests.post(
            "http://localhost:8000/chat",
            json={"query": prompt}
        ).json()
        bot_response = response["response"]
    except:
        bot_response = "⚠️ Service unavailable. Please try later."

    # Add bot response to chat history
    st.session_state.messages.append({"role": "assistant", "content": bot_response})
    with st.chat_message("assistant"):
        st.markdown(bot_response)