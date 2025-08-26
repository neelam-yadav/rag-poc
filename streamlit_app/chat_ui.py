# streamlit_app/chat_ui.py
import streamlit as st
import requests

API_URL = "http://localhost:8000/chat"

st.set_page_config(page_title="RAG Chatbot", page_icon="🤖", layout="wide")
st.title("💬 RAG Chatbot")

# Keep chat history
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# Input box
user_query = st.chat_input("Type your question...")

if user_query:
    # Add user message
    st.session_state["messages"].append({"role": "user", "content": user_query})
    # Add assistant placeholder (None = waiting)
    st.session_state["messages"].append({"role": "assistant", "content": None})
    st.rerun()

# Render conversation
for i, msg in enumerate(st.session_state["messages"]):
    if msg["role"] == "user":
        st.chat_message("user").write(msg["content"])
    else:
        if msg["content"] is None:
            with st.chat_message("assistant"):
                placeholder = st.empty()
                placeholder.markdown("🤖 *Bot is thinking...*")
        else:
            st.chat_message("assistant").write(msg["content"])

# If last assistant is still None → fetch answer
if st.session_state["messages"]:
    last_msg = st.session_state["messages"][-1]
    if last_msg["role"] == "assistant" and last_msg["content"] is None:
        query_text = st.session_state["messages"][-2]["content"]
        try:
            payload = {"query": query_text, "user_email": "demo@lilly.com", "product_id": 1}
            response = requests.post(API_URL, json=payload, timeout=None)

            if response.status_code == 200:
                answer = response.json().get("answer", "⚠️ No response")
            else:
                answer = "❌ Backend error"
        except Exception as e:
            answer = f"❌ Request failed: {str(e)}"

        # Update last assistant message
        st.session_state["messages"][-1]["content"] = answer
        st.rerun()
