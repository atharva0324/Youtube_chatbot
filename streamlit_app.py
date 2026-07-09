import streamlit as st
import requests

BACKEND_URL = "http://localhost:8000"

st.set_page_config(page_title="YouTube RAG Chatbot", page_icon="🎥")
st.title("🎥 YouTube Video Chatbot")

if "video_id" not in st.session_state:
    st.session_state.video_id = None
if "messages" not in st.session_state:
    st.session_state.messages = []

video_url = st.text_input("Paste a YouTube video link:")

if st.button("Load Video"):
    with st.spinner("Fetching transcript and building index..."):
        response = requests.post(f"{BACKEND_URL}/ingest", json={"video_url": video_url})
        if response.status_code == 200:
            data = response.json()
            st.session_state.video_id = data["video_id"]
            st.session_state.messages = []
            st.success(f"Video loaded ({data['status']}). Ask away!")
        else:
            st.error(response.json().get("detail", "Failed to load video"))

if st.session_state.video_id:
    st.divider()
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    question = st.chat_input("Ask something about the video...")
    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.write(question)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = requests.post(f"{BACKEND_URL}/ask", json={
                    "video_id": st.session_state.video_id,
                    "question": question
                })
                if response.status_code == 200:
                    answer = response.json()["answer"]
                else:
                    answer = response.json().get("detail", "Something went wrong.")
                st.write(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})