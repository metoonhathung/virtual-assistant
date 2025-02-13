import streamlit as st
import requests
from dotenv import load_dotenv, find_dotenv

_ = load_dotenv(find_dotenv())
# host = "http://localhost"
host = "https://virtual-assistant-ri1b.onrender.com"

def natural_language_processing():
    model = st.session_state["model"]
    room_id = st.session_state["room_id"]
    text = st.session_state["text"]
    if not text: return
    url = f"{host}{'/v2' if model == 'langgraph' else ''}/chat/{room_id}"
    headers = { "Content-Type": "application/json" }
    data = { "room_id": room_id, "text": text }
    response = requests.post(url, json=data, headers=headers)
    llm_text = response.json()["response"]
    return llm_text
    
def update_retriever_tool(type):
    csv_file = st.session_state["csv"]
    pdf_file = st.session_state["pdf"]
    url_file = st.session_state["url"].strip()
    room_id = st.session_state["room_id"]
    if type == "csv" and csv_file:
        files = { "file": (csv_file.name, csv_file, csv_file.type) }
        data = { "type": "csv" }
    elif type == "pdf" and pdf_file:
        files = { "file": (pdf_file.name, pdf_file, pdf_file.type) }
        data = { "type": "pdf" }
    elif type == "url" and url_file:
        files = {}
        data = { "type": "url", "url": url_file }
    else:
        return
    url = f"{host}/tool/{room_id}"
    response = requests.post(url, files=files, data=data)

def delete_chat_history():
    room_id = st.session_state["room_id"]
    model = st.session_state["model"]
    url = f"{host}{'/v2' if model == 'langgraph' else ''}/chat/{room_id}"
    response = requests.delete(url)
    st.session_state["room_id"] = ""

def render_chat_history():
    room_id = st.session_state["room_id"]
    model = st.session_state["model"]
    url = f"{host}{'/v2' if model == 'langgraph' else ''}/chat/{room_id}"
    response = requests.get(url)
    messages = response.json()
    for message in messages:
        st.chat_message(message["type"]).write(message["content"])

def render_rooms():
    model = st.session_state["model"]
    url = f"{host}{'/v2' if model == 'langgraph' else ''}/chat"
    response = requests.get(url)
    rooms = response.json()
    st.table(rooms)

def render_tools():
    room_id = st.session_state["room_id"]
    if not room_id: return
    url = f"{host}/tool/{room_id}"
    response = requests.get(url)
    tools = response.json()
    st.table(tools)

def room_sidebar():
    with st.sidebar:
        st.selectbox("Model", ["langchain", "langgraph"], key="model")
        st.text_input("Room ID", key="room_id")
        render_rooms()
        render_tools()
        if st.session_state["room_id"]:
            with st.expander("Settings"):
                st.text_input("URL Retriever", key="url", on_change=update_retriever_tool, kwargs={"type": "url"})
                st.file_uploader("CSV Retriever", key="csv", on_change=update_retriever_tool, kwargs={"type": "csv"}, type=["csv"])
                st.file_uploader("PDF Retriever", key="pdf", on_change=update_retriever_tool, kwargs={"type": "pdf"}, type=["pdf"])
                st.button("Delete chat", on_click=delete_chat_history, type="primary", use_container_width=True)
    if not st.session_state["room_id"]:
        st.info("Enter room ID to continue")
        st.stop()
    
def main():
    st.set_page_config(page_title="Virtual Assistant", page_icon="random")
    st.title("Virtual Assistant")
    room_sidebar()
    st.chat_input("Search web, Generate music/image/video, Analyze CSV/PDF/URL, ...", key="text", on_submit=natural_language_processing)
    render_chat_history()

if __name__ == "__main__":
    main()
