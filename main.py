import streamlit as st
import speech_recognition as sr
from gtts import gTTS
from streamlit_webrtc import webrtc_streamer, WebRtcMode
from aiortc.contrib.media import MediaRecorder
import tempfile
import os
import sqlite3
from dotenv import load_dotenv, find_dotenv
from langchain_openai import ChatOpenAI
from langchain_community.llms import Ollama, HuggingFaceEndpoint
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import ChatMessageHistory, SQLChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.agents import AgentExecutor, create_openai_functions_agent, create_json_chat_agent
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler
from langchain import hub
from tools import search, get_sql_tool, get_retriever_tool, generate_music, generate_image, generate_video
from constants import *

_ = load_dotenv(find_dotenv())
SQLITE_URL = "sqlite:///sqlite.db"
TMP_WAV = "tmp.wav"

def automatic_speech_recognition():
    recognizer = sr.Recognizer()
    # microphone = sr.Microphone()
    # with microphone as source:
    #     recognizer.adjust_for_ambient_noise(source)
    #     audio = recognizer.listen(source)
    with sr.AudioFile(TMP_WAV) as source:
        audio = recognizer.record(source)
    try:
        speech_text = recognizer.recognize_google(audio)
        return speech_text
    except sr.UnknownValueError:
        return ""
    except sr.RequestError as e:
        return ""

def text_to_speech(text):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_file:
        tts = gTTS(text=text, lang="en", slow=False)
        tts.write_to_fp(temp_file)
        temp_file.flush()
        # os.system(f"mpg123 {temp_file.name}")
        st.audio(temp_file.name, format="audio/mpeg")

def get_key(key, default_value):
    if key not in st.session_state:
        st.session_state[key] = default_value
    return st.session_state[key]
    
def get_gpt_llm():
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    tools = get_key("tools", [search, generate_music, generate_image, generate_video])
    # prompt = hub.pull("hwchase17/openai-functions-agent")
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])
    agent = create_openai_functions_agent(llm, tools, prompt)
    return agent

def get_mistral_llm():
    # llm = Ollama(model="mistral:7b", temperature=0)
    llm = HuggingFaceEndpoint(repo_id="mistralai/Mistral-7B-Instruct-v0.2", temperature=0.01)
    tools = get_key("tools", [search, generate_music, generate_image, generate_video])
    # prompt = hub.pull("hwchase17/react-chat-json")
    prompt = react_chat_json_prompt
    agent = create_json_chat_agent(llm, tools, prompt)
    return agent

def get_agent():
    room_id = get_key("room_id", "")
    tools = get_key("tools", [search, generate_music, generate_image, generate_video])
    agent = get_key("llm", get_mistral_llm())
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)
    agent_with_chat_history = RunnableWithMessageHistory(
        agent_executor,
        # lambda session_id: get_key(session_id, ChatMessageHistory()),
        lambda session_id: get_key(session_id, SQLChatMessageHistory(session_id=room_id, connection_string=SQLITE_URL)),
        input_messages_key="input",
        history_messages_key="chat_history",
    )
    return agent_with_chat_history

def natural_language_processing(text):
    with st.sidebar:
        st_callback = StreamlitCallbackHandler(st.container())
    agent = get_key("agent", get_agent())
    response = agent.invoke({"input": text}, config={"configurable": {"session_id": "chat_history"}, "callbacks": [st_callback]})["output"]
    return response

def update_llm_model():
    model = st.session_state["model"]
    st.session_state["llm"] = get_mistral_llm() if model == "mistral" else get_gpt_llm()
    st.session_state["agent"] = get_agent()

def update_retriever_tool(type):
    model = get_key("model", "mistral")
    tool_dict = get_key("tool_dict", {})
    updated_tool_dict = {}
    csv_files = st.session_state["csv_files"]
    pdf_files = st.session_state["pdf_files"]
    urls = [url.strip() for url in st.session_state["urls"].split("\n") if url.strip()]
    for csv_file in csv_files:
        if csv_file.name not in tool_dict:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as temp_file:
                temp_file.write(csv_file.getvalue())
            updated_tool_dict[csv_file.name] = get_sql_tool(temp_file.name, csv_file.name, model)
        else:
            updated_tool_dict[csv_file.name] = tool_dict[csv_file.name]
    for pdf_file in pdf_files:
        if pdf_file.name not in tool_dict:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
                temp_file.write(pdf_file.getvalue())
            updated_tool_dict[pdf_file.name] = get_retriever_tool(temp_file.name, pdf_file.name, "pdf", model)
        else:
            updated_tool_dict[pdf_file.name] = tool_dict[pdf_file.name]
    for url in urls:
        if url not in tool_dict:
            updated_tool_dict[url] = get_retriever_tool(url, url, "url", model)
        else:
            updated_tool_dict[url] = tool_dict[url]
    st.session_state["tool_dict"] = updated_tool_dict
    st.session_state["tools"] = [search, generate_music, generate_image, generate_video] + [value for value in updated_tool_dict.values()]
    st.session_state["llm"] = get_mistral_llm() if model == "mistral" else get_gpt_llm()
    st.session_state["agent"] = get_agent()

def update_chat_history():
    room_id = st.session_state["room_id"]
    st.session_state["chat_history"] = SQLChatMessageHistory(session_id=room_id, connection_string=SQLITE_URL)
    st.session_state["agent"] = get_agent()

def delete_chat_history():
    room_id = get_key("room_id", "")
    chat_history = get_key("chat_history", SQLChatMessageHistory(session_id=room_id, connection_string=SQLITE_URL))
    chat_history.clear()
    st.session_state["room_id"] = ""

def chat_flow():
    chat_text = st.chat_input("Answer question, Generate music/image/video, Analyze CSV/PDF/URL document, ...")
    if chat_text:
        llm_text = natural_language_processing(chat_text)
        # chat_text = ""

def voice_flow():
    webrtc_ctx = webrtc_streamer(
        key="webrtc_ctx",
        mode=WebRtcMode.SENDRECV,
        in_recorder_factory=lambda: MediaRecorder(TMP_WAV),
        media_stream_constraints={"audio": True, "video": False},
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    )
    if (not webrtc_ctx.state.playing) and os.path.exists(TMP_WAV):
        speech_text = automatic_speech_recognition()
        os.remove(TMP_WAV)
        if speech_text:
            llm_text = natural_language_processing(speech_text)
            if llm_text and not (llm_text.endswith(".wav") or llm_text.endswith(".jpeg") or llm_text.endswith(".mp4")):
                text_to_speech(llm_text)

def render_chat_history():
    room_id = get_key("room_id", "")
    chat_history = get_key("chat_history", SQLChatMessageHistory(session_id=room_id, connection_string=SQLITE_URL))
    for index, message in enumerate(chat_history.messages):
        if index % 2 == 0:
            st.chat_message(message.type).write(message.content)
        else:
            if message.content.endswith(".wav"):
                with st.chat_message(message.type):
                    st.write("Here is your music:")
                    st.audio(message.content, format="audio/wav")
            elif message.content.endswith(".jpeg"):
                with st.chat_message(message.type):
                    st.write("Here is your image:")
                    st.image(message.content)
            elif message.content.endswith(".mp4"):
                with st.chat_message(message.type):
                    st.write("Here is your video:")
                    st.video(message.content, format="video/mp4")
            else:
                st.chat_message(message.type).write(message.content)

def render_rooms(): # Table "message_store" has 3 columns: ["id", "session_id", "message"].
    # conn = st.connection("sqlite_db", type="sql", url=SQLITE_URL)
    # room_ids = conn.query("SELECT session_id, COUNT(*) AS row_count FROM message_store GROUP BY session_id")
    # st.dataframe(room_ids)
    conn = sqlite3.connect("sqlite.db")
    cursor = conn.cursor()
    cursor.execute("SELECT session_id, COUNT(*) AS row_count FROM message_store GROUP BY session_id")
    results = cursor.fetchall()
    room_ids = [{"Room ID": session_id, "Messages": row_count} for session_id, row_count in results]
    cursor.close()
    conn.close()
    st.table(room_ids)

def room_sidebar():
    with st.sidebar:
        st.text_input("Room ID", key="room_id", on_change=update_chat_history)
        render_rooms()
        if st.session_state["room_id"]:
            with st.expander("Settings"):
                st.selectbox("Model", ["mistral", "gpt"], key="model", on_change=update_llm_model)
                st.text_area("URLs Retriever", key="urls", on_change=update_retriever_tool, kwargs={"type": "url"})
                st.file_uploader("CSVs Retriever", key="csv_files", on_change=update_retriever_tool, kwargs={"type": "csv"}, type=["csv"], accept_multiple_files=True)
                st.file_uploader("PDFs Retriever", key="pdf_files", on_change=update_retriever_tool, kwargs={"type": "pdf"}, type=["pdf"], accept_multiple_files=True)
                st.button("Delete chat", on_click=delete_chat_history, type="primary", use_container_width=True)
            voice_flow()
    if not st.session_state["room_id"]:
        st.info("Enter existing or new room ID to continue")
        st.stop()
    
def main():
    st.set_page_config(page_title="Virtual Assistant", page_icon="random")
    st.title("Virtual Assistant")
    room_sidebar()
    chat_flow()
    render_chat_history()

if __name__ == "__main__":
    main()
