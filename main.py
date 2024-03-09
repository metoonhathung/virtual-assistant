import streamlit as st
import speech_recognition as sr
from gtts import gTTS
from streamlit_webrtc import webrtc_streamer, WebRtcMode
from aiortc.contrib.media import MediaRecorder
import tempfile
import os
from dotenv import load_dotenv, find_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.agents import AgentExecutor, create_openai_functions_agent
from tools import search, get_retriever_tool, generate_music, generate_image

_ = load_dotenv(find_dotenv())
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
        return default_value
    else:
        return st.session_state[key]

def get_agent():
    llm = get_key("llm", ChatOpenAI(model="gpt-3.5-turbo", temperature=0))
    tools = get_key("tools", [search, generate_music, generate_image])
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])
    agent = create_openai_functions_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)
    agent_with_chat_history = RunnableWithMessageHistory(
        agent_executor,
        lambda session_id: get_key(session_id, ChatMessageHistory()),
        input_messages_key="input",
        history_messages_key="chat_history",
    )
    return agent_with_chat_history

def natural_language_processing(text):
    agent = get_key("agent", get_agent())
    response = agent.invoke(
        {"input": text},
        config={"configurable": {"session_id": "chat_history"}},
    )
    return response["output"]

def update_llm_model():
    model = st.session_state["model"]
    st.session_state["llm"] = ChatOpenAI(model=model, temperature=0)
    st.session_state["agent"] = get_agent()

def update_retriever_tool():
    uploaded_file = st.session_state["uploaded_file"]
    if uploaded_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(uploaded_file.getvalue())
            retriever_tool = get_retriever_tool(temp_file.name)
        st.session_state["tools"] = [search, generate_music, generate_image, retriever_tool]
    else:
        st.session_state["tools"] = [search, generate_music, generate_image]
    st.session_state["agent"] = get_agent()

def chat_flow():
    chat_text = st.session_state["chat_text"]
    st.session_state["chat_text"] = ""
    print("Human:", chat_text)
    if chat_text:
        llm_text = natural_language_processing(chat_text)
        print("AI:", llm_text)

def voice_flow():
    if (os.path.exists(TMP_WAV)):
        speech_text = automatic_speech_recognition()
        os.remove(TMP_WAV)
        print("Human:", speech_text)
        if speech_text:
            llm_text = natural_language_processing(speech_text)
            print("AI:", llm_text)
            if llm_text and not (llm_text.endswith(".wav") or llm_text.endswith(".png")):
                text_to_speech(llm_text)

def render_chat_history():
    if "chat_history" in st.session_state:
        for index, message in enumerate(st.session_state["chat_history"].messages):
            if index % 2 == 0:
                st.markdown(
                    f'''<div style="background-color: #DCF8C6; padding: 10px; border-radius: 10px; text-align: right;">
                        You: {message.content}
                    </div>''',
                    unsafe_allow_html=True
                )
            else:
                if message.content.endswith(".wav"):
                    st.audio(message.content, format="audio/wav")
                elif message.content.endswith(".png"):
                    st.image(message.content)
                else:
                    st.markdown(
                        f'''<div style="background-color: #EAEAEA; padding: 10px; border-radius: 10px; text-align: left;">
                            AI: {message.content}
                        </div>''',
                        unsafe_allow_html=True
                    )

def record_audio():
    webrtc_ctx = webrtc_streamer(
        key="sendrecv",
        mode=WebRtcMode.SENDRECV,
        in_recorder_factory=lambda: MediaRecorder(TMP_WAV),
        media_stream_constraints={"audio": True, "video": False},
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        async_processing=True,
    )
    if not webrtc_ctx.state.playing:
        voice_flow()
    
def main():
    st.title("Virtual Assistant")
    st.selectbox("Model", ["gpt-3.5-turbo", "gpt-4-turbo-preview"], key="model", on_change=update_llm_model)
    st.text_input("Start Typing", key="chat_text", on_change=chat_flow, placeholder="Answer question, Generate music, Generate image, Analyze document, ...")
    record_audio()
    st.file_uploader("Retrieve PDF information", type=["pdf"], key="uploaded_file", on_change=update_retriever_tool)
    render_chat_history()

if __name__ == "__main__":
    main()
