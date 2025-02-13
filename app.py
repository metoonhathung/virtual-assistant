from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv, find_dotenv
from pydantic import BaseModel
from typing import Optional
import sqlite3
import tempfile
import shutil
from agent import run_agent, SQLITE_URL
from tools import get_retriever_tool, get_sql_tool
from langchain_community.chat_message_histories import SQLChatMessageHistory

class ChatRequest(BaseModel):
    text: str

_ = load_dotenv(find_dotenv())
tools_dict = {}

app = FastAPI()

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

@app.get("/")
async def root():
    return {"message": "OK"}

@app.get("/chat")
async def get_chats():
    conn = sqlite3.connect("sqlite.db")
    cursor = conn.cursor()
    cursor.execute("SELECT session_id, COUNT(*) AS row_count FROM message_store GROUP BY session_id")
    results = cursor.fetchall()
    rooms = [{"Room ID": session_id, "Messages": row_count} for session_id, row_count in results]
    cursor.close()
    conn.close()
    return rooms

@app.get("/chat/{room_id}")
async def get_chat(room_id: str):
    chat_history = SQLChatMessageHistory(session_id=room_id, connection_string=SQLITE_URL)
    messages = [{ "type": m.type, "content": m.content } for m in chat_history.messages]
    return messages

@app.delete("/chat/{room_id}")
async def delete_chat(room_id: str):
    chat_history = SQLChatMessageHistory(session_id=room_id, connection_string=SQLITE_URL)
    chat_history.clear()
    tools_dict.pop(room_id, None)
    return { "message": f"Chat history for room {room_id} has been deleted" }

@app.post("/chat/{room_id}")
async def post_chat(room_id: str, payload: ChatRequest):
    response = run_agent(tools_dict, room_id, payload.text)
    return { "response": response }

@app.get("/tool/{room_id}")
async def get_tool(room_id: str):
    static_tools = ["search", "generate_music", "generate_image", "generate_video"]
    dynamic_tools = list(tools_dict.get(room_id, {}).keys())
    tools = static_tools + dynamic_tools
    names = [{"Tools": tool} for tool in tools]
    return names

@app.post("/tool/{room_id}")
async def post_tool(
    room_id: str,
    file: Optional[UploadFile] = File(None),
    type: str = Form(...),
    url: Optional[str] = Form(None)
):
    if type == "csv":
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            shutil.copyfileobj(file.file, temp_file)
            tool = get_sql_tool(temp_file.name, file.filename)
    elif type == "pdf":
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            shutil.copyfileobj(file.file, temp_file)
            tool = get_retriever_tool(temp_file.name, file.filename, "pdf")
    elif type == "url":
        tool = get_retriever_tool(url, url, "url")
    global tools_dict
    if room_id not in tools_dict:
        tools_dict[room_id] = {}
    tools_dict[room_id][tool.name] = tool
    return { "tool_name": tool.name }
