from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.pydantic_v1 import BaseModel, Field
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.tools.retriever import create_retriever_tool
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.tools import tool, StructuredTool
from dotenv import load_dotenv, find_dotenv
import requests
import uuid
import os
import re
import pandas as pd
from sqlalchemy import create_engine
from supabase import create_client
from langchain.chains import create_sql_query_chain
from langchain_community.utilities import SQLDatabase
from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool

_ = load_dotenv(find_dotenv())
supabase = create_client(os.environ['SUPABASE_URL'], os.environ['SUPABASE_KEY'])
DB_URL = "sqlite:///csv.db"

search = DuckDuckGoSearchRun()

def get_retriever_tool(file_path, file_name, type):
    if type == "pdf":
        loader = PyPDFLoader(file_path)
        pages = loader.load_and_split()
    if type == "url":
        loader = WebBaseLoader(file_path)
        docs = loader.load()
        pages = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200).split_documents(docs)
    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
    vector = FAISS.from_documents(pages, embeddings)
    retriever = vector.as_retriever()
    retriever_tool = create_retriever_tool(
        retriever,
        re.sub(r'[^a-zA-Z0-9_-]', '_', f"{file_name}_document_search"),
        f"Search for information about uploaded {file_name} document",
    )
    return retriever_tool

def get_sql_tool(file_path, file_name):
    engine = create_engine(DB_URL)
    df = pd.read_csv(file_path)
    df.to_sql(file_name, engine, index=False, if_exists="replace")
    db = SQLDatabase(engine=engine)
    llm = ChatOpenAI(model_name="gpt-4o", temperature=0)
    execute_query = QuerySQLDataBaseTool(db=db)
    write_query = create_sql_query_chain(llm, db)
    chain = write_query | execute_query
    def invoke_chain(query: str) -> str:
        return chain.invoke({"question": query})
    tool = StructuredTool.from_function(
        func=invoke_chain,
        name=re.sub(r'[^a-zA-Z0-9_-]', '_', f"{file_name}_document_search"),
        description=f"Search for information about uploaded {file_name} document",
    )
    return tool

def call_inference_api(model, inputs, type):
    url = f"https://api-inference.huggingface.co/models/{model}"
    headers = {"Authorization": f"Bearer {os.environ['HUGGINGFACEHUB_API_TOKEN']}"}
    payload = {"inputs": inputs}
    response = requests.post(url, headers=headers, json=payload)
    if response.status_code == 200:
        bucket = "virtual-assistant"
        path = f"{uuid.uuid4()}.{type.split('/')[1]}"
        response = supabase.storage.from_(bucket).upload(
            path=path,
            file=response.content,
            file_options={"content-type": type}
        )
        if response.path:
            public_url = supabase.storage.from_(bucket).get_public_url(response.path)
            return public_url
    else:
        return None

class ImageGenerationInput(BaseModel):
    desc: str = Field(description="Description of image to generate")

@tool("image-generation-tool", args_schema=ImageGenerationInput)
def generate_image(desc: str) -> str:
    """Generate image online and return a string containing .jpeg file path"""
    return call_inference_api("stabilityai/stable-diffusion-xl-base-1.0", desc, "image/jpeg")

class MusicGenerationInput(BaseModel):
    desc: str = Field(description="Description of music to generate")

@tool("music-generation-tool", args_schema=MusicGenerationInput)
def generate_music(desc: str) -> str:
    """Generate music online and return a string containing .wav file path"""
    return call_inference_api("facebook/musicgen-small", desc, "audio/wav")

class VideoGenerationInput(BaseModel):
    desc: str = Field(description="Description of video to generate")

@tool("video-generation-tool", args_schema=VideoGenerationInput)
def generate_video(desc: str) -> str:
    """Generate video online and return a string containing .mp4 file path"""
    return call_inference_api("ali-vilab/modelscope-damo-text-to-video-synthesis", desc, "video/mp4")
