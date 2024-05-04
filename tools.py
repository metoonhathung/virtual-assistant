from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.pydantic_v1 import BaseModel, Field
from langchain_openai import OpenAI, OpenAIEmbeddings
from langchain_community.embeddings import OllamaEmbeddings, HuggingFaceInferenceAPIEmbeddings
from langchain.tools.retriever import create_retriever_tool
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.tools import tool, StructuredTool
from gradio_tools.tools import TextToVideoTool
from dotenv import load_dotenv, find_dotenv
import requests
import tempfile
import os
import hashlib
from gptcache import Cache
from gptcache.adapter.api import init_similar_cache
from langchain.cache import InMemoryCache, GPTCache
from langchain.globals import set_llm_cache
from langchain.storage import InMemoryByteStore
from langchain.embeddings import CacheBackedEmbeddings
import pandas as pd
from sqlalchemy import create_engine
from langchain.chains import create_sql_query_chain
from langchain_community.utilities import SQLDatabase
from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool
from langchain_community.llms import HuggingFaceEndpoint

_ = load_dotenv(find_dotenv())
# set_llm_cache(InMemoryCache())
def init_gptcache(cache_obj: Cache, llm: str):
    hashed_llm = hashlib.sha256(llm.encode()).hexdigest()
    init_similar_cache(cache_obj=cache_obj, data_dir=f"similar_cache_{hashed_llm}")
# set_llm_cache(GPTCache(init_gptcache))
store = InMemoryByteStore()

search = DuckDuckGoSearchRun()

def get_retriever_tool(file_path, file_name, type, model):
    if type == "pdf":
        loader = PyPDFLoader(file_path)
        pages = loader.load_and_split()
    if type == "url":
        loader = WebBaseLoader(file_path)
        docs = loader.load()
        pages = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200).split_documents(docs)
    # underlying_embeddings = OllamaEmbeddings(model="mistral:7b")
    underlying_embeddings = HuggingFaceInferenceAPIEmbeddings(api_key=os.environ["HUGGINGFACEHUB_API_TOKEN"], model_name="sentence-transformers/all-MiniLM-l6-v2") if model == "mistral" else OpenAIEmbeddings(model="gpt-3.5-turbo")
    cached_embedder = CacheBackedEmbeddings.from_bytes_store(underlying_embeddings, store, namespace="mistral-embed" if model == "mistral" else underlying_embeddings.model)
    vector = FAISS.from_documents(pages, cached_embedder)
    retriever = vector.as_retriever()
    retriever_tool = create_retriever_tool(
        retriever,
        f"{file_name}_document_search",
        f"Search for information about uploaded {file_name} document",
    )
    return retriever_tool

def get_sql_tool(file_path, file_name, model):
    engine = create_engine("sqlite:///csv.db")
    df = pd.read_csv(file_path)
    df.to_sql(file_name, engine, index=False, if_exists="replace")
    db = SQLDatabase(engine=engine)
    llm = HuggingFaceEndpoint(repo_id="mistralai/Mistral-7B-Instruct-v0.2", temperature=0.01) if model == "mistral" else OpenAI(model_name="gpt-3.5-turbo", temperature=0)
    execute_query = QuerySQLDataBaseTool(db=db)
    write_query = create_sql_query_chain(llm, db)
    chain = write_query | execute_query
    def invoke_chain(query: str) -> str:
        return chain.invoke({"question": query})
    tool = StructuredTool.from_function(
        func=invoke_chain,
        name=f"{file_name}_document_search",
        description=f"Search for information about uploaded {file_name} document",
    )
    return tool

def call_inference_api(model, inputs, extension):
    url = f"https://api-inference.huggingface.co/models/{model}"
    headers = {"Authorization": f"Bearer {os.environ['HUGGINGFACEHUB_API_TOKEN']}"}
    payload = {"inputs": inputs}
    response = requests.post(url, headers=headers, json=payload)
    if response.status_code == 200:
        with tempfile.NamedTemporaryFile(suffix=extension, delete=False) as temp_file:
            temp_file.write(response.content)
            return temp_file.name
    else:
        return None

class ImageGenerationInput(BaseModel):
    desc: str = Field(description="Description of image to generate")

@tool("image-generation-tool", args_schema=ImageGenerationInput, return_direct=True)
def generate_image(desc: str) -> str:
    """Generate image online and return a string containing .jpeg file path"""
    return call_inference_api("stabilityai/stable-diffusion-xl-base-1.0", desc, ".jpeg")

class MusicGenerationInput(BaseModel):
    desc: str = Field(description="Description of music to generate")

@tool("music-generation-tool", args_schema=MusicGenerationInput, return_direct=True)
def generate_music(desc: str) -> str:
    """Generate music online and return a string containing .wav file path"""
    return call_inference_api("facebook/musicgen-small", desc, ".wav")

class VideoGenerationInput(BaseModel):
    desc: str = Field(description="Description of video to generate")

@tool("video-generation-tool", args_schema=VideoGenerationInput, return_direct=True)
def generate_video(desc: str) -> str:
    """Generate video online and return a string containing .mp4 file path"""
    text2video = TextToVideoTool() # ali-vilab/modelscope-damo-text-to-video-synthesis
    return text2video.langchain.run(desc)
