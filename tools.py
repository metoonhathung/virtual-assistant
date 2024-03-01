from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain.pydantic_v1 import BaseModel, Field
from langchain_openai import OpenAIEmbeddings
from langchain.tools.retriever import create_retriever_tool
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.tools import tool
from openai import OpenAI
from dotenv import load_dotenv, find_dotenv
import requests
import tempfile
import os

_ = load_dotenv(find_dotenv())

search = TavilySearchResults()

def get_retriever_tool(file_path):
    loader = PyPDFLoader(file_path)
    pages = loader.load_and_split()
    vector = FAISS.from_documents(pages, OpenAIEmbeddings())
    retriever = vector.as_retriever()
    retriever_tool = create_retriever_tool(
        retriever,
        "pdf_document_search",
        "Search for information about uploaded PDF document. For any questions about uploaded PDF document, you must use this tool!",
    )
    return retriever_tool

class ImageGenerationInput(BaseModel):
    desc: str = Field(description="Description of image to generate")

@tool("image-generation-tool", args_schema=ImageGenerationInput, return_direct=True)
def generate_image(desc: str) -> str:
    """Generate image online and return a string containing .png file path"""
    client = OpenAI()
    response = client.images.generate(model="dall-e-3", prompt=desc, size="1024x1024", quality="standard", n=1)
    url = response.data[0].url
    response = requests.get(url)
    if response.status_code == 200:
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_file:
            temp_file.write(response.content)
            temp_file_path = temp_file.name
            return temp_file_path
    else:
        return None

class MusicGenerationInput(BaseModel):
    desc: str = Field(description="Description of music to generate")

@tool("music-generation-tool", args_schema=MusicGenerationInput, return_direct=True)
def generate_music(desc: str) -> str:
    """Generate music online and return a string containing .wav file path"""
    url = "https://api-inference.huggingface.co/models/facebook/musicgen-small"
    headers = { "Authorization": f'Bearer {os.environ["HUGGINGFACE_API_TOKEN"]}' }
    data = { "inputs": desc }
    response = requests.post(url, json=data, headers=headers)
    if response.status_code == 200:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
            temp_file.write(response.content)
            temp_file_path = temp_file.name
            return temp_file_path
    else:
        return None
