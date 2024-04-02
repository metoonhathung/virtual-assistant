from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain.pydantic_v1 import BaseModel, Field
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import OllamaEmbeddings, HuggingFaceInferenceAPIEmbeddings
from langchain.tools.retriever import create_retriever_tool
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.tools import tool
from gradio_tools.tools import TextToVideoTool
from dotenv import load_dotenv, find_dotenv
import requests
import tempfile
import os

_ = load_dotenv(find_dotenv())

search = DuckDuckGoSearchRun()
text2video = TextToVideoTool() # ali-vilab/modelscope-damo-text-to-video-synthesis

def get_retriever_tool(file_path, file_name, model):
    loader = PyPDFLoader(file_path)
    pages = loader.load_and_split()
    # embeddings = OllamaEmbeddings(model="mistral:7b") if model == "mistral" else OpenAIEmbeddings(model="gpt-3.5-turbo")
    embeddings = HuggingFaceInferenceAPIEmbeddings(api_key = os.environ["HUGGINGFACEHUB_API_TOKEN"], model_name="sentence-transformers/all-MiniLM-l6-v2") if model == "mistral" else OpenAIEmbeddings(model="gpt-3.5-turbo")
    vector = FAISS.from_documents(pages, embeddings)
    retriever = vector.as_retriever()
    retriever_tool = create_retriever_tool(
        retriever,
        "pdf_document_search",
        f"Search for information about uploaded PDF document {file_name}. For any questions about uploaded PDF document {file_name}, you must use this tool!",
    )
    return retriever_tool

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
    return text2video.langchain.run(desc)
