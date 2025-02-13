from langchain_openai import ChatOpenAI
from langgraph_supervisor import create_supervisor
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore
from tools import search, generate_music, generate_image, generate_video

memory = InMemorySaver()
store = InMemoryStore()

def get_graph(tools_dict, room_id):
    model = ChatOpenAI(model="gpt-4o")
    web_scraper_agent = create_react_agent(
        model=model,
        tools=[search],
        name="web_scraper",
        prompt="You are a web scraper."
    )
    media_generator_agent = create_react_agent(
        model=model,
        tools=[generate_image, generate_music, generate_video],
        name="media_generator",
        prompt="You are a media generator."
    )
    doc_tools = list(tools_dict.get(room_id, {}).values())
    document_searcher_agent = create_react_agent(
        model=model,
        tools=doc_tools,
        name="document_searcher",
        prompt="You are a document searcher."
    )
    workflow = create_supervisor(
        [web_scraper_agent, media_generator_agent, document_searcher_agent],
        output_mode="full_history",
        model=model,
        prompt=(
            "You are a team supervisor managing a web scraper, a media generator, and a document searcher."
            "Use web_scraper to help you find information on the web."
            "Use media_generator to generate media content such as images, music, and videos."
            "Use document_searcher to search for information in uploaded documents (CSV, PDF, webpage)."
        )
    )
    app = workflow.compile(
        checkpointer=memory,
        store=store
    )
    return app

def run_graph(tools_dict, room_id, text):
    global memory
    graph = get_graph(tools_dict, room_id)
    config = {"configurable": {"thread_id": room_id}}
    response = graph.invoke({"messages": [("user", text)]}, config)
    return response["messages"][-1].content
