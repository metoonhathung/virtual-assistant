from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import SQLChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.agents import AgentExecutor, create_openai_functions_agent
from tools import search, generate_music, generate_image, generate_video

SQLITE_URL = "sqlite:///sqlite.db"

def get_agent(tools_dict, room_id):
    static_tools = [search, generate_music, generate_image, generate_video]
    dynamic_tools = list(tools_dict.get(room_id, {}).values())
    tools = static_tools + dynamic_tools
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])
    agent = create_openai_functions_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)
    chat_history = SQLChatMessageHistory(session_id=room_id, connection_string=SQLITE_URL)
    agent_with_chat_history = RunnableWithMessageHistory(
        agent_executor,
        lambda session_id: chat_history,
        input_messages_key="input",
        history_messages_key="chat_history",
    )
    return agent_with_chat_history

def run_agent(tools_dict, room_id, text):
    agent = get_agent(tools_dict, room_id)
    response = agent.invoke({"input": text}, config={"configurable": {"session_id": "chat_history"}})["output"]
    return response
