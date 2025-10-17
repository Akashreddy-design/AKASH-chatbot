from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph.message import add_messages
from dotenv import load_dotenv
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.tools import tool
import sqlite3
import requests

# -------------------- Load environment --------------------
load_dotenv()

# -------------------- Initialize model --------------------
import os

api_key = os.environ["OPENAI_API_KEY"]
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7, openai_api_key=api_key)


# -------------------- Chat State --------------------
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

# -------------------- Tools --------------------
search1 = DuckDuckGoSearchRun()

@tool
def basic_calculator(first_number: float, second_number: float, operation: str) -> dict:
    """Performs basic arithmetic operations."""
    try:
        if operation == "add":
            result = first_number + second_number
        elif operation == "subtract":
            result = first_number - second_number
        elif operation == "multiply":
            result = first_number * second_number
        elif operation == "divide":
            if second_number == 0:
                return {"error": "Division by zero not allowed"}
            result = first_number / second_number
        else:
            return {"error": "Unsupported operation"}
    except Exception as e:
        return {"error": str(e)}
    return {
        "first_number": first_number,
        "second_number": second_number,
        "operation": operation,
        "result": result,
    }

@tool
def api_calculate(expression: str) -> dict:
    """Calculate mathematical expressions using MathJS API."""
    url = f"https://api.mathjs.org/v4/?expr={expression}"
    r = requests.get(url)
    return {"result": r.text}


@tool
def fetch_stock_price(symbol: str) -> dict:
    """Fetch current stock price for a symbol using Alpha Vantage API."""
    url = (
        f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}"
        f"&apikey=IPJCQ23SVBHPEMAD"
    )
    try:
        response = requests.get(url, timeout=10)
        data = response.json()
        return data.get("Global Quote", {"error": "No data found"})
    except Exception as e:
        return {"error": str(e)}

tools = [search1, basic_calculator, fetch_stock_price,api_calculate]
llm_with_tools = llm.bind_tools(tools)

# -------------------- Chat Node --------------------
def chat_node(state: ChatState):
    user_messages = state["messages"]

    # ✅ Always prepend your custom identity instruction
    system_msg = SystemMessage(
        content=(
            "You are AkashBot, an intelligent AI assistant created by Akash Reddy Kontham. "
            "You are powered by OpenAI’s GPT-4o-mini model and integrated using LangGraph and Streamlit. "
            "Always refer to yourself as AkashBot, not ChatGPT or OpenAI. "
            "Be friendly, concise, and professional."
            "Whenever you perform calculations, return plain numeric answers only. "
            "Do not use LaTeX, markdown, or formatting. "
        )
    )

    # ✅ Send messages explicitly (so your system message stays on top)
    response = llm_with_tools.invoke([system_msg] + user_messages)
    return {"messages": [response]}


# -------------------- SQLite Checkpointer --------------------
conn = sqlite3.connect("chatbot.db", check_same_thread=False)
checkpointer = SqliteSaver(conn=conn)

# -------------------- Tool Node --------------------
tool_node = ToolNode(tools=tools)

# -------------------- Build Graph --------------------
graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_node("tools", tool_node)

graph.add_edge(START, "chat_node")
graph.add_conditional_edges("chat_node", tools_condition)
graph.add_edge("tools", "chat_node")
graph.add_edge("chat_node", END)

# Compile the chatbot graph
chatbot = graph.compile(checkpointer=checkpointer)

# -------------------- Helper: Retrieve All Chat Threads --------------------
def retrieve_all_threads():
    all_threads = set()
    for checkpoint in checkpointer.list(None):
        all_threads.add(checkpoint.config["configurable"]["thread_id"])
    return list(all_threads)

# -------------------- Example Run --------------------
if __name__ == "__main__":
    print("🤖 AkashBot is running...")

    result = chatbot.invoke(
        {"messages": [HumanMessage(content="Who created you?")]},
        config={"configurable": {"thread_id": "test_thread_1"}},
    )

    print("🧠 Response:", result["messages"][-1].content)
