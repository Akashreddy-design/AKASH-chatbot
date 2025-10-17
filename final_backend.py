# final_backend.py
from langgraph.graph import StateGraph, START, END
import streamlit as st
from typing import TypedDict, Annotated, List, Dict, Any, Tuple
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
import time
import os
from pathlib import Path

# -------------------- Load environment (local fallback) --------------------
load_dotenv()

# -------------------- Model factory --------------------
def make_llm():
    # prefer streamlit secrets (when deployed), fallback to environment variable
    api_key = None
    try:
        api_key = st.secrets.get("OPENAI_API_KEY")  # works in cloud
    except Exception:
        pass
    if not api_key:
        api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        # raise error at runtime when used so developer knows
        raise RuntimeError(
            "OpenAI API key not found. Set OPENAI_API_KEY in env or Streamlit secrets."
        )

    return ChatOpenAI(model="gpt-4o-mini", temperature=0.7, openai_api_key=api_key)

# -------------------- Chat State --------------------
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

# -------------------- Tools --------------------
search1 = DuckDuckGoSearchRun()

@tool
def basic_calculator(first_number: str, second_number: str, operation: str) -> dict:
    """Performs arithmetic using Python ints for big integers. Inputs are strings to preserve precision."""
    try:
        # try integer first
        a = int(first_number)
        b = int(second_number)
        if operation == "add":
            result = a + b
        elif operation == "subtract":
            result = a - b
        elif operation == "multiply":
            result = a * b
        elif operation == "divide":
            if b == 0:
                return {"error": "Division by zero not allowed"}
            # return float division as string
            result = a / b
        else:
            return {"error": "Unsupported operation"}
        return {"first_number": first_number, "second_number": second_number, "operation": operation, "result": str(result)}
    except ValueError:
        # fallback to float
        try:
            a = float(first_number)
            b = float(second_number)
            if operation == "add":
                result = a + b
            elif operation == "subtract":
                result = a - b
            elif operation == "multiply":
                result = a * b
            elif operation == "divide":
                if b == 0:
                    return {"error": "Division by zero not allowed"}
                result = a / b
            else:
                return {"error": "Unsupported operation"}
            return {"first_number": first_number, "second_number": second_number, "operation": operation, "result": str(result)}
        except Exception as e:
            return {"error": str(e)}
    except Exception as e:
        return {"error": str(e)}

@tool
def api_calculate(expression: str) -> dict:
    """Calculate mathematical expressions using MathJS API (useful for floats & big expressions)."""
    url = f"https://api.mathjs.org/v4/?expr={requests.utils.requote_uri(expression)}"
    r = requests.get(url, timeout=10)
    return {"result": r.text}

@tool
def fetch_stock_price(symbol: str) -> dict:
    """Fetch current stock price for a symbol using Alpha Vantage API."""
    # NOTE: For production, move API key to secrets and don't hardcode.
    alpha_key = os.getenv("ALPHA_VANTAGE_KEY", None)
    url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}&apikey={alpha_key or 'demo'}"
    try:
        response = requests.get(url, timeout=10)
        data = response.json()
        return data.get("Global Quote", {"error": "No data found"})
    except Exception as e:
        return {"error": str(e)}

tools = [search1, basic_calculator, fetch_stock_price, api_calculate]

# -------------------- Per-session chatbot cache --------------------
# We'll compile one graph/checkpointer per session_id to isolate threads.
_CHATBOTS: Dict[str, Any] = {}  # session_id -> {"chatbot": compiled, "db_path": str, "checkpointer": obj}

# global metrics DB (aggregated)
_METRICS_DB = "metrics.db"
Path(_METRICS_DB).parent.mkdir(parents=True, exist_ok=True)
# ensure metrics table
_conn_metrics = sqlite3.connect(_METRICS_DB, check_same_thread=False)
_cur = _conn_metrics.cursor()
_cur.execute(
    """
    CREATE TABLE IF NOT EXISTS metrics (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id TEXT,
        timestamp INTEGER,
        latency_ms REAL,
        token_count INTEGER
    )
    """
)
_conn_metrics.commit()

def _log_metrics(session_id: str, latency_ms: float, token_count: int | None):
    cur = _conn_metrics.cursor()
    cur.execute(
        "INSERT INTO metrics (session_id, timestamp, latency_ms, token_count) VALUES (?, ?, ?, ?)",
        (session_id, int(time.time()), latency_ms, token_count),
    )
    _conn_metrics.commit()

def _get_llm_for_session(session_id: str):
    # create one ChatOpenAI instance per process — they are lightweight
    return make_llm()

def _build_graph_for_session(session_id: str):
    """
    Create or reuse compiled chatbot for session. Each session gets its own SqliteSaver checkpoint file.
    """
    if session_id in _CHATBOTS:
        return _CHATBOTS[session_id]["chatbot"], _CHATBOTS[session_id]["checkpointer"]

    # create session specific DB path
    db_path = f"chatbot_{session_id}.db"
    conn = sqlite3.connect(db_path, check_same_thread=False)
    checkpointer = SqliteSaver(conn=conn)

    # build llm bound with tools
    llm = _get_llm_for_session(session_id)
    llm_with_tools = llm.bind_tools(tools)

    # define chat_node and nodes similar to single-graph version but scoped closures
    def chat_node(state: ChatState):
        user_messages = state["messages"]

        # system message forcing identity and numeric format
        system_msg = SystemMessage(
            content=(
                "You are AkashBot, an intelligent AI assistant created by Akash Reddy Kontham. "
                "You are powered by OpenAI’s GPT-4o-mini model and integrated using LangGraph and Streamlit. "
                "Always refer to yourself as AkashBot, not ChatGPT or OpenAI. "
                "Be friendly, concise, and professional. "
                "Whenever you perform calculations, return plain numeric answers only. "
                "Do not use LaTeX, markdown, or formatting."
            )
        )

        response = llm_with_tools.invoke([system_msg] + user_messages)
        return {"messages": [response]}

    tool_node = ToolNode(tools=tools)

    graph = StateGraph(ChatState)
    graph.add_node("chat_node", chat_node)
    graph.add_node("tools", tool_node)
    graph.add_edge(START, "chat_node")
    graph.add_conditional_edges("chat_node", tools_condition)
    graph.add_edge("tools", "chat_node")
    graph.add_edge("chat_node", END)

    compiled = graph.compile(checkpointer=checkpointer)

    _CHATBOTS[session_id] = {"chatbot": compiled, "checkpointer": checkpointer, "db_path": db_path}
    return compiled, checkpointer

# -------------------- Public API for frontend --------------------

def get_compiled_chatbot(session_id: str):
    """Return compiled chatbot executor for given session_id (creates if needed)."""
    chatbot, checkpointer = _build_graph_for_session(session_id)
    return chatbot

def stream_chat(session_id: str, messages: List[BaseMessage], config: dict):
    """
    Stream wrapper that yields (message_chunk, metadata) exactly like `chatbot.stream`.
    Also records latency and attempts to capture token usage from the final message metadata.
    """
    chatbot = get_compiled_chatbot(session_id)

    # We'll measure time from start to finish and log at the end.
    start_ts = time.time()
    final_response = None
    # yield messages from the chatbot.stream generator directly
    for message_chunk, metadata in chatbot.stream({"messages": messages}, config=config, stream_mode="messages"):
        # each yielded chunk we pass through
        yield message_chunk, metadata
        # attempt to capture final chunk (AIMessage) for token info later
        if hasattr(message_chunk, "content"):
            final_response = message_chunk

    end_ts = time.time()
    latency_ms = (end_ts - start_ts) * 1000.0

    # attempt to extract token usage from final_response metadata if any (best-effort)
    token_count = None
    try:
        # In some LLM wrappers, token usage may be in message_chunk.metadata or attributes
        if final_response is not None:
            md = getattr(final_response, "metadata", {}) or {}
            # common locations (best-effort)
            token_count = md.get("token_usage") or md.get("usage", {}).get("total_tokens") or md.get("token_count")
            if token_count is not None:
                token_count = int(token_count)
    except Exception:
        token_count = None

    _log_metrics(session_id, latency_ms, token_count)

def retrieve_all_threads_for_session(session_id: str) -> List[str]:
    """List all thread_ids saved for a given session (reads that session's DB)."""
    chatbot, checkpointer = _build_graph_for_session(session_id)
    threads = set()
    for checkpoint in checkpointer.list(None):
        cfg = checkpoint.config.get("configurable") or {}
        tid = cfg.get("thread_id") or checkpoint.config.get("configurable", {}).get("thread_id")
        if tid:
            threads.add(tid)
    return list(threads)

def get_metrics(session_id: str | None = None, days: int = 7) -> Dict[str, Any]:
    """
    Return aggregated metrics. If session_id is provided, filter to that session.
    Returns (summary dict): avg_latency_ms, total_calls, avg_tokens_per_call, tokens_missing_count
    """
    cur = _conn_metrics.cursor()
    now = int(time.time())
    cutoff = now - days * 24 * 3600
    if session_id:
        cur.execute(
            "SELECT latency_ms, token_count FROM metrics WHERE session_id = ? AND timestamp >= ?",
            (session_id, cutoff),
        )
    else:
        cur.execute("SELECT latency_ms, token_count FROM metrics WHERE timestamp >= ?", (cutoff,))
    rows = cur.fetchall()
    if not rows:
        return {"avg_latency_ms": None, "total_calls": 0, "avg_tokens_per_call": None, "missing_token_info": 0}
    latencies = [r[0] for r in rows if r[0] is not None]
    tokens = [r[1] for r in rows if r[1] is not None]
    missing = len(rows) - len(tokens)
    avg_latency = sum(latencies) / len(latencies) if latencies else None
    avg_tokens = sum(tokens) / len(tokens) if tokens else None
    return {
        "avg_latency_ms": avg_latency,
        "total_calls": len(rows),
        "avg_tokens_per_call": avg_tokens,
        "missing_token_info": missing,
    }
