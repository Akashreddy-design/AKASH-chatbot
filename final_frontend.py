# final_frontend.py
import streamlit as st
from final_backend import stream_chat, retrieve_all_threads_for_session, get_metrics, get_compiled_chatbot
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
import uuid
from datetime import datetime

st.set_page_config(page_title="AkashBot", layout="wide")

# ---------- Utility ----------
def generate_thread_id():
    return str(uuid.uuid4())

# ---------- Session identity ----------
if "user_id" not in st.session_state:
    st.session_state["user_id"] = generate_thread_id()  # persistent unique user id per visitor

if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = generate_thread_id()

if "message_history" not in st.session_state:
    st.session_state["message_history"] = []

# expose admin flag (only you should toggle on your browser)
if "is_admin" not in st.session_state:
    st.session_state["is_admin"] = False

# ---------- Sidebar ----------
st.sidebar.title("🤖 AkashBot")
st.sidebar.markdown("Smart AI assistant built with LangGraph + Streamlit")

if st.sidebar.button("🆕 New Chat"):
    st.session_state["thread_id"] = generate_thread_id()
    st.session_state["message_history"] = []

st.sidebar.divider()
st.sidebar.subheader("🗂️ My Conversations")

# show only this user's threads (this makes threads per visitor isolated)
threads = retrieve_all_threads_for_session(st.session_state["user_id"])
for t in reversed(threads):
    if st.sidebar.button(f"💬 {t[:8]}..."):
        st.session_state["thread_id"] = t
        # load messages from backend
        state_messages = get_compiled_chatbot = None  # placeholder to avoid linter warning
        # fetch messages via backend get_state through compiled chatbot
        compiled = get_compiled_chatbot = get_compiled_chatbot = None
        # Instead of exposing compiled directly, reuse earlier approach:
        # We'll call chatbot.get_state via compiled object returned from backend
        try:
            compiled_bot = get_compiled_chatbot(st.session_state["user_id"])
            state = compiled_bot.get_state(config={"configurable": {"thread_id": st.session_state["thread_id"]}})
            messages = state.values.get("messages", [])
            hist = []
            for msg in messages:
                role = "user" if isinstance(msg, HumanMessage) else "assistant"
                hist.append({"role": role, "content": msg.content})
            st.session_state["message_history"] = hist
        except Exception:
            st.session_state["message_history"] = []

# admin toggle (for owner)
if st.sidebar.checkbox("Admin: Show metrics / enable admin", value=False):
    st.session_state["is_admin"] = True
else:
    st.session_state["is_admin"] = False

if st.session_state["is_admin"]:
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Admin controls**")
    if st.sidebar.button("Reload metrics"):
        pass

# ---------- Main UI ----------
st.markdown(
    """
    <h2 style='text-align: center;'>🤖 AkashBot</h2>
    <p style='text-align: center; color: gray;'>Powered by LangGraph + OpenAI + Streamlit</p>
    """,
    unsafe_allow_html=True
)

# Render existing messages
for message in st.session_state["message_history"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

CONFIG = {
    "configurable": {"thread_id": st.session_state["thread_id"]},
    "metadata": {
        "thread_id": st.session_state["thread_id"],
        "user_id": st.session_state["user_id"],
    },
    "run_name": f"chat_with_akash_{st.session_state['user_id'][:8]}",
}

# ---------- Chat input and streaming ----------
if user_input := st.chat_input("Type your message..."):
    # append user message locally
    st.session_state["message_history"].append({"role": "user", "content": user_input})
    # render user immediately
    with st.chat_message("user"):
        st.markdown(user_input)

    # Stream assistant response using backend stream_chat which logs metrics
    with st.chat_message("assistant"):
        status_holder = {"box": None}

        def generator():
            # stream_chat yields (message_chunk, metadata)
            for message_chunk, metadata in stream_chat(
                st.session_state["user_id"],
                [HumanMessage(content=user_input)],
                config=CONFIG,
            ):
                # If tool is invoked, metadata may indicate current node/tool.
                if isinstance(message_chunk, ToolMessage):
                    tool_name = getattr(message_chunk, "name", "tool")
                    if status_holder["box"] is None:
                        status_holder["box"] = st.status(f"🔧 Using `{tool_name}` …", expanded=True)
                    else:
                        status_holder["box"].update(label=f"🔧 Using `{tool_name}` …", state="running", expanded=True)

                if isinstance(message_chunk, AIMessage):
                    # yield assistant content chunks
                    yield message_chunk.content

            # finalize status
            if status_holder["box"] is not None:
                status_holder["box"].update(label="✅ Tool finished", state="complete", expanded=False)

        assistant_text = st.write_stream(generator())

    # save assistant message to local history
    st.session_state["message_history"].append({"role": "assistant", "content": assistant_text})

# ---------- Admin Metrics view ----------
if st.session_state["is_admin"]:
    st.markdown("---")
    st.markdown("## 📊 Observability & Metrics (last 7 days)")
    agg = get_metrics(None, days=7)  # global aggregated
    st.write("**Global summary (last 7 days)**")
    st.write(f"Total calls: {agg['total_calls']}")
    st.write(f"Avg latency (ms): {agg['avg_latency_ms']}")
    st.write(f"Avg tokens / call (if reported): {agg['avg_tokens_per_call']}")
    st.write(f"Calls with missing token info: {agg['missing_token_info']}")

    st.markdown("**Per-session view**")
    # show recent sessions (from metrics DB)
    import sqlite3, time
    conn = sqlite3.connect("metrics.db")
    cur = conn.cursor()
    cur.execute("SELECT session_id, count(*), avg(latency_ms) FROM metrics GROUP BY session_id ORDER BY max(timestamp) DESC LIMIT 25")
    rows = cur.fetchall()
    for row in rows:
        sid, cnt, avg_lat = row
        st.write(f"Session `{sid[:8]}` — calls: {cnt}, avg latency ms: {avg_lat:.1f}")

# ---------- Footer ----------
st.markdown(
    f"""
    <div style='text-align:center; color:gray; font-size:12px; margin-top:20px;'>
        Session: {st.session_state['user_id'][:8]} | Thread: {st.session_state['thread_id'][:8]} | {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
    </div>
    """,
    unsafe_allow_html=True,
)
