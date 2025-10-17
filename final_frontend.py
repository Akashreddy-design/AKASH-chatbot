import streamlit as st
from final_backend import chatbot, retrieve_all_threads
from langchain_core.messages import HumanMessage, AIMessage,ToolMessage
import uuid
from datetime import datetime

# ---------- Utility ----------
def generate_thread_id():
    return str(uuid.uuid4())

def reset_chat():
    new_thread_id = generate_thread_id()
    st.session_state["thread_id"] = new_thread_id
    add_chat_thread(new_thread_id)
    st.session_state["message_history"] = []

def add_chat_thread(thread_id):
    if thread_id not in st.session_state["threads_chat"]:
        st.session_state["threads_chat"].append(thread_id)

def get_messages(thread_id):
    state = chatbot.get_state(config={"configurable": {"thread_id": thread_id}})
    return state.values.get("messages", [])

# ---------- Session Management ----------
if "message_history" not in st.session_state:
    st.session_state["message_history"] = []

if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = generate_thread_id()

if "threads_chat" not in st.session_state:
    st.session_state["threads_chat"] = retrieve_all_threads()

add_chat_thread(st.session_state["thread_id"])

# ---------- Sidebar ----------
st.sidebar.title("🤖 Akash Chatbot")
st.sidebar.markdown("**Smart AI assistant built with LangGraph + Streamlit**")

if st.sidebar.button("🆕 New Chat"):
    reset_chat()

st.sidebar.divider()
st.sidebar.subheader("🗂️ Chat History")

for thread_id in reversed(st.session_state["threads_chat"]):
    if st.sidebar.button(f"💬 {thread_id[:8]}..."):
        st.session_state["thread_id"] = thread_id
        messages = get_messages(thread_id)
        history = []
        for msg in messages:
            role = "user" if isinstance(msg, HumanMessage) else "assistant"
            history.append({"role": role, "content": msg.content})
        st.session_state["message_history"] = history

# ---------- Main Chat UI ----------
st.markdown(
    """
    <h2 style='text-align: center;'>🤖 Akash Chatbot</h2>
    <p style='text-align: center; color: gray;'>Powered by LangGraph, OpenAI, and Streamlit</p>
    """,
    unsafe_allow_html=True
)

# Display messages
for message in st.session_state["message_history"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

CONFIG = {
    "configurable": {"thread_id": st.session_state["thread_id"]},
    "metadata": {'thread_id': st.session_state["thread_id"]},
    "run_name": "chat_with_akash",
}

# ---------- Chat Input ----------
if user_input := st.chat_input("Type your message..."):
    st.session_state["message_history"].append({"role": "user", "content": user_input})

    with st.chat_message("user"):
        st.markdown(user_input)

    # with st.chat_message("assistant"):
    #     def ai_only_stream():
    #         for message_chunk, metadata in chatbot.stream(
    #             {"messages": [HumanMessage(content=user_input)]},
    #             config=CONFIG,
    #             stream_mode="messages",
    #         ):
    #             if isinstance(message_chunk, AIMessage):
    #                 text = message_chunk.content.replace("\\(", "").replace("\\)", "")
    #                 yield text

    #     response_text = st.write_stream(ai_only_stream())
    #     st.session_state["message_history"].append(
    #         {"role": "assistant", "content": response_text}
    #     )
    with st.chat_message("assistant"):
        # Use a mutable holder so the generator can set/modify it
        status_holder = {"box": None}

        def ai_only_stream():
            for message_chunk, metadata in chatbot.stream(
                {"messages": [HumanMessage(content=user_input)]},
                config=CONFIG,
                stream_mode="messages",
            ):
                # Lazily create & update the SAME status container when any tool runs
                if isinstance(message_chunk, ToolMessage):
                    tool_name = getattr(message_chunk, "name", "tool")
                    if status_holder["box"] is None:
                        status_holder["box"] = st.status(
                            f"🔧 Using `{tool_name}` …", expanded=True
                        )
                    else:
                        status_holder["box"].update(
                            label=f"🔧 Using `{tool_name}` …",
                            state="running",
                            expanded=True,
                        )

                # Stream ONLY assistant tokens
                if isinstance(message_chunk, AIMessage):
                    yield message_chunk.content

        ai_message = st.write_stream(ai_only_stream())

        # Finalize only if a tool was actually used
        if status_holder["box"] is not None:
            status_holder["box"].update(
                label="✅ Tool finished", state="complete", expanded=False
            )

    # Save assistant message
    st.session_state["message_history"].append(
        {"role": "assistant", "content": ai_message}
    )


# ---------- Footer ----------
st.markdown(
    f"""
    <div style='text-align:center; color:gray; font-size:12px; margin-top:20px;'>
        Session ID: {st.session_state["thread_id"][:8]} | {datetime.now().strftime("%H:%M:%S")}
    </div>
    """,
    unsafe_allow_html=True
)

with open("custom.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
