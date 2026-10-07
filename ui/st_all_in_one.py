import os
import sys
import time
import uuid

import streamlit as st
import logfire

# --- Make the repo root importable so `import app...` works ---
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# --- Copy Streamlit secrets into environment variables BEFORE importing app.config ---
try:
    for _k, _v in st.secrets.items():
        if isinstance(_v, (str, int, float, bool)):
            os.environ.setdefault(str(_k), str(_v))
except Exception:
    pass

# --- Logfire (optional: only sends data if a token exists) ---
try:
    logfire.configure(token=os.getenv("LOGFIRE_TOKEN"), send_to_logfire="if-token-present")
    LOGFIRE_STATUS = "Connected & Tracing" if os.getenv("LOGFIRE_TOKEN") else "Standby (No Token)"
except Exception:
    LOGFIRE_STATUS = "Standby (No Token)"

ENABLE_GUARDRAILS = os.getenv("ENABLE_GUARDRAILS", "true").lower() != "false"


@st.cache_resource(show_spinner="Loading RAG pipeline (first run takes a minute)...")
def load_backend():
    """Load the LangGraph agent (and optionally NeMo Guardrails) once per server process."""
    from app.agents.graph import rag_agent
    if ENABLE_GUARDRAILS:
        from app.guardrails import initialize_rails, guard
        initialize_rails()
    else:
        def guard(_message):
            return False, None
    return rag_agent, guard


def run_query(q: str, thread_id: str) -> dict:
    """Same logic as POST /query in app/main.py, but called directly."""
    rag_agent, guard = load_backend()

    rail_fired, rail_response = guard(q)
    if rail_fired:
        return {
            "question": q,
            "answer": rail_response,
            "thought_process": ["Intent: Guardrails Fired", "Retrieval: Skipped"],
            "status": "Blocked by guardrails.",
            "sources": [],
        }

    initial_state = {
        "messages": [{"role": "user", "content": q}],
        "current_query": q,
        "documents": [],
        "plan": ["Start"],
        "status": "Initializing Graph...",
    }
    config = {"configurable": {"thread_id": thread_id}}
    out = rag_agent.invoke(initial_state, config=config)
    return {
        "question": q,
        "answer": out.get("final_answer"),
        "thought_process": out.get("plan"),
        "status": out.get("status"),
        "sources": out.get("documents", []),
    }


# --- PAGE CONFIG ---
st.set_page_config(
    page_title="Enterprise Agentic RAG",
    page_icon="🤖",
    layout="wide",
)

# --- AVATARS ---
AI_AVATAR = "🤖"
USER_AVATAR = "👤"

# --- SESSION MANAGEMENT ---
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
    logfire.info(f"✨ New User Session Created: {st.session_state.session_id}")

if "messages" not in st.session_state:
    st.session_state.messages = []

# --- SIDEBAR ---
with st.sidebar:
    st.title("🧠 Agent OS")
    st.markdown("---")


    st.markdown("---")
    st.success(f"Logfire: {LOGFIRE_STATUS}")
    st.info(f"Memory ID: {st.session_state.session_id[:8]}")
    
    if st.button("🗑️ Clear History & Memory", width="stretch", type="primary"):
        logfire.warning(f"🗑️ Memory Wipe Triggered for session: {st.session_state.session_id}")
        st.session_state.messages = []
        st.session_state.session_id = str(uuid.uuid4())
        st.rerun()

# --- MAIN CHAT ---
st.title("🤖 Enterprise Agentic Assistant")

# Display history
for message in st.session_state.messages:
    avatar = AI_AVATAR if message["role"] == "assistant" else USER_AVATAR
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])

# Chat Input
if prompt := st.chat_input("Ask about your documentation..."):
    # START TRACE: User Interaction
    with logfire.span("💬 User Chat Interaction", user_query=prompt, session_id=st.session_state.session_id):
        
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user", avatar=USER_AVATAR):
            st.markdown(prompt)

        # Assistant Response
        with st.chat_message("assistant", avatar=AI_AVATAR):
            data = {}
            with st.status("🔍 Agent is thinking...", expanded=True) as status:
                try:
                    with logfire.span("🧠 Running RAG pipeline"):
                        data = run_query(prompt, st.session_state.session_id)

                    steps = data.get("thought_process", [])
                    for step in steps:
                        st.markdown(f"⚙️ {step}", unsafe_allow_html=False)

                    status.update(label="✅ Answer Synthesized", state="complete", expanded=False)

                except Exception as e:
                    logfire.error(f"❌ UI-Backend Connection Failed: {e}")
                    status.update(label="❌ Pipeline Failed", state="error")
                    st.error(f"Pipeline error: {e}")
                    st.stop()

            # Answer streaming — outside status so it's always visible
            answer_placeholder = st.empty()
            full_answer = data.get("answer", "No response.")

            curr_text = ""
            for char in full_answer:
                curr_text += char
                answer_placeholder.markdown(curr_text + "▌")
                time.sleep(0.005)
            answer_placeholder.markdown(full_answer)

            # Sources — outside status so they're visible after it collapses
            sources = data.get("sources", [])
            if sources:
                with st.expander(f"📄 Retrieved Context ({len(sources)} chunks)"):
                    for i, source in enumerate(sources):
                        st.caption(f"Chunk {i + 1}")
                        st.info(source)
            else:
                st.caption("ℹ️ No context retrieved — conversational response.")

            st.session_state.messages.append({"role": "assistant", "content": full_answer})
            logfire.info("✅ Chat cycle completed successfully.")