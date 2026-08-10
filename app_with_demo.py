import streamlit as st
from main import run_pipeline
from core.rag_engine import ask_question

st.set_page_config(
    page_title="AI Video Assistant",
    page_icon="🎬",
    layout="wide"
)

st.title("🎬 AI Video & Audio Assistant")
st.caption("Summarize videos/audio and chat with your content using RAG")

# ---- SAFE DEMO MODE ----
# WHY: processing an arbitrary user-supplied YouTube URL in a public demo
# has real costs -- API usage (Mistral/Sarvam billing per call), compute
# (Whisper transcription is slow on CPU), and legal/ToS risk (downloading
# arbitrary YouTube content). A single hardcoded, pre-approved short sample
# video sidesteps all three: fixed known cost, fixed known runtime, and
# content you've verified you're allowed to use.
DEMO_SOURCE = "https://www.youtube.com/watch?v=REPLACE_WITH_YOUR_SHORT_SAMPLE_VIDEO"
DEMO_LANGUAGE = "english"

with st.sidebar:
    st.header("Configuration")

    mode = st.radio(
        "Mode:",
        options=["🎯 Try Demo (recommended)", "🔧 Bring Your Own Video"],
        index=0
    )

    if mode == "🎯 Try Demo (recommended)":
        st.info(
            "This runs the full pipeline on a pre-selected short sample video, "
            "so you can see real results without waiting for a full download "
            "and transcription of your own content."
        )
        source = DEMO_SOURCE
        language = DEMO_LANGUAGE
        process_btn = st.button("Run Demo", type="primary", use_container_width=True)
    else:
        st.warning(
            "⚠️ Processing your own video uses live API calls and may take "
            "several minutes depending on video length. For a quick look, "
            "try Demo Mode instead."
        )
        source = st.text_input(
            "Source (YouTube URL or local path):",
            placeholder="https://youtube.com/... or path/to/file.mp3"
        )
        language = st.selectbox("Language:", options=["english", "hinglish"], index=0)
        process_btn = st.button("Process Video", type="primary", use_container_width=True)

if "pipeline_result" not in st.session_state:
    st.session_state.pipeline_result = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if process_btn:
    if not source or not source.strip() or "REPLACE_WITH" in source:
        st.error(
            "Demo video not configured yet -- set DEMO_SOURCE in app.py to a "
            "real, short (2-3 min) sample video URL you have rights to use."
        )
    else:
        with st.spinner("Processing video, generating summary, and building RAG index..."):
            try:
                result = run_pipeline(source=source.strip(), language=language)
                st.session_state.pipeline_result = result
                st.session_state.chat_history = []
                st.success("Processing complete!")
            except Exception as e:
                st.error(
                    f"Something went wrong while processing this video. "
                    f"This can happen due to a temporary API issue -- try again "
                    f"in a moment. (Details: {e})"
                )

if st.session_state.pipeline_result:
    res = st.session_state.pipeline_result

    st.header(res.get("title", "Analysis Summary"))
    st.divider()

    tab1, tab2, tab3 = st.tabs(["📝 Summary & Insights", "💬 Chat with Video", "📋 Extracted Details"])

    with tab1:
        st.subheader("Summary")
        st.write(res.get("summary", "No summary available."))

    with tab2:
        st.subheader("Interactive Q&A")
        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        if user_query := st.chat_input("Ask a question about this video..."):
            with st.chat_message("user"):
                st.markdown(user_query)
            st.session_state.chat_history.append({"role": "user", "content": user_query})

            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    try:
                        answer = ask_question(res["rag_chain"], user_query)
                    except Exception as e:
                        answer = f"Sorry, I couldn't answer that right now (API error: {e}). Please try again."
                    st.markdown(answer)
            st.session_state.chat_history.append({"role": "assistant", "content": answer})

    with tab3:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.subheader("Action Items")
            st.write(res.get("action_item", "None"))
        with col2:
            st.subheader("Key Decisions")
            st.write(res.get("key_decision", "None"))
        with col3:
            st.subheader("Open Questions")
            st.write(res.get("open_question", "None"))
else:
    st.info("Click **Run Demo** in the sidebar to see the assistant in action, "
            "or switch to 'Bring Your Own Video' to process your own content.")