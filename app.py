import streamlit as st
from main import run_pipeline
from core.rag_engine import ask_question

# Page Configuration
st.set_page_config(
    page_title="AI Video Assistant",
    page_icon="🎬",
    layout="wide"
)

st.title("🎬 AI Video & Audio Assistant")
st.caption("Summarize videos/audio and chat with your content using RAG")

# Sidebar - Settings & Inputs
with st.sidebar:
    st.header("Configuration")
    
    source = st.text_input(
        "Source (YouTube URL or local path):",
        placeholder="https://youtube.com/... or path/to/file.mp3"
    )
    
    language = st.selectbox(
        "Language:",
        options=["english", "hinglish"],
        index=0
    )
    
    process_btn = st.button("Process Video", type="primary", use_container_width=True)

# Session State Initialization
if "pipeline_result" not in st.session_state:
    st.session_state.pipeline_result = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Pipeline Processing Logic
if process_btn:
    if not source.strip():
        st.error("Please enter a valid YouTube URL or local file path.")
    else:
        with st.spinner("Processing video, generating summary, and building RAG index..."):
            try:
                # Run your main pipeline
                result = run_pipeline(source=source.strip(), language=language)
                st.session_state.pipeline_result = result
                st.session_state.chat_history = [] # Reset chat for new source
                st.success("Processing complete!")
            except Exception as e:
                st.error(f"An error occurred during processing: {e}")

# Main Layout
if st.session_state.pipeline_result:
    res = st.session_state.pipeline_result
    
    # Title Header
    st.header(res.get("title", "Analysis Summary"))
    st.divider()

    # Results Grid (Tabs)
    tab1, tab2, tab3 = st.tabs(["📝 Summary & Insights", "💬 Chat with Video", "📋 Extracted Details"])

    with tab1:
        st.subheader("Summary")
        st.write(res.get("summary", "No summary available."))

    with tab2:
        st.subheader("Interactive Q&A")
        
        # Display past chat messages
        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        # User Question Input
        if user_query := st.chat_input("Ask a question about this video..."):
            # Render user message
            with st.chat_message("user"):
                st.markdown(user_query)
            st.session_state.chat_history.append({"role": "user", "content": user_query})

            # Generate and render assistant response
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    answer = ask_question(res["rag_chain"], user_query)
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
    st.info(" Enter a URL or file path in the sidebar and click **Process Video** to get started.")