import streamlit as st
import requests
import os

# Configuration
API_BASE_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

st.set_page_config(
    page_title="AI Document Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        color: #1E3A8A;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #4B5563;
        margin-bottom: 2rem;
    }
    .source-box {
        background-color: #F3F4F6;
        border-left: 4px solid #3B82F6;
        padding: 10px 15px;
        margin: 8px 0;
        border-radius: 4px;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-header">📚 AI Document Assistant (RAG)</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Upload documents, search semantically, and ask questions powered by PostgreSQL pgvector and Claude.</p>', unsafe_allow_html=True)

# Sidebar - Document Management & Upload
st.sidebar.title("📁 Document Library")

uploaded_file = st.sidebar.file_uploader("Upload Document", type=["pdf", "docx", "txt", "md"])
if uploaded_file is not None:
    if st.sidebar.button("Process & Ingest"):
        with st.spinner("Processing document, chunking, and embedding..."):
            try:
                files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                response = requests.post(f"{API_BASE_URL}/api/documents/upload", files=files)
                if response.status_code == 201:
                    data = response.json()
                    st.sidebar.success(f"Successfully uploaded '{data['filename']}' ({data['chunk_count']} chunks)!")
                    st.rerun()
                else:
                    err_detail = response.json().get("detail", "Unknown error")
                    st.sidebar.error(f"Upload failed: {err_detail}")
            except Exception as e:
                st.sidebar.error(f"Connection error: {str(e)}")

st.sidebar.markdown("---")
st.sidebar.subheader("Uploaded Documents")

try:
    docs_response = requests.get(f"{API_BASE_URL}/api/documents")
    if docs_response.status_code == 200:
        documents = docs_response.json()
        if not documents:
            st.sidebar.info("No documents uploaded yet.")
        else:
            for doc in documents:
                col1, col2 = st.sidebar.columns([3, 1])
                with col1:
                    st.text(f"📄 {doc['filename']} ({doc['chunk_count']} chunks)")
                with col2:
                    if st.button("🗑️", key=f"del_{doc['id']}", help="Delete document"):
                        del_res = requests.delete(f"{API_BASE_URL}/api/documents/{doc['id']}")
                        if del_res.status_code == 204:
                            st.success(f"Deleted {doc['filename']}")
                            st.rerun()
                        else:
                            st.error("Failed to delete")
    else:
        st.sidebar.error("Could not fetch document list.")
except Exception:
    st.sidebar.warning("Backend API is offline or unreachable.")

# Main Area Tabs
tab_chat, tab_search, tab_docs = st.tabs(["💬 Chat & Q&A", "🔍 Semantic Search", "📋 Document Details"])

# --- TAB 1: Chat & Q&A ---
with tab_chat:
    st.subheader("Chat with your Documents")

    # Initialize chat history in session state
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "sources" in message and message["sources"]:
                with st.expander("📚 View Sources & Citations"):
                    for src in message["sources"]:
                        st.markdown(f"""
                        <div class="source-box">
                            <b>Document ID:</b> {src['document_id']} | <b>Page:</b> {src['page_number']} | <b>Similarity:</b> {src.get('similarity', 'N/A')}<br>
                            {src['content']}
                        </div>
                        """, unsafe_allow_html=True)

    # Chat input
    user_question = st.chat_input("Ask a question about your documents...")
    if user_question:
        st.session_state.messages.append({"role": "user", "content": user_question})
        with st.chat_message("user"):
            st.markdown(user_question)

        with st.chat_message("assistant"):
            with st.spinner("Searching documents and generating answer with Claude..."):
                try:
                    payload = {"question": user_question, "top_k": 5}
                    res = requests.post(f"{API_BASE_URL}/api/rag/chat", json=payload)
                    if res.status_code == 200:
                        data = res.json()
                        answer = data["answer"]
                        sources = data["sources"]

                        st.markdown(answer)
                        if sources:
                            with st.expander("📚 View Sources & Citations"):
                                for src in sources:
                                    st.markdown(f"""
                                    <div class="source-box">
                                        <b>Document ID:</b> {src['document_id']} | <b>Page:</b> {src['page_number']} | <b>Similarity:</b> {src.get('similarity', 'N/A')}<br>
                                        {src['content']}
                                    </div>
                                    """, unsafe_allow_html=True)

                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": answer,
                            "sources": sources
                        })
                    else:
                        err = res.json().get("detail", "Error generating response")
                        st.error(f"Error: {err}")
                except Exception as e:
                    st.error(f"Failed to connect to backend: {str(e)}")

# --- TAB 2: Semantic Search ---
with tab_search:
    st.subheader("Semantic Vector Search")
    search_query = st.text_input("Enter search query:")
    top_k = st.slider("Top results", min_value=1, max_value=20, value=5)

    if st.button("Search", key="search_btn"):
        if search_query.strip():
            with st.spinner("Searching..."):
                try:
                    res = requests.post(f"{API_BASE_URL}/api/rag/search", json={"query": search_query, "top_k": top_k})
                    if res.status_code == 200:
                        results = res.json()
                        if not results:
                            st.info("No matching chunks found.")
                        else:
                            st.success(f"Found {len(results)} matching chunks:")
                            for i, chunk in enumerate(results, 1):
                                st.markdown(f"""
                                <div class="source-box">
                                    <b>Result {i}</b> | <b>Document ID:</b> {chunk['document_id']} | <b>Page:</b> {chunk['page_number']} | <b>Similarity Score:</b> {chunk.get('similarity', 'N/A')}
                                    <p style="margin-top: 5px;">{chunk['content']}</p>
                                </div>
                                """, unsafe_allow_html=True)
                    else:
                        st.error("Search request failed.")
                except Exception as e:
                    st.error(f"Connection error: {str(e)}")

# --- TAB 3: Document Details ---
with tab_docs:
    st.subheader("Document Content & Chunks Browser")
    try:
        docs_res = requests.get(f"{API_BASE_URL}/api/documents")
        if docs_res.status_code == 200:
            docs = docs_res.json()
            if docs:
                selected_doc_name = st.selectbox("Select Document", [d['filename'] for d in docs])
                selected_doc = next((d for d in docs if d['filename'] == selected_doc_name), None)

                if selected_doc:
                    detail_res = requests.get(f"{API_BASE_URL}/api/documents/{selected_doc['id']}")
                    if detail_res.status_code == 200:
                        detail = detail_res.json()
                        st.markdown(f"**Filename:** {detail['filename']}")
                        st.markdown(f"**File Size:** {detail.get('file_size', 0)} bytes")
                        st.markdown(f"**Upload Date:** {detail['upload_date']}")
                        st.markdown(f"**Total Chunks:** {len(detail['chunks'])}")

                        st.markdown("---")
                        st.markdown("### Extracted Chunks")
                        for chunk in detail['chunks']:
                            st.markdown(f"""
                            <div class="source-box">
                                <b>Chunk ID:</b> {chunk['id']} | <b>Page:</b> {chunk['page_number']} | <b>Index:</b> {chunk['chunk_index']}
                                <p style="margin-top: 5px;">{chunk['content']}</p>
                            </div>
                            """, unsafe_allow_html=True)
            else:
                st.info("No documents available.")
    except Exception:
        st.warning("Backend API is offline or unreachable.")
