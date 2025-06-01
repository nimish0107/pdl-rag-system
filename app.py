import streamlit as st
import asyncio
import uuid
import os
import shutil
from utils import DATA_DIR
from typing import AsyncGenerator
import pandas as pd
from datetime import datetime

# Import your RAG functions (assuming they're in the same directory or properly installed)
from services import add_document, query_chatbot, delete_doc_by_id, delete_all_docs
from TTS.tts_engine import generate_audio

# Configure page
st.set_page_config(
    page_title="Multilingual RAG Pipeline",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'document_history' not in st.session_state:
    st.session_state.document_history = []

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Sidebar navigation
st.sidebar.title("🔍 RAG Pipeline Navigation")
page = st.sidebar.selectbox(
    "Choose a page:",
    ["💬 Query Chatbot", "📄 Document Ingestion", "🗑️ Document Management"]
)

# Helper function to run async functions
def run_async(coro):
    """Helper to run async functions in Streamlit"""
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop.run_until_complete(coro)

# Page 1: Query Chatbot
if page == "💬 Query Chatbot":
    st.title("💬 Multilingual RAG Chatbot")
    st.markdown("Ask questions in Punjabi, Hindi, or English!")
    
    # Language selection
    col1, col2 = st.columns([1, 3])
    with col1:
        language = st.selectbox(
            "Select Language:",
            ["punjabi", "hindi", "english"],
            format_func=lambda x: {
                "punjabi": "ਪੰਜਾਬੀ (Punjabi)",
                "hindi": "हिंदी (Hindi)", 
                "english": "English"
            }[x]
        )
    
    with col2:
        k_value = st.slider("Number of relevant chunks to retrieve:", 1, 10, 6)
    
    # Chat interface
    st.markdown("### Chat Interface")
    
    # Display chat history
    chat_container = st.container()
    with chat_container:
        for i, (user_msg, bot_msg, lang) in enumerate(st.session_state.chat_history):
            with st.chat_message("user"):
                st.write(f"**[{lang.title()}]** {user_msg}")
            with st.chat_message("assistant"):
                st.write(bot_msg)

                col1, col2 = st.columns([1, 4])
                with col1:
                    if st.button(f"🔊 Play Audio", key=f"audio_btn_{i}"):
                        with st.spinner("Generating audio..."):
                            audio_data = generate_audio(bot_msg, lang)
                            if audio_data:
                                st.audio(audio_data, format='audio/mp3')
                            else:
                                st.error("Failed to generate audio")
    # Query input
    query = st.chat_input("Enter your question here...")
    
    if query:
        # Add user message to chat
        with st.chat_message("user"):
            st.write(f"**[{language.title()}]** {query}")
        
        # Generate response
        with st.chat_message("assistant"):
            response_placeholder = st.empty()
            # full_response = ""
            response_chunks = []
            
            try:
                with st.spinner("Generating response..."):
                    # Create async generator and collect response
                    async def stream_response():
                        async for chunk in query_chatbot(query, language, k_value):
                            response_chunks.append(chunk)
                            response_placeholder.markdown("".join(response_chunks))

                    run_async(stream_response())
                    response_placeholder.markdown("".join(response_chunks))
                    st.success("Response generated successfully!")
                col1, col2 = st.columns([1, 4])
                with col1:
                    if st.button("🔊 Generate Audio", key="new_audio_btn"):
                        with st.spinner("Generating audio..."):
                            audio_data = generate_audio("".join(response_chunks), language)
                            if audio_data:
                                st.audio(audio_data, format='audio/mp3')
                            else:
                                st.error("Failed to generate audio")
                # Add to chat history
                st.session_state.chat_history.append((query, "".join(response_chunks), language))

            except Exception as e:
                st.error(f"Error generating response: {str(e)}")
                response_placeholder.write("Sorry, I encountered an error while processing your question.")
    
    # Clear chat history button
    if st.button("🗑️ Clear Chat History"):
        st.session_state.chat_history = []
        st.rerun()

# Page 2: Document Ingestion
elif page == "📄 Document Ingestion":
    st.title("📄 Document Ingestion")
    st.markdown("Upload image documents to add them to the RAG system.")
    
    # File uploader
    uploaded_files = st.file_uploader(
        "Choose image files",
        type=['png', 'jpg', 'jpeg'],
        accept_multiple_files=True,
        help="Upload images containing text in Punjabi, Hindi, or English"
    )
    
    if uploaded_files:
        st.markdown(f"### Selected Files ({len(uploaded_files)})")
        
        # Display file details
        file_details = []
        for file in uploaded_files:
            file_details.append({
                "Filename": file.name,
                "Size": f"{file.size / 1024:.1f} KB",
                "Type": file.type
            })
        
        df = pd.DataFrame(file_details)
        st.dataframe(df, use_container_width=True)
        
        # Process files
        if st.button("🚀 Process Documents", type="primary"):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for i, uploaded_file in enumerate(uploaded_files):
                try:
                    # Update progress
                    progress = (i + 1) / len(uploaded_files)
                    progress_bar.progress(progress)
                    status_text.text(f"Processing {uploaded_file.name}...")
                    
                    # Generate unique document ID
                    doc_uuid = str(uuid.uuid4())
                    file_path = os.path.join(DATA_DIR, doc_uuid + os.path.splitext(uploaded_file.name)[-1])
                    if os.path.exists(file_path):
                        raise ValueError(f"File {uploaded_file.name} already exists in the system.")
                    with open(file_path, "wb") as buffer:
                        shutil.copyfileobj(uploaded_file.file, buffer)

                    try:
                        # Process the document
                        chunk_count = run_async(add_document(file_path, doc_uuid))

                        # Add to document history
                        st.session_state.document_history.append({
                            "doc_id": doc_uuid,
                            "filename": uploaded_file.name,
                            "upload_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "chunk_count": chunk_count,
                            "size_kb": f"{uploaded_file.size / 1024:.1f}"
                        })
                        
                        st.success(f"✅ Successfully processed {uploaded_file.name} ({chunk_count} chunks created)")
                        
                    finally:
                        # Clean up the uploaded file from memory
                        uploaded_file.file.close()
                        
                except Exception as e:
                    st.error(f"❌ Error processing {uploaded_file.name}: {str(e)}")
            
            progress_bar.progress(1.0)
            status_text.text("✅ All documents processed!")
            
    # Display document history
    if st.session_state.document_history:
        st.markdown("### Document History")
        history_df = pd.DataFrame(st.session_state.document_history)
        history_df = history_df.rename(columns={
            "doc_id": "Document ID",
            "filename": "Filename",
            "upload_time": "Upload Time",
            "chunk_count": "Chunks",
            "size_kb": "Size (KB)"
        })
        st.dataframe(history_df, use_container_width=True)

# Page 3: Document Management
elif page == "🗑️ Document Management":
    st.title("🗑️ Document Management")
    st.markdown("Manage and delete documents from the RAG system.")
    
    if not st.session_state.document_history:
        st.info("No documents found. Please add some documents first using the Document Ingestion page.")
    else:
        # Display current documents
        st.markdown("### Current Documents")
        history_df = pd.DataFrame(st.session_state.document_history)
        display_df = history_df.rename(columns={
            "doc_id": "Document ID",
            "filename": "Filename", 
            "upload_time": "Upload Time",
            "chunk_count": "Chunks",
            "size_kb": "Size (KB)"
        })
        st.dataframe(display_df, use_container_width=True)
        
        # Delete specific document
        st.markdown("### Delete Specific Document")
        col1, col2 = st.columns([3, 1])
        
        with col1:
            # Create a selectbox with filename (doc_id) format
            doc_options = {
                f"{doc['filename']} ({doc['doc_id'][:8]}...)": doc['doc_id'] 
                for doc in st.session_state.document_history
            }
            
            if doc_options:
                selected_doc = st.selectbox(
                    "Select document to delete:",
                    options=list(doc_options.keys()),
                    help="Select a document to delete from the system"
                )
        
        with col2:
            if st.button("🗑️ Delete Selected", type="secondary"):
                if selected_doc:
                    doc_id = doc_options[selected_doc]
                    try:
                        with st.spinner("Deleting document..."):
                            success = run_async(delete_doc_by_id(doc_id))
                            
                        if success:
                            # Remove from session state
                            st.session_state.document_history = [
                                doc for doc in st.session_state.document_history 
                                if doc['doc_id'] != doc_id
                            ]
                            st.success("✅ Document deleted successfully!")
                            st.rerun()
                        else:
                            st.error("❌ Failed to delete document.")
                            
                    except Exception as e:
                        st.error(f"❌ Error deleting document: {str(e)}")
        
        # Delete all documents
        st.markdown("### Delete All Documents")
        st.warning("⚠️ This action will delete ALL documents from the system and cannot be undone!")
        
        col1, col2 = st.columns([3, 1])
        with col1:
            confirm_delete_all = st.checkbox("I understand that this will delete all documents permanently")
        
        with col2:
            if st.button("🗑️ Delete All", type="secondary", disabled=not confirm_delete_all):
                try:
                    with st.spinner("Deleting all documents..."):
                        success = run_async(delete_all_docs())
                    
                    if success:
                        st.session_state.document_history = []
                        st.success("✅ All documents deleted successfully!")
                        st.rerun()
                    else:
                        st.error("❌ Failed to delete all documents.")
                        
                except Exception as e:
                    st.error(f"❌ Error deleting all documents: {str(e)}")

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 System Stats")
if st.session_state.document_history:
    total_docs = len(st.session_state.document_history)
    total_chunks = sum(doc['chunk_count'] for doc in st.session_state.document_history)
    st.sidebar.metric("Total Documents", total_docs)
    st.sidebar.metric("Total Chunks", total_chunks)
else:
    st.sidebar.info("No documents loaded")

st.sidebar.markdown("### 💡 Tips")
st.sidebar.markdown("""
- Upload clear, high-quality images for better OCR results
- Supported languages: Punjabi, Hindi, English
- Use specific questions for better search results
- Delete unused documents to keep the system clean
""")