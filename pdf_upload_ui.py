import streamlit as st
import os
import shutil
from pathlib import Path
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# Load environment variables from .env file
load_dotenv()

# Configuration
DATA_PATH = 'data/'
DB_FAISS_PATH = 'vectorstores/db_faiss'

# Ensure directories exist
os.makedirs(DATA_PATH, exist_ok=True)
os.makedirs('vectorstores', exist_ok=True)

def load_documents(data_path):
    """Load PDF documents from directory"""
    loader = DirectoryLoader(data_path, glob='*.pdf', loader_cls=PyPDFLoader)
    documents = loader.load()
    return documents

def split_documents(documents):
    """Split documents into chunks"""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    texts = text_splitter.split_documents(documents)
    return texts

def create_embeddings(texts):
    """Create embeddings and save to FAISS database"""
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    db = FAISS.from_documents(texts, embeddings)
    db.save_local(DB_FAISS_PATH)
    return db

def process_uploaded_files(uploaded_files):
    """Process uploaded PDF files"""
    if not uploaded_files:
        return False
    
    # Save uploaded files to data directory
    saved_files = []
    for uploaded_file in uploaded_files:
        file_path = os.path.join(DATA_PATH, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        saved_files.append(uploaded_file.name)
    
    try:
        # Process documents
        with st.spinner("Loading documents..."):
            documents = load_documents(DATA_PATH)
            st.success(f"Loaded {len(documents)} documents")
        
        with st.spinner("Splitting documents into chunks..."):
            texts = split_documents(documents)
            st.success(f"Created {len(texts)} text chunks")
        
        with st.spinner("Creating embeddings and building vector database..."):
            create_embeddings(texts)
            st.success("Vector database created successfully!")
        
        return True
    except Exception as e:
        st.error(f"Error processing files: {str(e)}")
        return False

def search_documents(query, k=3):
    """Search documents using the vector database"""
    try:
        if not os.path.exists(DB_FAISS_PATH):
            return None, "No vector database found. Please upload and process PDF files first."
        
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
        
        # Perform similarity search
        docs = db.similarity_search(query, k=k)
        return docs, None
    except Exception as e:
        return None, f"Error searching documents: {str(e)}"

# Streamlit UI
def main():
    st.set_page_config(
        page_title="Medical PDF Upload & Search",
        page_icon="📄",
        layout="wide"
    )
    
    st.title("📄 Medical PDF Upload & Search System")
    st.markdown("Upload medical PDFs and search through them using AI-powered semantic search")
    
    # Sidebar for file management
    with st.sidebar:
        st.header("📁 Document Management")
        
        # File upload
        uploaded_files = st.file_uploader(
            "Upload PDF files",
            type=['pdf'],
            accept_multiple_files=True,
            help="Upload medical PDFs to add to the knowledge base"
        )
        
        if uploaded_files:
            st.write(f"Selected {len(uploaded_files)} file(s):")
            for file in uploaded_files:
                st.write(f"- {file.name}")
        
        # Process button
        if st.button("🔄 Process Documents", disabled=not uploaded_files):
            success = process_uploaded_files(uploaded_files)
            if success:
                st.rerun()
        
        st.divider()
        
        # Show existing files
        st.subheader("📚 Current Knowledge Base")
        if os.path.exists(DATA_PATH):
            pdf_files = [f for f in os.listdir(DATA_PATH) if f.endswith('.pdf')]
            if pdf_files:
                st.write(f"**{len(pdf_files)} PDF file(s) in database:**")
                for file in pdf_files:
                    st.write(f"- {file}")
            else:
                st.write("No PDF files in database")
        
        # Database status
        st.divider()
        st.subheader("🗄️ Database Status")
        if os.path.exists(DB_FAISS_PATH):
            st.success("✅ Vector database ready")
        else:
            st.warning("⚠️ No vector database found")
        
        # Clear database option
        if st.button("🗑️ Clear Database", help="Remove all processed documents"):
            if os.path.exists(DB_FAISS_PATH):
                shutil.rmtree(DB_FAISS_PATH)
            if os.path.exists(DATA_PATH):
                for file in os.listdir(DATA_PATH):
                    if file.endswith('.pdf'):
                        os.remove(os.path.join(DATA_PATH, file))
            st.success("Database cleared!")
            st.rerun()
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("🔍 Document Search")
        
        # Check if database exists
        if not os.path.exists(DB_FAISS_PATH):
            st.warning("⚠️ Please upload and process PDF documents first using the sidebar.")
            st.info("👈 Use the sidebar to upload medical PDFs and build the knowledge base.")
        else:
            # Search interface
            search_query = st.text_input(
                "Enter your search query:",
                placeholder="e.g., What are the symptoms of diabetes?"
            )
            
            num_results = st.slider("Number of results", 1, 10, 3)
            
            if st.button("🔍 Search") and search_query:
                docs, error = search_documents(search_query, k=num_results)
                
                if error:
                    st.error(error)
                elif docs:
                    st.success(f"Found {len(docs)} relevant documents:")
                    
                    for i, doc in enumerate(docs, 1):
                        with st.expander(f"📖 Result {i}"):
                            st.write("**Content:**")
                            st.write(doc.page_content)
                            if hasattr(doc, 'metadata') and doc.metadata:
                                st.write("**Source:**")
                                st.json(doc.metadata)
                else:
                    st.info("No relevant documents found.")
    
    with col2:
        st.header("💬 AI Chat (Optional)")
        
        # HuggingFace Token Input
        hf_token_input = st.text_input(
            "HuggingFace Token (for AI chat)",
            type="password",
            value=st.session_state.get("hf_token", ""),
            help="Enter your HuggingFace API token for AI-powered chat. Get one free at https://huggingface.co/settings/tokens"
        )
        
        if hf_token_input:
            st.session_state.hf_token = hf_token_input
            st.success("✅ Token saved! You can now use AI chat.")
            
            # Simple chat interface
            if os.path.exists(DB_FAISS_PATH):
                chat_query = st.text_input("Ask a medical question:", key="chat_input")
                if st.button("💬 Ask AI") and chat_query:
                    st.info("AI chat functionality requires additional setup. For now, use the search feature on the left.")
            else:
                st.warning("Upload and process documents first to enable chat.")
        else:
            st.info("Enter your HuggingFace token above to enable AI chat functionality.")
            st.markdown("**For now, you can use the document search feature on the left.**")
    
    # Instructions
    with st.expander("ℹ️ How to use this system"):
        st.markdown("""
        ### Getting Started:
        1. **Upload PDFs**: Use the sidebar to upload medical PDF documents
        2. **Process Documents**: Click "Process Documents" to build the knowledge base
        3. **Search Documents**: Use the search feature to find relevant information
        4. **Optional AI Chat**: Enter HuggingFace token for AI-powered conversations
        
        ### Features:
        - 📄 **Multi-PDF Upload**: Upload multiple medical documents at once
        - 🔍 **Semantic Search**: AI-powered search through your documents
        - 📖 **Source References**: See which documents contain relevant information
        - 🗑️ **Database Management**: Clear and rebuild your knowledge base as needed
        - 💬 **AI Chat**: Optional AI-powered question answering (requires HuggingFace token)
        
        ### Tips:
        - Use specific medical terms for better search results
        - The system will show you the most relevant document sections
        - You can upload additional PDFs anytime to expand the knowledge base
        - Search works without any API tokens - only chat requires HuggingFace token
        """)

if __name__ == "__main__":
    main()
