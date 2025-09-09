import streamlit as st
import os
import tempfile
import shutil
from pathlib import Path
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA

# Load environment variables from .env file
load_dotenv()

# Configuration
DATA_PATH = 'data/'
DB_FAISS_PATH = 'vectorstores/db_faiss'
HUGGING_FACE_REPO_ID = "mistralai/Mistral-7B-Instruct-v0.3"

def get_hf_token():
    """Get HuggingFace token from environment or session state"""
    # First try environment variable
    env_token = os.getenv("HF_TOKEN")
    if env_token:
        return env_token

    # Then try session state
    if "hf_token" in st.session_state and st.session_state.hf_token:
        return st.session_state.hf_token

    return None

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

def load_llm():
    """Load the language model"""
    hf_token = get_hf_token()
    if not hf_token:
        st.error("Please provide your HuggingFace token")
        return None

    llm = HuggingFaceEndpoint(
        repo_id=HUGGING_FACE_REPO_ID,
        task="text-generation",
        model_kwargs={"temperature": 0.7, "max_new_tokens": 500},
        huggingfacehub_api_token=hf_token
    )
    return llm

def create_qa_chain():
    """Create the QA chain"""
    try:
        # Load embeddings and database
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        
        if not os.path.exists(DB_FAISS_PATH):
            st.error("No vector database found. Please upload and process PDF files first.")
            return None
            
        db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
        
        # Load LLM
        llm = load_llm()
        if not llm:
            return None
        
        # Create custom prompt
        custom_prompt_template = """
        Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.
        
        {context}
        
        Question: {question}
        Answer:
        """
        
        prompt = PromptTemplate(
            template=custom_prompt_template,
            input_variables=["context", "question"]
        )
        
        # Create QA chain
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=db.as_retriever(search_kwargs={"k": 3}),
            chain_type_kwargs={"prompt": prompt},
            return_source_documents=True
        )
        
        return qa_chain
    except Exception as e:
        st.error(f"Error creating QA chain: {str(e)}")
        return None

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
            db = create_embeddings(texts)
            st.success("Vector database created successfully!")
        
        return True
    except Exception as e:
        st.error(f"Error processing files: {str(e)}")
        return False

# Streamlit UI
def main():
    st.set_page_config(
        page_title="Medical AI RAG System",
        page_icon="🏥",
        layout="wide"
    )
    
    st.title("🏥 Medical AI RAG System")
    st.markdown("Upload medical PDFs and ask questions about medical information")
    
    # Sidebar for file management
    with st.sidebar:
        st.header("� Configuration")

        # HuggingFace Token Input
        hf_token_input = st.text_input(
            "HuggingFace Token",
            type="password",
            value=st.session_state.get("hf_token", ""),
            help="Enter your HuggingFace API token. Get one free at https://huggingface.co/settings/tokens"
        )

        if hf_token_input:
            st.session_state.hf_token = hf_token_input
            st.success("✅ Token saved!")
        elif not get_hf_token():
            st.warning("⚠️ HuggingFace token required for chat functionality")
            st.info("Get a free token at: https://huggingface.co/settings/tokens")

        st.divider()
        st.header("�📁 Document Management")
        
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
                try:
                    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
                    db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
                    docs = db.similarity_search(search_query, k=num_results)

                    if docs:
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
                except Exception as e:
                    st.error(f"Error searching documents: {str(e)}")

    with col2:
        st.header("💬 AI Chat")

        # Check for token
        hf_token = get_hf_token()
        if not hf_token:
            st.warning("⚠️ Please enter your HuggingFace token in the sidebar to enable AI chat.")
            st.info("For now, you can use the document search feature on the left.")
            return

        # Check if database exists
        if not os.path.exists(DB_FAISS_PATH):
            st.warning("⚠️ Please upload and process PDF documents first.")
            return

        # Initialize chat history
        if "messages" not in st.session_state:
            st.session_state.messages = []

        # Display chat messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                if message["role"] == "assistant" and "sources" in message:
                    with st.expander("📖 View Sources"):
                        for i, source in enumerate(message["sources"], 1):
                            st.write(f"**Source {i}:**")
                            st.write(source)

        # Chat input
        if prompt := st.chat_input("Ask a medical question..."):
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": prompt})

            # Display user message
            with st.chat_message("user"):
                st.markdown(prompt)

            # Generate response
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    try:
                        # Create QA chain
                        qa_chain = create_qa_chain()
                        if qa_chain:
                            # Get response
                            response = qa_chain.invoke({"query": prompt})
                            answer = response.get("result", "I couldn't generate an answer.")
                            sources = response.get("source_documents", [])

                            # Display answer
                            st.markdown(answer)

                            # Prepare sources for storage
                            source_texts = []
                            if sources:
                                for doc in sources:
                                    source_text = doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
                                    source_texts.append(source_text)

                            # Add assistant message to chat history
                            st.session_state.messages.append({
                                "role": "assistant",
                                "content": answer,
                                "sources": source_texts
                            })

                            # Show sources
                            if sources:
                                with st.expander("📖 View Sources"):
                                    for i, doc in enumerate(sources, 1):
                                        st.write(f"**Source {i}:**")
                                        st.write(doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content)
                        else:
                            st.error("Failed to create QA chain. Please check your configuration.")
                    except Exception as e:
                        st.error(f"Error generating response: {str(e)}")
    
    # Instructions
    with st.expander("ℹ️ How to use this system"):
        st.markdown("""
        ### Getting Started:
        1. **Upload PDFs**: Use the sidebar to upload medical PDF documents
        2. **Process Documents**: Click "Process Documents" to build the knowledge base
        3. **Ask Questions**: Use the chat interface to ask medical questions
        
        ### Features:
        - 📄 **Multi-PDF Upload**: Upload multiple medical documents at once
        - 🔍 **Intelligent Search**: AI-powered semantic search through your documents
        - 💬 **Chat Interface**: Natural conversation with source citations
        - 📖 **Source References**: See which documents were used to answer your questions
        - 🗑️ **Database Management**: Clear and rebuild your knowledge base as needed
        
        ### Tips:
        - Ask specific medical questions for best results
        - The system will cite sources from your uploaded documents
        - You can upload additional PDFs anytime to expand the knowledge base
        """)

if __name__ == "__main__":
    main()
