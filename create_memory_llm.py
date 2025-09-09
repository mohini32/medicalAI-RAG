from langchain_community.document_loaders import PyPDFLoader,DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# step 1: load data

data_path = 'data/'
def load_data(data):
    loader=DirectoryLoader(data_path, glob='*.pdf',
                            loader_cls=PyPDFLoader)
    documents = loader.load()
    return documents

# step 2 :split data

def split_data(documents):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    texts = text_splitter.split_documents(documents)
    return texts
    
# step 3: create embedding and store in faiss db
def get_embedding(texts):
    DB_FAISS_PATH = 'vectorstores/db_faiss'
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    db = FAISS.from_documents(texts, embeddings)
    db.save_local(DB_FAISS_PATH)
    return db

# Main execution
if __name__ == "__main__":
    print("Starting Medical AI RAG pipeline...")

    # Check if data directory has PDF files
    import os
    if not os.path.exists(data_path):
        print(f"Error: {data_path} directory not found!")
        exit(1)

    pdf_files = [f for f in os.listdir(data_path) if f.endswith('.pdf')]
    if not pdf_files:
        print(f"No PDF files found in {data_path} directory.")
        print("Please add some PDF files to the data/ directory and run again.")
        exit(1)

    print(f"Found {len(pdf_files)} PDF files: {pdf_files}")

    # Step 1: Load documents
    print("Step 1: Loading documents...")
    documents = load_data(data_path)
    print(f"Loaded {len(documents)} documents")

    # Step 2: Split documents
    print("Step 2: Splitting documents...")
    texts = split_data(documents)
    print(f"Created {len(texts)} text chunks")

    # Step 3: Create embeddings and save to FAISS
    print("Step 3: Creating embeddings and saving to FAISS database...")
    db = get_embedding(texts)
    print("FAISS database created and saved successfully!")
    print("Vector database saved to: vectorstores/db_faiss")
