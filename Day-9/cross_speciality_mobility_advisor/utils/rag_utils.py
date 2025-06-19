# utils/rag_utils.py
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_chroma import Chroma # Make sure this is updated!
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

load_dotenv()

# Global variables
CHROMA_DB_PATH = "./chroma_db"
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001") # Ensure model name is correct

# Ensure this is defined before usage
vectorstore = None

def setup_vectordb():
    global vectorstore # Declare that you intend to modify the global variable
    if vectorstore:
        print("Vectorstore already initialized in memory.")
        return vectorstore

    print("Attempting to set up ChromaDB...")
    
    # Check if the Chroma DB already exists
    if os.path.exists(CHROMA_DB_PATH) and os.listdir(CHROMA_DB_PATH):
        print("Loading existing ChromaDB...")
        try:
            vectorstore = Chroma(persist_directory=CHROMA_DB_PATH, embedding_function=embeddings)
            print("ChromaDB loaded successfully.")
        except Exception as e:
            print(f"Error loading existing ChromaDB: {e}")
            import traceback
            traceback.print_exc() # Print full traceback
            vectorstore = None # Reset if loading fails
            print("Attempting to re-create ChromaDB from scratch due to load error.")
            return create_vectordb_from_data() # Try to recreate
    else:
        print("ChromaDB not found or empty. Creating new ChromaDB...")
        vectorstore = create_vectordb_from_data()
    
    return vectorstore

def create_vectordb_from_data():
    documents = []
    data_path = "./data"
    print(f"Loading documents from {data_path}...")
    
    if not os.path.exists(data_path):
        print(f"Warning: Data directory '{data_path}' does not exist. No documents will be loaded.")
        return None # Or raise an error
        
    for filename in os.listdir(data_path):
        filepath = os.path.join(data_path, filename)
        if filename.endswith(".txt"):
            try:
                loader = TextLoader(filepath)
                documents.extend(loader.load())
                print(f"Loaded {filename} as Text.")
            except Exception as e:
                print(f"Error loading {filename} as Text: {e}")
                
        elif filename.endswith(".pdf"):
            try:
                loader = PyPDFLoader(filepath)
                documents.extend(loader.load())
                print(f"Loaded {filename} as PDF.")
            except Exception as e:
                print(f"Error loading {filename} as PDF: {e}")

    if not documents:
        print("No documents loaded. Cannot create ChromaDB.")
        return None

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(splits)} chunks.")

    print("Creating new ChromaDB from splits...")
    try:
        vectordb = Chroma.from_documents(
            documents=splits,
            embedding=embeddings,
            persist_directory=CHROMA_DB_PATH
        )
        vectordb.persist()
        print("ChromaDB created and persisted successfully.")
        return vectordb
    except Exception as e:
        print(f"Error creating ChromaDB from documents: {e}")
        import traceback
        traceback.print_exc() # Print full traceback
        return None

def get_retriever():
    global vectorstore # Make sure to access the global variable
    if vectorstore is None:
        print("Retriever requested but vectorstore is not initialized. Initializing now...")
        vectorstore = setup_vectordb()
        if vectorstore is None:
            print("Failed to initialize vectorstore. Cannot create retriever.")
            return None
    
    print("Creating retriever from vectorstore.")
    return vectorstore.as_retriever()

# Initial setup when the module is imported
# This ensures vectorstore is initialized when get_retriever() is first called
if __name__ == '__main__':
    print("Running rag_utils.py as main to create/update ChromaDB.")
    setup_vectordb()