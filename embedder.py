# embedder.py

from langchain_openai import AzureOpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
import os
from sentence_transformers import SentenceTransformer

# Use the new langchain_huggingface package for local embeddings
try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    raise ImportError("Please install langchain-huggingface cehck env.yml")

# Load environment variables
load_dotenv()

# 1. Define Azure OpenAI configuration variables
AZURE_OPENAI_EMBEDDER_API_KEY = os.getenv("AZURE_OPENAI_EMBEDDER_API_KEY")
AZURE_OPENAI_EMBEDDER_ENDPOINT = os.getenv("AZURE_OPENAI_EMBEDDER_ENDPOINT")
AZURE_OPENAI_EMBEDDER_DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_EMBEDDER_DEPLOYMENT_NAME")
AZURE_OPENAI_EMBEDDER_API_VERSION = os.getenv("AZURE_OPENAI_EMBEDDER_API_VERSION")

# Read embedding mode from environment
EMBEDDING_MODE = os.getenv("EMBEDDING_MODE").lower()

# Local embedder setup
local_embedder = None
if EMBEDDING_MODE == "local":
    local_embedder = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

class LocalEmbedder:
    @staticmethod
    def embed_documents(texts):
        return local_embedder.encode(texts, show_progress_bar=True)
    def __call__(self, texts):
        return self.embed_documents(texts)

# 2. Initialize Azure OpenAI Embeddings
azure_embedder = AzureOpenAIEmbeddings(
    azure_deployment = AZURE_OPENAI_EMBEDDER_DEPLOYMENT_NAME,
    azure_endpoint = AZURE_OPENAI_EMBEDDER_ENDPOINT,
    openai_api_key = AZURE_OPENAI_EMBEDDER_API_KEY,
    api_version = AZURE_OPENAI_EMBEDDER_API_VERSION
)

# 3. Create embeddings and Inspect the vectorstore
def create_embeddings(documents, db_path="./vectorstore"):
    """Create and validate embeddings using Azure or local model based on EMBEDDING_MODE."""
    print(f"Input type: {type(documents[0])}")
    try:
        if EMBEDDING_MODE == "azure":
            # Generate embeddings
            #embeddings = azure_embedder.embed_documents(documents) # Happens automatically through FAISS
            vectorstore = FAISS.from_documents(documents, azure_embedder)
        else:
            vectorstore = FAISS.from_documents(documents, local_embedder)
        
        # Get embeddings for the first 3 documents
        print(f"Embedding dimensions: {vectorstore.index.d}") 
        doc_embeddings = vectorstore.index.reconstruct_n(0, 3)  # Returns numpy array
        print(f"First document embedding (first 5 dims): {doc_embeddings[0][:5]}")
        print(f"Total documents in vectorstore: {vectorstore.index.ntotal}")

        inspect_vectorstore(vectorstore)  # Inspect the vectorstore

        # Save the vectorstore embeddings to disk
        print("Saving vectorstore to disk...")
        vectorstore.save_local(db_path)

        return vectorstore
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        raise

def inspect_vectorstore(vectorstore):
    """Properly inspect FAISS vectorstore contents"""
    print(f"\n{'='*50}\nVectorstore Inspection\n{'='*50}")
    
    # 1. vectorstore property
    print(f"FAISS Index Type: {type(vectorstore.index).__name__}")
        # FAISS structure name based on data size
            # IndexFlatL2: Exact search (small datasets)
            # IndexIVFFlat: Approximate search (large datasets)
            # IndexIVFPQ: Compressed vectors (huge datasets)

    print(f"Vectors: {vectorstore.index.ntotal}")
    print(f"Dimensions: {vectorstore.index.d}")


    # 2. Get all document IDs (FAISS uses hashes, not sequential integers)
    doc_ids = list(vectorstore.docstore._dict.keys())
    print(f"📄 Total Documents: {len(doc_ids)}")
    
    if not doc_ids:
        print("No documents found!")
        return
    
    # 3. Get first document (using actual ID, not assumed index 0)
    first_id = doc_ids[0]
    first_doc = vectorstore.docstore._dict[first_id]
    print(f"\n📝 First Document:\nContent: {first_doc.page_content[30:60]}...")
    print(f"Metadata: {first_doc.metadata}")
    
    # 4. Check embedding dimensions
    try:
        # FAISS indexes may not support reconstruct() for all index types
        embedding_dim = vectorstore.index.d
        print(f"\n🧮 Embedding Dimensions: {embedding_dim} (per vector)") #1536-dimensional vector per document
        
        # For index types that support reconstruction:
        if hasattr(vectorstore.index, 'reconstruct'):
            sample_embedding = vectorstore.index.reconstruct(0)  # Now refers to vector position
            print(f"Sample Vector (first 5): {sample_embedding[:5]}...")
    except Exception as e:
        print(f"⚠️ Could not inspect embeddings: {str(e)}")
    
    # 5. Test search
    print("\n🔍 Testing search...")
    try:
        results = vectorstore.similarity_search("cricket", k=1)
        if results:
            print(f"Top Result: {results[0].page_content[:200]}...")
        else:
            print("No results found")
    except Exception as e:
        print(f"Search failed: {str(e)}")

def load_embeddings(db_path="./vectorstore"):
    """Load pre-built vectorstore."""
    if EMBEDDING_MODE == "azure":
        return FAISS.load_local(
            folder_path=db_path,
            embeddings=azure_embedder,
            allow_dangerous_deserialization=True # need for security reasons
        )
    else:
        return FAISS.load_local(
            folder_path=db_path,
            embeddings=local_embedder,
            allow_dangerous_deserialization=True
        )
