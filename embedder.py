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
    print("🧮 Creating Embeddings...")
    try:
        if EMBEDDING_MODE == "azure":
            # Generate embeddings
            #embeddings = azure_embedder.embed_documents(documents) # Happens automatically through FAISS
            vectorstore = FAISS.from_documents(documents, azure_embedder)
        else:
            vectorstore = FAISS.from_documents(documents, local_embedder)
        
        # debug only
        inspect_vectorstore(vectorstore)  # Inspect the vectorstore

        # Save the vectorstore embeddings to disk
        print("💾 Saving vectorstore to disk...")
        vectorstore.save_local(db_path)

        return vectorstore
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        raise

def inspect_vectorstore(vectorstore):
    """Properly inspect FAISS vectorstore contents"""
    print(f"\n{'='*50}\n🧪 Start New Vectorstore Inspection\n{'='*50}")
    
    try:
        # 1. vecttorstore properties
        print(f"FAISS Embedded Structure Type: {type(vectorstore.index).__name__}")
            # FAISS structure name based on data size
                # IndexFlatL2: Exact search (small datasets)
                # IndexIVFFlat: Approximate search (large datasets)
                # IndexIVFPQ: Compressed vectors (huge datasets)

        print(f"\n🔍 Total Vectors: {vectorstore.index.ntotal}")
        print(f"🧮 Embedding Dimensions: {vectorstore.index.d}")
        print(f"📄 Total documents in vectorstore: {vectorstore.index.ntotal}")

        doc_embeddings = vectorstore.index.reconstruct_n(0, 3)
        print(f"First document embedding (first 5 dims): {doc_embeddings[0][:5]}")

        # 2. document content
        doc_ids = list(vectorstore.docstore._dict.keys())
        if not doc_ids:
            print("⚠️ No documents found!")
            return

        first_id = doc_ids[0]
        first_doc = vectorstore.docstore._dict[first_id]
        print(f"\n📝 First Document Content: {first_doc.page_content[30:60]}...")
        print(f"📝 Metadata.source: {first_doc.metadata.get('source', 'No source metadata')}")

    except Exception as e:
        print(f"⚠️ Could not inspect embeddings: {str(e)}")

    print(f"\n{'='*50}\n✔️  Vectorstore Inspection Completed\n{'='*50}")


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
