# embedder.py

from langchain_openai import AzureOpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Define Azure OpenAI configuration variables
AZURE_OPENAI_EMBEDDER_API_KEY = os.getenv("AZURE_OPENAI_EMBEDDER_API_KEY")
AZURE_OPENAI_EMBEDDER_ENDPOINT = os.getenv("AZURE_OPENAI_EMBEDDER_ENDPOINT")
AZURE_OPENAI_EMBEDDER_DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_EMBEDDER_DEPLOYMENT_NAME")
AZURE_OPENAI_EMBEDDER_API_VERSION = os.getenv("AZURE_OPENAI_EMBEDDER_API_VERSION")

# 1. Initialize Azure OpenAI Embeddings
embedder = AzureOpenAIEmbeddings(
    azure_deployment = AZURE_OPENAI_EMBEDDER_DEPLOYMENT_NAME,
    azure_endpoint = AZURE_OPENAI_EMBEDDER_ENDPOINT,
    openai_api_key = AZURE_OPENAI_EMBEDDER_API_KEY,
    api_version = AZURE_OPENAI_EMBEDDER_API_VERSION
)

# 2. Custom Validation Function
class EmbeddingValidator:
    @staticmethod
    def validate(embeddings):
        """Validate embeddings structure and content."""
        if not embeddings:
            raise ValueError("No embeddings returned!")
        
        for i, emb in enumerate(embeddings):
            if not isinstance(emb, list):
                raise ValueError(f"Embedding {i} is not a list")
            if len(emb) == 0:
                raise ValueError(f"Embedding {i} is empty")
            if not all(isinstance(x, float) for x in emb):
                raise ValueError(f"Embedding {i} contains non-float values")
        
        print(f"✅ Validated {len(embeddings)} embeddings (dim={len(embeddings[0])})")



def create_embeddings(documents):
    """Create and validate embeddings using Azure OpenAI."""

    print(f"Input type: {type(documents[0])}")

    try:
        # Generate embeddings
        #embeddings = embedder.embed_documents(documents)
        
        # Validate
        #EmbeddingValidator.validate(embeddings)
        
        # Create FAISS vectorstore
        vectorstore = FAISS.from_documents(documents, embedder)

        # Get embeddings for the first 3 documents
        
        print(f"Embedding dimensions: {vectorstore.index.d}") 
        doc_embeddings = vectorstore.index.reconstruct_n(0, 3)  # Returns numpy array
        print(f"First document embedding (first 5 dims): {doc_embeddings[0][:5]}")

        print(f"Total documents in vectorstore: {vectorstore.index.ntotal}")
        return vectorstore
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        raise
