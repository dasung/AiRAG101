# embedder.py

from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

def create_embeddings(splits, api_key):
    """
    Function to create embeddings for the given chunks and store them in a FAISS vector store.
    """
    embeddings = OpenAIEmbeddings(api_key=api_key)
    vectorstore = FAISS.from_documents(documents=splits, embedding=embeddings)
    return vectorstore
