# pdf_loader.py

import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

documents = []

def load_pdfs(pdf_folder_path):
    """
    Function to load all PDFs from a given folder path.
    """
    for file_name in os.listdir(pdf_folder_path):
        if file_name.endswith('.pdf'):
            file_path = os.path.join(pdf_folder_path, file_name)
            loader = PyPDFLoader(file_path)
            docs = loader.load()
            documents.extend(docs)

    for doc in documents:
        print("***************************************************************")
        print(doc.page_content)

def chunk_pdf_data():
    """
    Function to chunk the documents using RecursiveCharacterTextSplitter.
    """
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(documents)
    return splits
