# pdf_loader.py

import os
from langchain.document_loaders import PyPDFLoader

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
