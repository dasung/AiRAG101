# main.py

from pdf_loader import load_pdfs, chunk_pdf_data
from embedder import create_embeddings, load_embeddings
from prompt_engine import create_prompt_template, initialize_LLM, invoke_rag_chain

import os


def main():

    if not os.path.exists("./vectorstore"):
        # Step 1: Load PDFs
        print("⚠️  Vector DB is not found!")
        pdf_folder_path = "./test/"  
        load_pdfs(pdf_folder_path)

        # Step 2: Chunk the loaded PDFs
        splits = chunk_pdf_data()

        # Step 3: Create embeddings for the chunks
        vectorstore = create_embeddings(splits)
    else:
        # Normal operation
        print("✅  Vector DB found, loading from disk...")
        vectorstore = load_embeddings()

    # Step 4: Initialize the LLM
    prompt = create_prompt_template()
    rag_chain = initialize_LLM(vectorstore, prompt)

    # Step 5: Invoke the RAG Chain
    question = "Tell me most 3 important components of a trading system"  # sample question
    print("\n❓  Query on LLM: ", question)
    result = invoke_rag_chain(rag_chain, question)

    # Step 6: Display the result
    print("\n📋  LLM Response:\n", result)

if __name__ == "__main__":
    main()
