# main.py

from pdf_loader import load_pdfs, chunk_pdf_data
from embedder import create_embeddings
from prompt_engine import create_prompt_template, initialize_LLM, invoke_rag_chain

def main():
    # Step 1: Load PDFs
    pdf_folder_path = "./data/"  # Replace with the actual path
    load_pdfs(pdf_folder_path)

    # Step 2: Chunk the loaded PDFs
    splits = chunk_pdf_data()

    # Step 3: Create embeddings for the chunks
    vectorstore = create_embeddings(splits)

    # Step 4: Initialize the LLM
    prompt = create_prompt_template()
    rag_chain = initialize_LLM(vectorstore, prompt)

    # Step 5: Invoke the RAG Chain
    question = "Named me best 3 hotels to stay in dubai"  # Example question
    result = invoke_rag_chain(rag_chain, question)

    # Step 6: Display the result
    print("Result:", result)

if __name__ == "__main__":
    main()
