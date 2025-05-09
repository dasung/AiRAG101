# prompt_engine.py

from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# Create a Prompt Template
def create_prompt_template():
    """
    Function to create a prompt template for the RAG Chain.
    """
    prompt = """You are a cricket expert. You need to answer the question related to the law of cricket. 
    Given below is the context and question of the user.
    context = {context}
    question = {question}
    """
    return ChatPromptTemplate.from_template(prompt)

def initialize_LLM(vectorstore, prompt):
    """
    Function to initialize the LLM with the given vectorstore and prompt template.
    """
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0, api_key="<Enter your API Key>")
    retriever = vectorstore.as_retriever()

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return rag_chain

def invoke_rag_chain(rag_chain, question):
    """
    Function to invoke the RAG Chain with a given question.
    """
    return rag_chain.invoke(question)
