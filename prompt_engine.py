# prompt_engine.py
import os
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, AzureChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
import openai
from dotenv import load_dotenv
from memory import get_history

# Load environment variables
load_dotenv()

# Define Azure OpenAI configuration variables
class Config:
    AZURE_OPENAI_LLM_API_KEY = os.getenv("AZURE_OPENAI_LLM_API_KEY")
    AZURE_OPENAI_LLM_ENDPOINT = os.getenv("AZURE_OPENAI_LLM_ENDPOINT")
    AZURE_OPENAI_LLM_DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_LLM_DEPLOYMENT_NAME")
    AZURE_OPENAI_LLM_API_VERSION = os.getenv("AZURE_OPENAI_LLM_API_VERSION")
    LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", 0.4))

config = Config()

# Create a Prompt Template with chat history

def create_prompt_template(user_id=None):
    """
    Function to create a prompt template for the RAG Chain, injecting chat history if user_id is provided.
    """
    history = []
    if user_id is not None:
        history = get_history(user_id)
    # Format history as a string
    history_str = "\n".join(history) if history else "No recent conversation."
    
    prompt = f"""You are a training assistant to onboarding engineers at a trading system-based software company.
    You need to answer questions related to Stock Exchange and Related Software Components.
    **Recent Conversation History:**
    {{history}}
    **Current Context:**
    context = {{context}}
    **User's Current Qeary:**
    question = {{question}}"""

    # partial fill history variable but {context} and {question}) to be filled in later.
    return ChatPromptTemplate.from_template(prompt).partial(history=history_str)

def initialize_LLM(vectorstore, prompt_template):
    """
    Function to initialize the LLM with the given vectorstore and prompt template using Azure OpenAI.
    """
    # AzureChatOpenAI eeminates the need for a separate embedding model
    llm  = AzureChatOpenAI(
        deployment_name = config.AZURE_OPENAI_LLM_DEPLOYMENT_NAME,  # Must match Azure portal
        model_name = config.AZURE_OPENAI_LLM_DEPLOYMENT_NAME,  # Or your specific model
        openai_api_key = config.AZURE_OPENAI_LLM_API_KEY,
        openai_api_version = config.AZURE_OPENAI_LLM_API_VERSION,
        azure_endpoint = config.AZURE_OPENAI_LLM_ENDPOINT,
        temperature = config.LLM_TEMPERATURE
    )

    # The "magic" happens inside LangChain's retriever abstraction:
    # search_type="similarity", "mmr", "svm" -> Default (similarity) relevant documents from the vector store
    # search_kwargs={"k": 4} -> Number of docs to retrieve
    retriever = vectorstore.as_retriever()


    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    # RAG Integration
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt_template
        | llm
        | StrOutputParser()
    )
    return rag_chain

def invoke_rag_chain(rag_chain, question):
    """Invokes the RAG chain with error handling"""

    # Rag chain is a callable object, so we can directly call it with the question
    # Frist question text is automatically embedded using the same embedder
    #       This happens inside [retriever.get_relevant_documents(question)]
    # LLM: Handles text→text (generation only)
    try:
        return rag_chain.invoke(question)
    except Exception as e:
        print(f"Error invoking RAG chain: {e}")
        raise
