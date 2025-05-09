# AiRAG101

1. create [env.yml] file to install following in our pythonbackend project.
    - We will be using LangChain framework
    - python=3.9
    - PyPDFLoader  for pdf processing
    - FAISS

2. create [pdf_loader.py]. -> load_pdfs()
    - this contains logic for loading pdf function.
    - use for loop too load reach an every downloaded pdf to be loaded
    - keeps documents[] globally

```
    def load_pdfs(pdf_floder_path):
    for doc in docs:
        print("***************************************************************")
        print(doc.page_content)
```

3. [pdf_loader.py] add a new function chuck_pdf_data() to chunking the document read by LangChain.
    - use documents[]
    - use RecursiveCharacterTextSplitter method of LangChain to do this
```
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
```
    - returns splits to callee

4. [embedder.py] this new file 
    - function create_embeddings() to find embeddings of chunks. use OpenAIEmbeddings()
    - use FAISS as the Vector Store to save our embeddings.

```
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
vectorstore = FAISS.from_documents(documents=splits, embedding=OpenAIEmbeddings(api_key="<Enter Your Open AI Key>"))
```

5. [prompt_engine.py] this new file to creation of RAG Chain

- we need to create our Prompt Template.

```
from langchain.prompts import ChatPromptTemplate
prompt = """You are a cricket expert. You need to answer the question related to the law of cricket. 
Given below is the context and question of the user.
context = {context}
question = {question}
"""
prompt = ChatPromptTemplate.from_template(prompt)
```

6. Add new function to [prompt_engine.py] ->  initialize_LLM()

```
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

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
```

7. use seperate function to [prompt_engine.py] ->  invoke_rag_chain()

```
rag_chain.invoke("Tell me more about law number 30")
```