# AiRAG101

### **Setup Env**

Use the following commands to create env:

    $ conda env create -f env.yml
    $ conda init


### **Usual System Startup**
    $ conda activate rag101
    $ uvicorn main:app --reload
    $ conda env update -f env.yml (if env.yml is updated)

*Note-* If you change EMBEDDING_MODE configuration, make sure to delete vectorstore created locally.

### **Current Architecture**

| Component          | Hosting        | Model Used               | Purpose                     |
|--------------------|---------------|--------------------------|-----------------------------|
| `embedder.py`      | Azure OpenAI  | `text-embedding-ada-002` | Generates document embeddings |
| `prompt_engine.py` | Azure OpenAI  | `gpt-4.1` (or similar)   | Handles chat/completions     |


### **Design**

![design](https://github.com/user-attachments/assets/1bebfef0-6e78-4dde-bafd-6e4f257d5b20)


### **Project Structure**

```
AiRAG101/
├── data/
│   └── *.pdf store
├── .env            -> to specify your local configurations
|
├── vectorstore/ 
│   ├── index.fasis -> optimized vector search structure by FASIS
│   └── index.pkl   -> document store - original text chunks + metadata
├── src/
│   ├── embedder.py
│   ├── prompt_engine.py
│   └── main.py
|
├── docs/
│   └── design.astah 
|
└── README.md
```


### **Code Review**

1. [env.yml] file to install following in our pythonbackend project.
    - We will be using LangChain framework
    - python=3.9
    - PyPDFLoader  for pdf processing
    - FAISS

2. [pdf_loader.py]. -> load_pdfs()
    - this contains logic for loading pdf function.
    - use for loop too load reach an every downloaded pdf to be loaded
    - keeps documents[] globally

```
    def load_pdfs(pdf_floder_path):
    for doc in docs:
        print("***************************************************************")
        print(doc.page_content)
```

3. [pdf_loader.py] to chunking the document read by LangChain.
    - use documents[]
    - use RecursiveCharacterTextSplitter method of LangChain to do this
```
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
```
    - returns splits to callee

4. [embedder.py] 
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

6. [prompt_engine.py] ->  initialize_LLM()

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

8. [main.py]
    - this loads pdf and chunk it using [pdf_loader.py]
    - splits passed to [embedder.py]
    - vector store passed to initialize_LLM()
    - finally calls invoke_rag_chain() and display the LLM result. 
