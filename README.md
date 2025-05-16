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

![design (2)](https://github.com/user-attachments/assets/fad023fe-dace-4f75-9f60-713585df8fba)



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


