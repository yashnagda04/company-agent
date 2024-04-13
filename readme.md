# Company Agent
This agent is built on LangChain. It can answer any question from company policy documents.
Here are some main features of the agent.

## Key Features 
1. Built on the LangChain framework.
2. Can work with most open and closed source LLMs with minimal changes.
3. RAG - Rag built on top of PostgreSQL and pgvector.
4. **Hyde** - Hypothetical document embedding is used to increase the performance of RAG. It works by generating a hypothetical response to the query and then using that to perform cosine similarity searches instead of using the original query. It is effective because the hypothetical answer is closer to the real answer in the embedding space.
5. **Re-Ranking** - Re-ranking is another method used to improve the performance of RAG answers. In re-ranking, a cross-encoder model is used at inference time, and two documents are passed through it simultaneously to obtain a similarity score. It is very effective but takes more time for large volumes of documents.
6. **RAG + Re-Ranking** - Initially, we perform a cosine similarity search, and RAG acts as an initial filter. Then, we pass the retrieved document to the re-ranker to get the most relevant documents.

## How to Run
1. Set up a virtual environment using Conda or venv.
2. Install required libraries from **requirements.txt**.
3. Create a .env file in the company_agent folder and add the following keys:
    ```sh
    OPENAI_API_KEY=<your-openai-key>
    CONNECTION_STRING=<connection-string-of-postgres-db>
    COHERE_API_KEY=<your-cohere-api-key>
    ```
4. Edit these two variables in the test.py file: 

   a. **questions** - enter your questions here.

   b. **document_paths** - path of your PDF document (company policy). You can also pass **multiple documents** in this.

6. The program will chunk and embed your PDF and store it in the PostgreSQL database. It also generates a **unique job ID** and stores it in the metadata of documents, so when you run a new job, it will not refer to the old documents and will only search the current documents.
7. At the end, it will create a **JSON file in the output folder** with the timestamp.
