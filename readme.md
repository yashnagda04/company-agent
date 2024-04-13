# Companny Agent
This agent is built on langchain platform. It can answer any question from company policy documents.
Here are some main feature of the agent.

## Key Features 
1. Built on langchain framework.
2. Can work with most of the open and closed source LLM with minimal changes.
3. RAG - Rag built on top of postgres and pgvector.
4. **Hyde** - Hypothetical document embedding is used to increase the peformance of RAG. It work by generating a hypothetical response to the query and then uses that to do the cosine similarity search instead of orginal query. It works because hypothetical answer is way closer to the real answer in the embedding space.
5. **Re-Ranking** - Re-ranking is one more method used to improve the performance of the RAG answers. In re-ranking a transformer model at Inference time and two document are passed through it to get a similarity score. It is very effective but takes more time.
6. **RAG + Re-Ranking** - In this intially we do a cosine a similarity search, so rag act as a intial filter and then we pass the retrieved document to re-ranker and get the most relevant documents.


## How to Run
1. setup virtual enviorement using conda or venv.
2. Install required libraries from **requirements.txt**
3. create .env file in the company_agent folder and add below keys.
    ```sh
    OPENAI_API_KEY=<your-open-ai-key>
    CONNECTION_STRING=<connection-string-of-postgres-db>
    COHERE_API_KEY=<your-cohere-api-key>
    ```
4. Edit these 2 variables in the test.py file.

    a. **questions** - enter your questions here.

    b. **document_paths** = path of your pdf document (company policy). You can also pass **multiple documents** in this.

5. The program will chunk and embed your pdf and store it in the postgres db. It also generate **unique job id** and store it in the metadata of documents, so when you run a new job it will not refer the old documents and only search in the current documents.
6. At the end it will create a **json file in the output folder** with the timestamp.

