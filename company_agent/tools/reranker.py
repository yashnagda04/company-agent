import cohere
import os

def rerank_documents(query, documents, top_n=4):
    co = cohere.Client(os.getenv('COHERE_API_KEY'))
    response = co.rerank(
        model='rerank-english-v2.0',
        query=query,
        documents=documents,
        top_n=top_n
    )
    reranked_documents = [documents[res.index] for res in response.results]
    return reranked_documents
