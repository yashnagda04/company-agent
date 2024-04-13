from langchain.agents import Tool
from langchain.vectorstores.pgvector import PGVector, DistanceStrategy
from langchain_openai import OpenAIEmbeddings
from company_agent.constant import EMBEDDINGS_MODEL, CONNECTION_STRING, COLLECTION_NAME
from .templates import (
    VECTOR_DB_TOOL_NAME,
    VECTOR_DB_TOOL_DESCRIPTION,
    VECTOR_DB_QUESTION_FIELD_DESCRIPTION,
)
from langchain.pydantic_v1 import BaseModel, Field
from .hyde import get_hyde_embedding

embeddings = OpenAIEmbeddings(model=EMBEDDINGS_MODEL)

vector_db = PGVector(
    collection_name=COLLECTION_NAME,
    connection_string=CONNECTION_STRING,
    embedding_function=OpenAIEmbeddings(model=EMBEDDINGS_MODEL),
    distance_strategy=DistanceStrategy.COSINE,
)


class VectorDBInput(BaseModel):
    query: str = Field(description=VECTOR_DB_QUESTION_FIELD_DESCRIPTION)


def get_retriever():
    retriever = vector_db.as_retriever(search_kwargs={"k": 4})
    return retriever


def format_docs(docs):
    res = f"\n{'-' * 100}\n".join(
        [
            f"Document {i+1}:\n\n" + d.metadata["source"] + "\n" + d.page_content
            for i, d in enumerate(docs)
        ]
    )
    return res


def get_vector_db_tool(job_id):
    def get_docs(query):
        docs = vector_db.similarity_search(query=query, k=5, filter={"job_id": job_id})
        return format_docs(docs=docs)

    vector_db_search_tool = Tool(
        name=VECTOR_DB_TOOL_NAME,
        func=get_docs,
        description=VECTOR_DB_TOOL_DESCRIPTION,
        args_schema=VectorDBInput,
    )
    return vector_db_search_tool
