import uuid
from company_agent.constant import EMBEDDINGS_MODEL, COLLECTION_NAME, CONNECTION_STRING
from langchain_community.vectorstores.pgvector import PGVector, DistanceStrategy
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders.pdf import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


class EmbeddingManager:
    def __init__(self):
        self.vector_store = PGVector(
            collection_name=COLLECTION_NAME,
            connection_string=CONNECTION_STRING,
            embedding_function=OpenAIEmbeddings(model=EMBEDDINGS_MODEL),
            distance_strategy=DistanceStrategy.COSINE,
        )

    def create_documents(self, file_path):
        pages = []
        loader = PyMuPDFLoader(file_path)
        pages.extend(loader.load_and_split())

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100,
            length_function=len,
        )
        documents = text_splitter.split_documents(pages)
        return documents

    def create_and_store_embedding(self, documents, job_id):
        texts = [doc.page_content for doc in documents]
        metadatas = [doc.metadata for doc in documents]
        # ids = [str(uuid.uuid1()) for _ in texts]
        for doc in documents:
            doc.metadata["job_id"] = job_id

        res = self.vector_store.add_texts(texts=texts, metadatas=metadatas)
        return res

    def process_all_documents(self, document_paths, job_id):
        for path in document_paths:
            documents = self.create_documents(path)
            self.create_and_store_embedding(documents, job_id)
            print(f"Processed {len(documents)} documents from {path}")
