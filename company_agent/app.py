from dotenv import load_dotenv

load_dotenv()
from company_agent.agent.agent import Agent
from company_agent.rag.create_embedding import EmbeddingManager
import uuid


class CompanyAgent:
    def __init__(self):
        pass

    def process(self, questions, pdf_paths):
        job_id = str(uuid.uuid1())

        ## Step 1 - Embed and store and document in vector db
        embedding_manager = EmbeddingManager()
        embedding_manager.process_all_documents(
            document_paths=pdf_paths, job_id=job_id
        )

        ## Step 2 - generate answer to all the questions
        qa_dict = {}
        for question in questions:
            agent = Agent(job_id=job_id)
            answer = agent.get_response(question=question)
            qa_dict[question] = answer

        return qa_dict
