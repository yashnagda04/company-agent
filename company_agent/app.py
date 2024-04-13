from dotenv import load_dotenv

load_dotenv()
from company_agent.agent.agent import Agent
from company_agent.rag.create_embedding import EmbeddingManager
import uuid
import os, json
from datetime import datetime


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

        output_path = self.save_dict(qa_dict=qa_dict, job_id=job_id)
        return output_path
    

    def save_dict(self, qa_dict, job_id):
        output_dir = "output"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M%S")
        filename = os.path.join(output_dir, f"{job_id}_{timestamp}.json")
        with open(filename, "w") as f:
            json.dump(qa_dict, f, ensure_ascii=False, indent=4)

        return filename
