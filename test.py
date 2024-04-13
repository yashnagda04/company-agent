from company_agent.app import CompanyAgent
import sys, os
import json
from datetime import datetime

questions = [
    "What is the name of the company?",
    "Who is the CEO of the company?",
    "What is their vacation policy?",
    "What is the termination policy?",
]

document_paths = ["data/handbook.pdf"]


def save_dict(qa_dict):
    output_dir = "output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(output_dir, f"qa_{timestamp}.json")
    with open(filename, "w") as f:
        json.dump(qa_dict, f, ensure_ascii=False, indent=4)


def main():
    agent = CompanyAgent()
    qa_dict = agent.process(questions=questions, pdf_paths=document_paths)
    save_dict(qa_dict)


if __name__ == "__main__":
    main()
