from company_agent.app import CompanyAgent

questions = [
    "What is the name of the company?",
    "Who is the CEO of the company?",
    "What is their vacation policy?",
    "What is the termination policy?",
]

document_paths = ["data/handbook.pdf"]


def main():
    agent = CompanyAgent()
    output_path = agent.process(questions=questions, pdf_paths=document_paths)
    print("Output:", output_path)


if __name__ == "__main__":
    main()
