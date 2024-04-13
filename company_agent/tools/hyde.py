from company_agent.constant import EMBEDDINGS_MODEL
from langchain.chains import HypotheticalDocumentEmbedder, LLMChain
from langchain_openai import OpenAIEmbeddings
from .templates import HYDE_TEMPLATE_USER
from company_agent.constant import LANUAGE_MODEL, SEED_VALUE
from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate as HumanTemplate,
)
from langchain_openai import ChatOpenAI


def get_hyde_embedding():
    base_embeddings = OpenAIEmbeddings(model=EMBEDDINGS_MODEL)
    model = ChatOpenAI(
        model=LANUAGE_MODEL,
        max_tokens=300,
        temperature=0,
        model_kwargs={"seed": SEED_VALUE},
    )
    chat_prompt = ChatPromptTemplate.from_messages(
        [HumanTemplate.from_template(HYDE_TEMPLATE_USER)]
    )
    llm_chain = LLMChain(llm=model, prompt=chat_prompt, verbose=True)
    embeddings = HypotheticalDocumentEmbedder(
        llm_chain=llm_chain, base_embeddings=base_embeddings
    )
    return embeddings
