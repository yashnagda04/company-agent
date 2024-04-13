from company_agent.tools.vector_db_tool import get_vector_db_tool
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate as SystemTemplate,
    HumanMessagePromptTemplate as HumanTemplate,
    MessagesPlaceholder,
)
from langchain.prompts import PromptTemplate
from company_agent.constant import LANUAGE_MODEL, SEED_VALUE
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_openai_tools_agent
from .templates import COMPANY_AGENT_SYSTEM_TEMPLATE, COMPANY_AGENT_USER_TEMPLATE


class Agent:
    def __init__(self, job_id):
        self.job_id = job_id

    def get_response(self, question):
        agent = self.__get_agent()
        result = agent.invoke({"question": question})
        return result["output"]

    def __get_agent(self):
        tools = self.__get_tools()
        llm = self.__get_llm()
        prompt = self.__get_prompt()

        agent = create_openai_tools_agent(llm, tools, prompt)
        agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
        return agent_executor

    def __get_tools(self):
        vector_db_tool = get_vector_db_tool(job_id=self.job_id)
        tools = [vector_db_tool]
        return tools

    def __get_llm(self):
        llm = ChatOpenAI(
            model=LANUAGE_MODEL,
            temperature=0,
            max_tokens=4000,
            model_kwargs={"seed": SEED_VALUE},
        )
        return llm

    def __get_prompt(self):
        prompt = ChatPromptTemplate.from_messages(
            [
                SystemTemplate(
                    prompt=PromptTemplate(
                        input_variables=[], template=COMPANY_AGENT_SYSTEM_TEMPLATE
                    )
                ),
                HumanTemplate(
                    prompt=PromptTemplate(
                        input_variables=["question"],
                        template=COMPANY_AGENT_USER_TEMPLATE,
                    )
                ),
                MessagesPlaceholder(variable_name="agent_scratchpad"),
            ]
        )
        return prompt
