from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain.chains import LLMChain
from langchain.chat_models import AzureChatOpenAI

from src import prompts


def run_chain(prompt_text, input_text):
    """
    Runs LLM Chain and return its response

    :param prompt_text: the static prompt
    :param input_text: the input for the prompt
    :return: resp: the response from the LLM chain
    """

    rules_prompt = ChatPromptTemplate.from_template(prompt_text)
    chain = LLMChain(
        llm=self.llm, prompt=rules_prompt
    )

    resp = chain.run(input_text=input_text)
    return resp
