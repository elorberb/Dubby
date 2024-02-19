from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chat_models import AzureChatOpenAI

from src import prompts


def run_chain(prompt_text, input_text, llm):
    """
    Runs LLM Chain and return its response

    :param llm: the llm engine running the chain
    :param prompt_text: the static prompt
    :param input_text: the input for the prompt
    :return: resp: the response from the LLM chain
    """

    rules_prompt = PromptTemplate.from_template(prompt_text)
    chain = LLMChain(
        llm=llm, prompt=rules_prompt
    )

    resp = chain.run(input_text=input_text)
    return resp
