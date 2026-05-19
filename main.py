from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama

from langchain_core.prompts import PromptTemplate

load_dotenv()


def main():

    information="""
    An AI boom[1][2] is a period of rapid growth in the field of artificial intelligence (AI). The most recent boom happened in the early 2020s[3] before seeing increased acceleration and media coverage. Examples of this include generative AI technologies, such as large language models (LLM) and AI image generators developed by companies like OpenAI, Google and Anthropic, as well as scientific advances, such as protein folding prediction led by Google DeepMind and Google AI. This period is sometimes referred to as an AI spring, a term used to differentiate it from previous AI winters.[4][5] As of 2025, ChatGPT has emerged as the 4th-most visited website globally, surpassed only by Google, YouTube, and Facebook.[6][7]
    The long-term trajectory of the AI boom remains a subject of intense debate. Investors and technologists frequently speculate whether massive capital expenditures in data center infrastructure represent a sustainable paradigm shift or a speculative market bubble.[8] However, as these technologies integrate further into society, their ultimate impact will depend on how effectively global industries navigate systemic disruptions, environmental costs, and evolving legal frameworks.[9][10]
    """

    summary_template = """
    Given the information {information} about an event, I want you to create:
    1. A short summary.
    2. 2 weird facts about it. 
    """

    # temperature between 0 and 0.3 will set the llm response to be deterministic, and probably repeatable, while high value between 0.8 and 1, will get very creative responses (e.g good for poetry)
    # llm = ChatOpenAI(temperature=0, model="gpt-5")
    llm = ChatOllama(temperature=0, model="gpt-oss")

    summary_prompt_template = PromptTemplate(input_variables=[information], template=summary_template)
    
    chain = summary_prompt_template | llm
    response = chain.invoke(input={"information": information})

    print(f"The response is: {response}")
    print("Claudia says Hello from langchain-course!")


if __name__ == "__main__":
    main()