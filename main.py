from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI

load_dotenv()


@tool
def search(query: str) -> str:
    """
    Tool that searches over internet
    :param query: The query to search for
    :return: The search result
    """
    print(f"Searching for {query}")
    return "Bucharest weather is sunny"


llm = ChatOllama(model="gpt-oss")
tools = [search]

agent = create_agent(model=llm, tools=tools)


def main():
    print("Hello from langchain-course!")

    result = agent.invoke(
        {"messages": HumanMessage(content="What is the weather in Bucharest?")}
    )

    print(result)


if __name__ == "__main__":
    main()
