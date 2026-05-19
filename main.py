from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama
# from langchain_openai import ChatOpenAI

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


llm = ChatOllama(model="gemma4", temperature=0.8)
tools = [search]

agent = create_agent(model=llm, tools=tools)


def main():
    print("Hello from langchain-course!")

    result = agent.invoke(
        {"messages": HumanMessage(content="What is the weather in Bucharest?")}
    )

    # What happened here - ReAct Agent Architecture in action -->
    # The first llm received the input, and decided what tool to use (from the ones available) and with which arguments,
    # then it called the search tool, then sent the result of the search tool to the llm again, given the initial user
    # query, the first llm output and the search tool call with arguments and output. This second llm did not choose to
    # execute a tool, but it chose to return the final answer, which was not exactly the output of the search tool.
    # You can check all of this in langsmith, and see how the agent is working step by step.
    # AWESOME! :)

    print(result)


if __name__ == "__main__":
    main()
