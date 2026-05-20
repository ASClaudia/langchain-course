from typing import Any, List

from pydantic import BaseModel, Field

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama
# from langchain_openai import ChatOpenAI
from tavily import TavilyClient
from langchain_tavily import TavilySearch

load_dotenv()
tavily = TavilyClient()

class Source(BaseModel):
    """Schema for a source used by the agent"""
    url: str = Field(description="Source URL")

class AgentResponse(BaseModel):
    """Schema for agent response with answer and sources"""
    answer: str = Field(description="The agent's answer to the user's query")
    # if we don't provide sources when instantiating the AgentResponse, it will be an empty list by default
    sources: List[Source] = Field(default_factory=list, description="List of sources used to generate the answer")

@tool
def search(query: str) -> dict[str, Any]:
    """
    Tool that searches over internet
    :param query: The query to search for
    :return: The search result
    """
    print(f"Searching for {query}")
    # return "Bucharest weather is sunny"
    return tavily.search(query=query)


llm = ChatOllama(model="gpt-oss", temperature=0.3)
tools = [search, TavilySearch()]

agent = create_agent(model=llm, tools=tools, response_format=AgentResponse)


def main():
    print("Hello from langchain-course!")

    result = agent.invoke(
        {"messages": HumanMessage(content="search for 3 job postings for an ai engineer using langchain in the "
                                          "Bucharest area on linkedin and list their details")}
    )

    # What happened here - ReAct Agent Architecture in action -->
    # The first llm received the input, and decided what tool to use (from the ones available) and with which arguments,
    # then it called the search tool, then sent the result of the search tool to the llm again, given the initial user
    # query, the first llm output and the search tool call with its arguments and output. This second llm did not choose
    # to execute a tool, but it chose to return the final answer, which was not exactly the output of the search tool.
    # You can check all of this in langsmith, and see how the agent is working step by step.
    # AWESOME! :)

    print(result)


if __name__ == "__main__":
    main()
