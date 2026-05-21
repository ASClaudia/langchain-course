from dotenv import load_dotenv

load_dotenv()

from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain.messages import HumanMessage, SystemMessage, ToolMessage
from langsmith import traceable

MAX_ITERATIONS = 10
MODEL = "qwen3:1.7b"

# --- Tools (LangChain @tool decorator) ---

# The functions docstrings are very important in order for the llm to know which function to call. Same for the type
# hinting for the arguments and return value. The tool decorator it takes all the information, and it formats it very
# nicely, according to every model provider's requirements. It creates one single interface to pass all of this meta
# information about the tools we want to pass to the model so we'll be able to use function calling with this model.


@tool
def get_product_price(product: str) -> float:
    """Look up the price of a product in the catalog."""
    print(f"    >> Executing get_product_price(product='{product}')")

    prices = {"laptop": 1299.99, "headphones": 149.95, "keyboard": 89.50}
    return prices.get(product, 0)

@tool
def apply_discount(price: float, discount_tier: str) -> float:
    """Apply a discount tier to a price and return the final price.
    Available tiers: bronze, silver, gold."""
    print(f"    >> Executing apply_discount(price={price}, discount_tier='{discount_tier}')")
    discount_percentages = {"bronze": 5, "silver": 12, "gold": 23}
    discount = discount_percentages.get(discount_tier, 0)
    return round(price * (1 - discount / 100), 2)

# --- Agent Loop ---

@traceable(name="LangChain Agent Loop")
def run_agent(question: str):

    tools = [get_product_price, apply_discount]

    # I need this dictionary because I'll get the llm response as a tool name and I need the function object to call it

    tools_dict = {t.name: t for t in tools}

    # This function call is so convenient because I don't need to import the class of the chat model itself, I just
    # need to know the provider and the model name and it will initialize it for me with all the right parameters for
    # that provider.

    llm = init_chat_model(f"ollama:{MODEL}", temperature=0)

    # this works only for function calling supporting llms
    llm_with_tools = llm.bind_tools(tools)

    # the brain of the agent
    print(f"Question: {question}")
    print("=" * 60)

    messages = [
        SystemMessage(
            content=(
                "You are a helpful shopping assistant. "
                "You have access to a product catalog tool "
                "and a discount tool.\n\n"
                "STRICT RULES — you must follow these exactly:\n"
                "1. NEVER guess or assume any product price. "
                "You MUST call get_product_price first to get the real price.\n"
                "2. Only call apply_discount AFTER you have received "
                "a price from get_product_price. Pass the exact price "
                "returned by get_product_price — do NOT pass a made-up number.\n"
                "3. NEVER calculate discounts yourself using math. "
                "Always use the apply_discount tool.\n"
                "4. If the user does not specify a discount tier, "
                "ask them which tier to use — do NOT assume one."
            )
        ),
        HumanMessage(content=question),
    ]

    # the ReAct loop
    for iteration in range(1, MAX_ITERATIONS + 1):
        print(f"\n--- Iteration {iteration} ---")

        # here is the thought process
        ai_message = llm_with_tools.invoke(messages)

        # here is the decision to call tools or not
        tool_calls = ai_message.tool_calls

        # If no tool calls, this is the final answer - the LLM decides it has the final answer
        if not tool_calls:
            print(f"\nFinal Answer: {ai_message.content}")
            return ai_message.content

        # Process only the FIRST tool call — force one tool per iteration
        # LLMs can call these days multiple tools at once
        # here is the action to call the first tool
        tool_call = tool_calls[0]
        tool_name = tool_call.get("name")
        tool_args = tool_call.get("args", {})
        tool_call_id = tool_call.get("id")

        print(f"  [Tool Selected] {tool_name} with args: {tool_args}")

        # here is the tool extraction
        tool_to_use = tools_dict.get(tool_name)
        if tool_to_use is None:
            raise ValueError(f"Tool '{tool_name}' not found")

        # here is the tool execution and the observation (the tool result)
        observation = tool_to_use.invoke(tool_args)

        print(f"  [Tool Result] {observation}")

        # this is where the ai decision to run the tool (and exactly what tool id) and with which arguments and the
        # result from the tool is sent back to the next llm call (together with the entire messages list)
        messages.append(ai_message)
        messages.append(
            ToolMessage(content=str(observation), tool_call_id=tool_call_id)
        )

    print("ERROR: Max iterations reached without a final answer")
    return None


if __name__ == "__main__":
    print("Hello LangChain Agent (.bind_tools)!")
    print()
    # here is the query
    run_agent("What is the price of a laptop after applying a gold discount?")