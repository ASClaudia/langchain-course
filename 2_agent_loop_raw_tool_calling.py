from dotenv import load_dotenv

load_dotenv()

import ollama
from langsmith import traceable

MAX_ITERATIONS = 10
MODEL = "qwen3:1.7b"

# --- Tools (LangChain @tool decorator) ---
@traceable(run_type="tool")
def get_product_price(product: str) -> float:
    """Look up the price of a product in the catalog."""
    print(f"    >> Executing get_product_price(product='{product}')")

    prices = {"laptop": 1299.99, "headphones": 149.95, "keyboard": 89.50}
    return prices.get(product, 0)

# Instead of tool decorator, we now have to manually trace the function for LangSmith,
# and we lose the automatic schema generation from the function signature and docstring that @tool was doing for us.
@traceable(run_type="tool")
def apply_discount(price: float, discount_tier: str) -> float:
    """Apply a discount tier to a price and return the final price.
    Available tiers: bronze, silver, gold."""
    print(f"    >> Executing apply_discount(price={price}, discount_tier='{discount_tier}')")
    discount_percentages = {"bronze": 5, "silver": 12, "gold": 23}
    discount = discount_percentages.get(discount_tier, 0)
    return round(price * (1 - discount / 100), 2)

# Difference 2: Without @tool, we must MANUALLY define the JSON schema for each function.
# This is exactly what LangChain's @tool decorator generates automatically
# from the function's type hints and docstring.

# --- Agent Loop ---

tools_for_llm = [
    {
        "type": "function",
        "function": {
            "name": "get_product_price",
            "description": "Look up the price of a product in the catalog.",
            "parameters": {
                "type": "object",
                "properties": {
                    "product": {
                        "type": "string",
                        "description": "The product name, e.g. 'laptop', 'headphones', 'keyboard'",
                    },
                },
                "required": ["product"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "apply_discount",
            "description": "Apply a discount tier to a price and return the final price. Available tiers: bronze, silver, gold.",
            "parameters": {
                "type": "object",
                "properties": {
                    "price": {"type": "number", "description": "The original price"},
                    "discount_tier": {
                        "type": "string",
                        "description": "The discount tier: 'bronze', 'silver', or 'gold'",
                    },
                },
                "required": ["price", "discount_tier"],
            },
        },
    },
]


# NOTE: Ollama can also auto-generate these schemas if you pass the functions
# directly as tools (similar to LangChain's @tool decorator):
#   tools_for_llm = [get_product_price, apply_discount]
# However, this requires your docstrings to follow the Google docstring format
# so Ollama can parse parameter descriptions from the Args section. For example:
#   def get_product_price(product: str) -> float:
#       """Look up the price of a product in the catalog.
#
#       Args:
#           product: The product name, e.g. 'laptop', 'headphones', 'keyboard'.
#
#       Returns:
#           The price of the product, or 0 if not found.
#       """
# We keep the manual JSON version here so you can see what @tool hides from you.


# --- Helper: traced Ollama call ---
# Difference 3: Without LangChain, we must manually trace LLM calls for LangSmith. With LangChain, the tracing is
# automatic and unified across all providers, using init_chat_model

@traceable(name="Ollama Chat", run_type="llm")
def ollama_chat_traced(messages):
    return ollama.chat(model=MODEL, tools=tools_for_llm, messages=messages)


@traceable(name="Ollama Agent Loop")
def run_agent(question: str):

    tools = [get_product_price, apply_discount]

    # I need this dictionary because I'll get the llm response as a tool name and I need the function object to call it
    # With langchain I had name attribute for each tool, with raw tools creation, I don't
    # tools_dict = {t.name: t for t in tools}

    tools_dict = {t.__name__: t for t in tools}

    # I don't need this function call because I created ollama_chat_traced.
    # llm = init_chat_model(f"ollama:{MODEL}", temperature=0)
    # llm_with_tools = llm.bind_tools(tools)

    # the brain of the agent
    print(f"Question: {question}")
    print("=" * 60)
    """
    # The initial ReACT prompt pushed by the man who created LangChain
    """

    messages = [
        # I don't have SystemMessage as in LangChain
        {
            "role": "system",
            "content": (
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
        },
        # HumanMessage(content=question), - this was the implementation with LangChain

        # this is the raw implementation, for ollama the role of the user is called user, but for other sdks is called
        # human, HumanMessage from langchain did that for us.
        {
            "role": "user",
            "content": question
        }

    ]

    # the ReAct loop
    for iteration in range(1, MAX_ITERATIONS + 1):
        print(f"\n--- Iteration {iteration} ---")

        # LangChain implementation
        # ai_message = llm_with_tools.invoke(messages)
        # tool_calls = ai_message.tool_calls

        # Raw implementation
        # it's calling ollama sdk directly, the response is ollama response, not langchain ai message object
        response = ollama_chat_traced(messages=messages)
        ai_message = response.message
        tool_calls = ai_message.tool_calls

        # If no tool calls, this is the final answer - the LLM decides it has the final answer
        if not tool_calls:
            print(f"\nFinal Answer: {ai_message.content}")
            return ai_message.content

        # Process only the FIRST tool call — force one tool per iteration
        tool_call = tool_calls[0]

        # LangChain tools details extraction
        # tool_name = tool_call.get("name")
        # tool_args = tool_call.get("args", {})
        # tool_call_id = tool_call.get("id")

        # Difference 6: Attribute access (.function.name) instead of dict access (.get("name"))
        # tool call id is strictly required by OpenAI API, but not by Ollama sdk. LangChain was taking care of this
        # returning a unified dictionary structure as above.
        tool_name = tool_call.function.name
        tool_args = tool_call.function.arguments

        print(f"  [Tool Selected] {tool_name} with args: {tool_args}")

        # here is the tool extraction
        tool_to_use = tools_dict.get(tool_name)
        if tool_to_use is None:
            raise ValueError(f"Tool '{tool_name}' not found")

        # LangChain version - the runnable interface
        # observation = tool_to_use.invoke(tool_args)

        # Raw version - we need to call the function directly, no runnable interface
        # Difference 7: Direct function call instead of tool.invoke()
        observation = tool_to_use(**tool_args)

        print(f"  [Tool Result] {observation}")

        # LangChain appending of the llm response to the entire chain of responses to create the ReAct loop
        messages.append(ai_message)
        messages.append(
            {
                "role": "tool",
                # "tool_name": tool_name, - # optional, but I don't have a standard for this when not using langchain
                "content": str(observation),
            }
        )

    print("ERROR: Max iterations reached without a final answer")
    return None


if __name__ == "__main__":
    print("Hello LangChain Agent (.bind_tools)!")
    print()
    # here is the query
    run_agent("What is the price of a laptop after applying a gold discount?")