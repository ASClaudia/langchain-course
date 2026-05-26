import os
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

from operator import itemgetter

load_dotenv()

print("Initializing components...")

embeddings = OllamaEmbeddings(model="nomic-embed-text")
llm = ChatOllama(model="gemma3:270m ", temperature=0.7)

vectorstore = PineconeVectorStore(
    index_name=os.environ["INDEX_NAME"], embedding=embeddings
)

# it returns a vectorestore with the same search capabilities as the vendor implementing it, it will have a search
# function
# I want to limit it when searching through vector store, to only 3 documents that are going to be used. if there are
# 10 documents relevant, I want to take the top 3

# this is the RETRIEVER from vectorstore of the top vectors similar to the user question.
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# this is the augmentation part - augmentation of our prompt
prompt_template = ChatPromptTemplate.from_template(
    """Answer the question based only on the following context:

{context}

Question: {question}

Provide a detailed answer:"""
)

# here we are creating the context from the retrieved documents, we are going to use it in the prompt template
def format_docs(docs):
    """Format retrieved documents into a single string."""
    return "\n\n".join(doc.page_content for doc in docs)


def retrieval_chain_without_lcel(query: str):
    """
    Simple retrieval chain without LCEL.
    Manually retrieves documents, formats them, and generates a response.

    Limitations:
    - Manual step-by-step execution
    - No built-in streaming support
    - No async support without additional code
    - Harder to compose with other chains
    - More verbose and error-prone
    """
    # Step 1: Retrieve relevant documents
    # retriever is a Runnable, which has an invoke method, and it's going ro run something which eventually runs
    # get_relevant_documents which gets our query and returns a list of documents which are going to be the most
    # relevant docs. this function is gonna be implemented by the vendor. (pinecone it has its own sdk, chrome another
    # and so on) We have also an async method.
    docs = retriever.invoke(query)

    # Step 2: Format documents into context string
    context = format_docs(docs)

    # Step 3: Format the prompt with context and question
    messages = prompt_template.format_messages(context=context, question=query)

    # Step 4: Invoke LLM with the formatted messages
    response = llm.invoke(messages)

    # Step 5: Return the content
    return response.content


# ============================================================================
# IMPLEMENTATION 2: With LCEL (LangChain Expression Language) - BETTER APPROACH
# ============================================================================
def create_retrieval_chain_with_lcel():
    """
    Create a retrieval chain using LCEL (LangChain Expression Language).
    Returns a chain that can be invoked with {"question": "..."} (a langchain runnable)

    Advantages over non-LCEL approach:
    - Declarative and composable: Easy to chain operations with pipe operator (|)
    - Built-in streaming: chain.stream() works out of the box - because of the `Runnable` interface
    - Built-in async: chain.ainvoke() and chain.astream() available - because of the `Runnable` interface
    - Batch processing: chain.batch() for multiple inputs - because of the `Runnable` interface
    - Type safety: Better integration with LangChain's type system - because of the `Runnable` interface
    - Less code: More concise and readable
    - Reusable: Chain can be saved, shared, and composed with other chains
    - Better debugging: LangChain provides better observability tools (better langsmith observability)
    """
    # retrieval_chain = (
    #     retriever # step 1 from the previous implementation, is a Runnable

    #     | format_docs # step 2 from the previous implementation, this is a function, which does not have an invoke
    #     # method, but this is going to work, because when we add regular python functions, they are automatically
    #     # converted into runnables lambdas RunnableLambda(format_docs)

    #     | prompt_template # step 3 from the previous implementation, it inherits a baseclass which is a Runnable, so
    #     # does have an invoke method and can be used with pipe operator in a langchain chain. Prompt template needs to
    #     # receive 2 args, the question and the context (a string), so we need to take the output of format_docs to
    #     # attribute it to the key of context, like in the implementation below.

    #     | llm # step 4 from the previous implementation

    #     | StrOutputParser() # step 5 from the previous implementation, to access the .content key of the response
    # )

    retrieval_chain = (
        # assign is going to create a new dictionary which is going to combine the original input dict
        # {"question": "What is Pinecone?"} with the new computed field which is explicitly mentioned here, so I'll get
        # a dict with {"question": "What is Pinecone?", "context": "the formatted context from the retrieved docs"}

        RunnablePassthrough.assign(
            # we add in the context field the value of this chain: itemgetter("question") | retriever | format_docs,
            # without changing the input question, we are just adding a new field to the input dict which is going to be
            # the context, and this context is going to be computed by taking the question from the input dict, passing
            # it through the retriever to get the relevant documents, and then passing those documents through the
            # format_docs function to get a formatted string that can be used as context in the prompt template.
            # itemgetter("question") is a equivalent to lambda x: x["question"]
            context=itemgetter("question") | retriever | format_docs
        )
        | prompt_template # step 3 from the previous implementation
        | llm # step 4 from the previous implementation
        | StrOutputParser() # step 5 from the previous implementation, to access the .content key of the response
    )
    return retrieval_chain


if __name__ == "__main__":
    print("Retrieving...")

    # Query
    query = "what is Pinecone in machine learning?"

    # ========================================================================
    # Option 0: Raw invocation without RAG
    # ========================================================================
    print("\n" + "=" * 70)
    print("IMPLEMENTATION 0: Raw LLM Invocation (No RAG)")
    print("=" * 70)
    result_raw = llm.invoke([HumanMessage(content=query)])
    print("\nAnswer:")
    print(result_raw.content)

    # ========================================================================
    # Option 1: Use implementation WITHOUT LCEL
    # ========================================================================

    # Very hard to trace it without langchain
    print("\n" + "=" * 70)
    print("IMPLEMENTATION 1: Without LCEL")
    print("=" * 70)
    result_without_lcel = retrieval_chain_without_lcel(query)
    print("\nAnswer:")
    print(result_without_lcel)

    # ========================================================================
    # Option 2: Use implementation WITH LCEL (Better Approach)
    # ========================================================================
    print("\n" + "=" * 70)
    print("IMPLEMENTATION 2: With LCEL - Better Approach")
    print("=" * 70)
    print("Why LCEL is better:")
    print("- More concise and declarative")
    print("- Built-in streaming: chain.stream()")
    print("- Built-in async: chain.ainvoke()")
    print("- Easy to compose with other chains")
    print("- Better for production use")
    print("=" * 70)

    chain_with_lcel = create_retrieval_chain_with_lcel()
    result_with_lcel = chain_with_lcel.invoke({"question": query})
    print("\nAnswer:")
    print(result_with_lcel)