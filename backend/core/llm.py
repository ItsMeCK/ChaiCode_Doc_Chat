import logging
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from qdrant_client import QdrantClient, models, AsyncQdrantClient # Import Async client
from qdrant_client.http.exceptions import UnexpectedResponse # More specific exception
from typing import List # Import List

from .config import settings
# Assuming retriever.py initializes a reusable qdrant_client instance is not ideal for async
# Initialize an async client here for listing collections within the async function
# Note: Ensure Qdrant server supports async connections if using gRPC

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Initialize LLM ---
try:
    # Use the standard ChatOpenAI which supports async methods (ainvoke)
    llm = ChatOpenAI(
        model=settings.LLM_MODEL_NAME,
        openai_api_key=settings.OPENAI_API_KEY,
        temperature=0 # Lower temperature for more deterministic routing/answers
    )
    logging.info(f"ChatOpenAI initialized successfully with model: {settings.LLM_MODEL_NAME}")
except Exception as e:
    logging.error(f"Failed to initialize ChatOpenAI: {e}", exc_info=True)
    raise RuntimeError("Could not initialize LLM. Check API key and model name.") from e

# --- Namespace Selection ---

async def get_available_namespaces() -> list[str]:
    """Fetches the list of collection names from Qdrant asynchronously."""
    # Initialize async client within the function or manage it globally if appropriate
    # For simplicity, initializing here. Consider connection pooling for high load.
    async_qdrant_client = AsyncQdrantClient(
        url=settings.QDRANT_URL,
        api_key=settings.QDRANT_API_KEY,
        prefer_grpc=False, # Async client might prefer HTTP/REST
    )
    try:
        # Use the async client's method
        collections_response = await async_qdrant_client.get_collections()
        # collections_response.collections is a list of CollectionDescription objects
        namespaces = [col.name for col in collections_response.collections]
        logging.info(f"Fetched available namespaces from Qdrant: {namespaces}")
        return namespaces
    except UnexpectedResponse as e:
        # Handle cases where Qdrant might be unreachable or respond unexpectedly
        logging.error(f"Qdrant error fetching collections: Status {e.status_code} - {e.content.decode() if e.content else 'No content'}", exc_info=False) # Log less detail for common errors
        return []
    except Exception as e:
        logging.error(f"Failed to fetch collections from Qdrant: {e}", exc_info=True)
        return [] # Return empty list on error
    finally:
         # Ensure the async client is closed if initialized locally
         # Qdrant client might have a close method or rely on context managers
         # await async_qdrant_client.close() # Check documentation for proper closing
         pass # Assuming context management handles closure if applicable


async def get_namespace_from_query(query: str, max_namespaces: int = 2) -> List[str]:
    """
    Uses the LLM to determine the relevant Qdrant collection names (namespaces) for a query,
    using a dynamically fetched list of available namespaces. Returns up to max_namespaces.

    Args:
        query: The user's input query.
        max_namespaces: The maximum number of relevant namespaces to return.

    Returns:
        A list of determined namespace names (strings). Empty list if determination fails.
    """
    logging.info(f"Determining up to {max_namespaces} namespaces for query: '{query}'")

    # 1. Fetch current namespaces from Qdrant
    available_namespaces = await get_available_namespaces()
    if not available_namespaces:
        logging.error("No namespaces found in Qdrant. Cannot perform routing.")
        return [] # Cannot route without knowing available collections

    namespace_list_str = ", ".join(available_namespaces)

    # 2. Create the dynamic prompt - *** MODIFIED TO ASK FOR MULTIPLE ***
    # Instruct the LLM to return a comma-separated list
    namespace_system_prompt = f"""
You are an expert assistant specializing in the Chai Code documentation hosted at docs.chaicode.com.
Your task is to identify the relevant documentation sections (namespaces) for a given user query.
The available namespaces (representing documentation sections currently in the database) are: {namespace_list_str}.

Analyze the user's query and identify the primary topics or technologies it relates to within the Chai Code context.
List ALL relevant namespaces from the provided list, separated by commas.
If only one namespace is relevant, list only that one. If multiple are relevant, list them all.
Prioritize the most relevant namespaces first if possible.
Respond ONLY with the comma-separated list of namespace names. Do not add any explanation, preamble, or formatting.

Example (assuming 'chai_aur_python', 'chai_aur_django', 'chai_aur_git', 'chai_aur_react' are in the list):
User Query: How do I set up a virtual environment in Python and use it with Django?
Your Response: chai_aur_python, chai_aur_django

User Query: Explain CSS selectors.
Your Response: chai_aur_css

User Query: How does git branching work?
YourResponse: chai_aur_git

User Query: What is React state and how does it compare to context?
YourResponse: chai_aur_react
"""

    namespace_prompt = ChatPromptTemplate.from_messages([
        ("system", namespace_system_prompt),
        ("human", "{query}"),
    ])

    # Chain for namespace selection (structure remains the same)
    namespace_chain = namespace_prompt | llm | StrOutputParser()

    # 3. Invoke LLM and Process the list
    try:
        # Use ainvoke as we are in an async context
        llm_response = await namespace_chain.ainvoke({"query": query})
        llm_response = llm_response.strip()

        # Split the comma-separated response and clean up each namespace
        potential_namespaces = [ns.strip().lower() for ns in llm_response.split(',') if ns.strip()]

        # Validate against available namespaces and limit the count
        valid_namespaces = []
        for ns in potential_namespaces:
            if ns in available_namespaces:
                valid_namespaces.append(ns)
            else:
                logging.warning(f"LLM suggested namespace '{ns}' which is not in the available list: {available_namespaces}")

            if len(valid_namespaces) >= max_namespaces:
                break # Stop once we reach the desired limit

        if not valid_namespaces:
             logging.warning(f"LLM did not return any valid namespaces from the list for query: '{query}'. LLM response: '{llm_response}'. Falling back.")
             # Fallback strategy: Try a default or return empty list
             if "getting_started" in available_namespaces:
                 logging.info("Falling back to 'getting_started'.")
                 return ["getting_started"]
             return [] # Indicate failure

        logging.info(f"Determined valid namespaces: {valid_namespaces}")
        return valid_namespaces

    except Exception as e:
        logging.error(f"Error determining namespaces for query '{query}': {e}", exc_info=True)
        return [] # Indicate failure on exception

# --- Answer Generation (RAG - No changes needed here) ---
RAG_SYSTEM_PROMPT = """
You are an expert assistant knowledgeable about the Chai Code documentation.
Answer the user's query based *only* on the provided context.
If the context does not contain the answer, state clearly that you cannot answer based on the provided information.
Do not make up information or answer based on prior knowledge outside the context.
Be concise and helpful. Format code snippets appropriately using markdown.

Context:
{context}
"""

rag_prompt = ChatPromptTemplate.from_messages([
    ("system", RAG_SYSTEM_PROMPT),
    ("human", "User Query: {query}"),
])

# Chain for answer generation
rag_chain = rag_prompt | llm | StrOutputParser()

async def get_answer_from_context(query: str, retrieved_docs: list) -> str:
    """
    Generates an answer using the LLM based on the query and retrieved context.
    (No changes from previous version needed here)
    """
    logging.info(f"Generating answer for query: '{query}' using {len(retrieved_docs)} retrieved documents.")
    context_str = "\n\n---\n\n".join([doc.page_content for doc in retrieved_docs])

    if not context_str:
        logging.warning("No context provided to generate answer.")
        # Provide a clearer message if no context was found after search
        return "I looked through the relevant documentation section(s), but couldn't find specific information to answer your question. You might want to try rephrasing or asking about a different topic."

    try:
        answer = await rag_chain.ainvoke({"query": query, "context": context_str})
        logging.info("Answer generated successfully.")
        return answer
    except Exception as e:
        logging.error(f"Error generating answer for query '{query}': {e}", exc_info=True)
        return "Sorry, I encountered an error while trying to generate an answer."

