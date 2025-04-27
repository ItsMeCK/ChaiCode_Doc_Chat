import logging
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from qdrant_client import QdrantClient, models, AsyncQdrantClient # Import Async client
from qdrant_client.http.exceptions import UnexpectedResponse # More specific exception

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


async def get_namespace_from_query(query: str) -> str | None:
    """
    Uses the LLM to determine the relevant Qdrant collection name (namespace) for a query,
    using a dynamically fetched list of available namespaces.

    Args:
        query: The user's input query.

    Returns:
        The determined namespace name (string) or None if determination fails or is invalid.
    """
    logging.info(f"Determining namespace for query: '{query}'")

    # 1. Fetch current namespaces from Qdrant
    available_namespaces = await get_available_namespaces()
    if not available_namespaces:
        logging.error("No namespaces found in Qdrant. Cannot perform routing.")
        # Maybe attempt a default search without routing? Or return error.
        # For now, returning None to indicate routing failure.
        return None

    namespace_list_str = ", ".join(available_namespaces)

    # 2. Create the dynamic prompt for the LLM
    # This prompt asks the LLM to choose from the *currently available* collections
    namespace_system_prompt = f"""
You are an expert assistant specializing in the Chai Code documentation hosted at docs.chaicode.com.
Your task is to determine the single most relevant documentation section (namespace) for a given user query.
The available namespaces (representing documentation sections currently in the database) are: {namespace_list_str}.

Analyze the user's query and identify the primary topic or technology it relates to within the Chai Code context.
Respond with ONLY the name of the most relevant namespace from the provided list.
Do not add any explanation, preamble, or formatting. Just the namespace name.

Example (assuming 'chai_aur_python', 'chai_aur_css', 'chai_aur_git', 'chai_aur_react' are in the list):
User Query: How do I set up a virtual environment in Python?
Your Response: chai_aur_python

User Query: Explain CSS selectors.
Your Response: chai_aur_css

User Query: How does git branching work?
YourResponse: chai_aur_git

User Query: What is React state?
YourResponse: chai_aur_react
"""

    namespace_prompt = ChatPromptTemplate.from_messages([
        ("system", namespace_system_prompt),
        ("human", "{query}"),
    ])

    # Chain for namespace selection (structure remains the same)
    namespace_chain = namespace_prompt | llm | StrOutputParser()

    # 3. Invoke LLM and Validate the response
    try:
        # Use ainvoke as we are in an async context
        determined_namespace = await namespace_chain.ainvoke({"query": query})
        determined_namespace = determined_namespace.strip().lower() # Clean up output

        # Validate the output against the dynamically fetched list
        if determined_namespace in available_namespaces:
            logging.info(f"Determined namespace: {determined_namespace}")
            return determined_namespace
        else:
            logging.warning(f"LLM returned a namespace ('{determined_namespace}') not found in Qdrant list: {available_namespaces}. Query: '{query}'.")
            # Fallback strategy: Try a default or return None
            # Check if 'getting_started' exists as a safe fallback
            if "getting_started" in available_namespaces:
                logging.info("Falling back to 'getting_started' namespace.")
                return "getting_started"
            # If no fallback, indicate routing failure
            logging.warning("No suitable fallback namespace found.")
            return None
    except Exception as e:
        logging.error(f"Error determining namespace for query '{query}': {e}", exc_info=True)
        return None # Indicate failure on exception

# --- Answer Generation (RAG - No changes needed in this part) ---
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

    # Format the context for the prompt
    context_str = "\n\n---\n\n".join([doc.page_content for doc in retrieved_docs])

    if not context_str:
        logging.warning("No context provided to generate answer.")
        # Provide a clearer message if no context was found after search
        return "I looked through the relevant documentation section, but couldn't find specific information to answer your question. You might want to try rephrasing or asking about a different topic."

    try:
        # Use ainvoke for async execution
        answer = await rag_chain.ainvoke({"query": query, "context": context_str})
        logging.info("Answer generated successfully.")
        return answer
    except Exception as e:
        logging.error(f"Error generating answer for query '{query}': {e}", exc_info=True)
        return "Sorry, I encountered an error while trying to generate an answer."

