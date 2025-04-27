import logging
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import Qdrant
from qdrant_client import QdrantClient, AsyncQdrantClient, models # Added AsyncQdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse # For specific error handling
from langchain_core.documents import Document
from typing import List # Import List

from .config import settings

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Initialize Embeddings and Qdrant Clients ---
# Initialize embeddings globally as they are stateless and reusable
try:
    embeddings = OpenAIEmbeddings(
        model=settings.EMBEDDING_MODEL_NAME,
        openai_api_key=settings.OPENAI_API_KEY
    )
    logging.info(f"OpenAIEmbeddings initialized (Model: {settings.EMBEDDING_MODEL_NAME})")
except Exception as e:
    logging.error(f"Failed to initialize OpenAI embeddings: {e}", exc_info=True)
    raise RuntimeError("Could not initialize embeddings.") from e

# Initialize a synchronous client for potential use elsewhere or simpler checks if needed
try:
    sync_qdrant_client = QdrantClient(
        url=settings.QDRANT_URL,
        api_key=settings.QDRANT_API_KEY,
        prefer_grpc=False, # Often simpler for non-heavy use
    )
    logging.info(f"Synchronous Qdrant client initialized (URL: {settings.QDRANT_URL})")
    # Optional: Perform a quick health check or list collections on startup
    # sync_qdrant_client.list_collections()
except Exception as e:
     logging.error(f"Failed to initialize synchronous Qdrant client: {e}", exc_info=True)
     # Decide if this is critical - maybe the app can run without it initially?
     # raise RuntimeError("Could not initialize synchronous Qdrant client.") from e
     sync_qdrant_client = None # Allow running without it if async is primary

# Note: The async client is initialized within get_available_namespaces in llm.py
# If needed elsewhere, consider a shared async client management strategy (e.g., lifespan events in FastAPI)


# --- Search Function ---
async def search_documents(query: str, namespaces: List[str]) -> list[Document]:
    """
    Performs similarity search across multiple Qdrant collections (namespaces)
    and combines the results.

    Args:
        query: The user's search query.
        namespaces: A list of Qdrant collection names to search within.

    Returns:
        A combined list of Langchain Document objects from all searched namespaces,
        or an empty list if the search fails or finds no results.
    """
    if not namespaces:
        logging.warning("No namespaces provided for search. Aborting search.")
        return []

    # If sync_qdrant_client failed to initialize, we cannot proceed.
    if not sync_qdrant_client:
         logging.error("Synchronous Qdrant client not available for search.")
         return []

    logging.info(f"Performing similarity search for query: '{query}' in namespaces: {namespaces}")

    all_retrieved_docs = []
    processed_doc_sources = set() # Keep track of source URLs to avoid duplicates

    for namespace in namespaces:
        if not namespace: # Skip empty namespace names in the list
            continue
        logging.debug(f"Searching in namespace: '{namespace}'")
        try:
            # Initialize Qdrant store object targeting the specific collection for retrieval
            qdrant_store = Qdrant(
                client=sync_qdrant_client, # Use the initialized synchronous client
                collection_name=namespace,
                embeddings=embeddings, # Use the globally initialized embeddings
            )

            # Perform similarity search using Langchain's retriever interface
            # Adjust k: maybe fetch fewer from each if searching many? Or keep K and merge/rank later.
            # Let's fetch K from each for now.
            retriever = qdrant_store.as_retriever(
                search_type="similarity",
                search_kwargs={'k': settings.SIMILARITY_TOP_K} # Get top K results per namespace
            )

            # Use ainvoke for async execution of the search
            retrieved_docs = await retriever.ainvoke(query)
            logging.debug(f"Retrieved {len(retrieved_docs)} documents from namespace '{namespace}'.")

            # Add unique documents to the combined list
            for doc in retrieved_docs:
                source = doc.metadata.get('source', None)
                # Add doc only if source is unique or missing (to avoid exact duplicates from same page if somehow retrieved twice)
                if not source or source not in processed_doc_sources:
                    all_retrieved_docs.append(doc)
                    if source:
                        processed_doc_sources.add(source)

        except UnexpectedResponse as e:
             # Handle specific Qdrant errors, e.g., collection not found
             logging.error(f"Qdrant error during search in namespace '{namespace}': Status {e.status_code} - {e.content.decode() if e.content else 'No content'}", exc_info=False)
             # Continue to the next namespace
             continue
        except Exception as e:
            # Catch other potential errors during search (network issues, embedding errors)
            logging.error(f"Error during similarity search in namespace '{namespace}' for query '{query}': {e}", exc_info=True)
            # Continue to the next namespace
            continue

    # Optional: Re-rank the combined results based on relevance score if available and needed.
    # For simplicity, we just return the concatenated (unique by source) list.
    logging.info(f"Retrieved a total of {len(all_retrieved_docs)} unique documents across {len(namespaces)} searched namespaces.")
    return all_retrieved_docs

