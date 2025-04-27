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
# Use the same LLM for generation and evaluation for simplicity,
# but you could use a different model instance if needed (e.g., gpt-4 for evaluation)
try:
    llm = ChatOpenAI(
        model=settings.LLM_MODEL_NAME,
        openai_api_key=settings.OPENAI_API_KEY,
        temperature=0 # Lower temperature for more deterministic routing/answers/evaluation
    )
    logging.info(f"ChatOpenAI initialized successfully with model: {settings.LLM_MODEL_NAME}")
except Exception as e:
    logging.error(f"Failed to initialize ChatOpenAI: {e}", exc_info=True)
    raise RuntimeError("Could not initialize LLM. Check API key and model name.") from e

# --- Namespace Selection ---

async def get_available_namespaces() -> list[str]:
    """Fetches the list of collection names from Qdrant asynchronously."""
    async_qdrant_client = AsyncQdrantClient(
        url=settings.QDRANT_URL,
        api_key=settings.QDRANT_API_KEY,
        prefer_grpc=False,
    )
    try:
        collections_response = await async_qdrant_client.get_collections()
        namespaces = [col.name for col in collections_response.collections]
        logging.info(f"Fetched available namespaces from Qdrant: {namespaces}")
        return namespaces
    except UnexpectedResponse as e:
        logging.error(f"Qdrant error fetching collections: Status {e.status_code} - {e.content.decode() if e.content else 'No content'}", exc_info=False)
        return []
    except Exception as e:
        logging.error(f"Failed to fetch collections from Qdrant: {e}", exc_info=True)
        return []
    finally:
         pass # Handle client closing if necessary


async def get_namespace_from_query(query: str, max_namespaces: int = 2) -> List[str]:
    """
    Uses the LLM to determine the relevant Qdrant collection names (namespaces) for a query,
    using a dynamically fetched list of available namespaces. Returns up to max_namespaces.
    """
    logging.info(f"Determining up to {max_namespaces} namespaces for query: '{query}'")
    available_namespaces = await get_available_namespaces()
    if not available_namespaces:
        logging.error("No namespaces found in Qdrant. Cannot perform routing.")
        return []

    namespace_list_str = ", ".join(available_namespaces)
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
"""
    namespace_prompt = ChatPromptTemplate.from_messages([
        ("system", namespace_system_prompt),
        ("human", "{query}"),
    ])
    namespace_chain = namespace_prompt | llm | StrOutputParser()

    try:
        llm_response = await namespace_chain.ainvoke({"query": query})
        llm_response = llm_response.strip()
        potential_namespaces = [ns.strip().lower() for ns in llm_response.split(',') if ns.strip()]
        valid_namespaces = []
        for ns in potential_namespaces:
            if ns in available_namespaces:
                valid_namespaces.append(ns)
            else:
                logging.warning(f"LLM suggested namespace '{ns}' which is not in the available list: {available_namespaces}")
            if len(valid_namespaces) >= max_namespaces: break

        if not valid_namespaces:
             logging.warning(f"LLM did not return any valid namespaces from the list for query: '{query}'. LLM response: '{llm_response}'. Falling back.")
             if "getting_started" in available_namespaces:
                 logging.info("Falling back to 'getting_started'.")
                 return ["getting_started"]
             return []
        logging.info(f"Determined valid namespaces: {valid_namespaces}")
        return valid_namespaces
    except Exception as e:
        logging.error(f"Error determining namespaces for query '{query}': {e}", exc_info=True)
        return []

# --- Answer Generation (RAG) ---
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
rag_chain = rag_prompt | llm | StrOutputParser()

async def get_answer_from_context(query: str, retrieved_docs: list) -> str:
    """
    Generates an answer using the LLM based on the query and retrieved context.
    """
    logging.info(f"Generating answer for query: '{query}' using {len(retrieved_docs)} retrieved documents.")
    context_str = "\n\n---\n\n".join([doc.page_content for doc in retrieved_docs])

    if not context_str:
        logging.warning("No context provided to generate answer.")
        return "I looked through the relevant documentation section(s), but couldn't find specific information to answer your question. You might want to try rephrasing or asking about a different topic."

    try:
        answer = await rag_chain.ainvoke({"query": query, "context": context_str})
        logging.info("Answer generated successfully.")
        return answer
    except Exception as e:
        logging.error(f"Error generating answer for query '{query}': {e}", exc_info=True)
        return "Sorry, I encountered an error while trying to generate an answer."


# --- *** NEW: Answer Evaluation *** ---
EVALUATION_SYSTEM_PROMPT = """
You are an expert evaluator assessing the quality of an AI assistant's answer based on a user query and provided context from documentation.
Your goal is to determine if the generated answer is faithful to the context and relevant to the query.

**Context:**
{context}

**User Query:**
{query}

**Generated Answer:**
{answer}

**Evaluation Criteria:**
1.  **Faithfulness:** Is the answer accurately supported by the provided context? Does it avoid making up information not present in the context?
2.  **Relevance:** Does the answer directly address the user's query?

**Task:**
Based on the criteria above, evaluate the "Generated Answer". Respond with ONLY ONE of the following ratings:
- **Correct**: The answer is faithful to the context and relevant to the query.
- **Incorrect - Off-Topic**: The answer is not relevant to the user's query, even if factually correct based on context.
- **Incorrect - Hallucination**: The answer includes information not supported by the provided context.
- **Incorrect - Partial**: The answer is partially correct but misses key information from the context or is partially irrelevant.
- **Cannot Evaluate**: The context is insufficient or the query/answer is unclear, making evaluation impossible.
"""

evaluation_prompt = ChatPromptTemplate.from_messages([
    ("system", EVALUATION_SYSTEM_PROMPT),
    # No human message needed as all info is in the system prompt context
])

# Chain for evaluation - uses the same LLM instance for simplicity
evaluation_chain = evaluation_prompt | llm | StrOutputParser()

async def evaluate_answer(query: str, context_docs: list, answer: str) -> str:
    """
    Evaluates the generated answer based on the query and context using an LLM.

    Args:
        query: The original user query.
        context_docs: The list of documents retrieved as context.
        answer: The generated answer to evaluate.

    Returns:
        A string indicating the evaluation result (e.g., "Correct", "Incorrect - Hallucination").
    """
    if not answer or "Sorry, I encountered an error" in answer or "couldn't find specific information" in answer:
         logging.info("Skipping evaluation for error or non-committal answer.")
         return "Not Evaluated" # Don't evaluate error messages or non-answers

    logging.info(f"Evaluating answer for query: '{query}'")
    context_str = "\n\n---\n\n".join([doc.page_content for doc in context_docs])

    if not context_str:
        logging.warning("No context provided for evaluation.")
        return "Cannot Evaluate - No Context"

    try:
        evaluation_result = await evaluation_chain.ainvoke({
            "query": query,
            "context": context_str,
            "answer": answer
        })
        evaluation_result = evaluation_result.strip()
        logging.info(f"Evaluation result: {evaluation_result}")
        # Optional: Add more robust parsing/validation if the LLM doesn't strictly follow instructions
        valid_ratings = ["Correct", "Incorrect - Off-Topic", "Incorrect - Hallucination", "Incorrect - Partial", "Cannot Evaluate", "Cannot Evaluate - No Context", "Not Evaluated"]
        if evaluation_result not in valid_ratings:
             logging.warning(f"LLM returned unexpected evaluation rating: '{evaluation_result}'. Using as is.")
             # return "Evaluation Failed" # Or return the raw result

        return evaluation_result
    except Exception as e:
        logging.error(f"Error during answer evaluation for query '{query}': {e}", exc_info=True)
        return "Evaluation Failed"

