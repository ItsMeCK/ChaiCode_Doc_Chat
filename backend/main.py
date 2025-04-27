import logging
import os
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles # If you add CSS/JS later
from typing import List # Import List

# Import core components
from core.config import settings
from core.models import ChatMessage, ChatResponse
from core.llm import get_namespace_from_query, get_answer_from_context
from core.retriever import search_documents

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize FastAPI app
app = FastAPI(title="Chai Code RAG API")

# --- Setup Templates ---
# Determine the base directory for templates relative to this file
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATE_DIR = os.path.join(BASE_DIR, "templates")

# Check if template directory exists
if not os.path.isdir(TEMPLATE_DIR):
    logging.error(f"Template directory not found at: {TEMPLATE_DIR}")
    # Depending on requirements, you might exit or continue without templates
    # For this app, templates are essential for the homepage.
    raise RuntimeError(f"Template directory not found: {TEMPLATE_DIR}")

templates = Jinja2Templates(directory=TEMPLATE_DIR)
logging.info(f"Templates directory set to: {TEMPLATE_DIR}")

# --- Setup Static Files (Optional) ---
# STATIC_DIR = os.path.join(BASE_DIR, "static")
# if os.path.isdir(STATIC_DIR):
#     app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
#     logging.info(f"Static files directory set to: {STATIC_DIR}")
# else:
#     logging.warning(f"Static directory not found at: {STATIC_DIR}. Static files will not be served.")


# --- API Endpoints ---

@app.get("/", response_class=HTMLResponse)
async def get_homepage(request: Request):
    """Serves the simple HTML chat interface."""
    logging.info("Serving homepage (index.html)")
    try:
        return templates.TemplateResponse("index.html", {"request": request})
    except Exception as e:
        logging.error(f"Error rendering template index.html: {e}", exc_info=True)
        # Provide a fallback response or raise a more specific HTTP error
        raise HTTPException(status_code=500, detail="Could not load homepage template.")


@app.post("/chat", response_model=ChatResponse)
async def handle_chat(message: ChatMessage):
    """Handles incoming chat messages, performs RAG, and returns the answer."""
    user_query = message.query
    logging.info(f"Received chat query: {user_query}")

    if not user_query:
        logging.warning("Received empty query.")
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    # 1. Determine Potentially Multiple Namespaces
    # Pass max_namespaces=2 to get up to 2 relevant namespaces
    namespaces = await get_namespace_from_query(user_query, max_namespaces=2)
    if not namespaces:
        # Handle routing failure (e.g., return default message)
        logging.warning(f"Could not determine any valid namespaces for query: '{user_query}'.")
        answer = "Sorry, I couldn't determine the relevant documentation section(s) for your query. Please try rephrasing."
        return ChatResponse(answer=answer, namespaces_used=None) # Return None or empty list for namespaces_used

    # 2. Retrieve Documents from determined namespaces
    retrieved_docs = await search_documents(user_query, namespaces)

    if not retrieved_docs:
        logging.info(f"No relevant documents found in namespace(s) {namespaces} for query: '{user_query}'.")
        # Respond that no context was found
        answer = "I looked through the relevant documentation section(s), but couldn't find specific information about that. You could try rephrasing your question."
        return ChatResponse(answer=answer, namespaces_used=namespaces) # Return the namespaces we searched

    # 3. Generate Answer using combined context
    answer = await get_answer_from_context(user_query, retrieved_docs)

    logging.info(f"Generated answer for query: '{user_query}' using namespace(s): {namespaces}")
    # Return the list of namespaces used for retrieval
    return ChatResponse(answer=answer, namespaces_used=namespaces)

# --- Health Check Endpoint (Optional) ---
@app.get("/health")
async def health_check():
    """Simple health check endpoint."""
    return {"status": "ok"}

# --- Run Instruction (for local development) ---
# To run this app:
# 1. Navigate to the 'backend' directory in your terminal.
# 2. Run: uvicorn main:app --reload --host 0.0.0.0 --port 8000
