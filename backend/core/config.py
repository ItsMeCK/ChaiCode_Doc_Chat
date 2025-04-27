import os
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

# Load environment variables from .env file relative to this config.py file
# This ensures it finds the .env file whether run from backend/ or scripts/
dotenv_path = os.path.join(os.path.dirname(__file__), '..', '..', '.env')
load_dotenv(dotenv_path=dotenv_path)

class Settings(BaseSettings):
    """Loads application settings from environment variables."""
    # OpenAI
    OPENAI_API_KEY: str
    LLM_MODEL_NAME: str = "gpt-4" # Default LLM model

    # Qdrant
    QDRANT_URL: str
    QDRANT_API_KEY: str | None = None # Optional API key for Qdrant Cloud

    # Embeddings
    EMBEDDING_MODEL_NAME: str = "text-embedding-3-small" # Default embedding model

    # RAG / Ingestion Settings
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 100
    SIMILARITY_TOP_K: int = 3 # Number of documents to retrieve

    # REMOVED: KNOWN_NAMESPACES list - will be fetched dynamically from Qdrant

    class Config:
        # Specifies the .env file encoding
        env_file_encoding = 'utf-8'
        # Makes the Settings class case-sensitive regarding environment variables
        case_sensitive = True
        # Allows extra fields not defined in the model (use with caution)
        # extra = 'allow'

# Instantiate the settings
settings = Settings()

# Basic validation after loading
if not settings.OPENAI_API_KEY or settings.OPENAI_API_KEY == "your_openai_api_key_here":
    raise ValueError("OPENAI_API_KEY is not set or is using the default placeholder. Please configure it in your .env file.")

if not settings.QDRANT_URL:
     raise ValueError("QDRANT_URL is not set. Please configure it in your .env file.")

