from pydantic import BaseModel
from typing import List, Optional # Import List and Optional

class ChatMessage(BaseModel):
    """Request model for incoming chat messages."""
    query: str

class ChatResponse(BaseModel):
    """Response model for outgoing chat messages."""
    answer: str
    namespaces_used: Optional[List[str]] = None # Use plural and make it a list
    evaluation: Optional[str] = None # Add field for evaluation result

