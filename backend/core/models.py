from pydantic import BaseModel

class ChatMessage(BaseModel):
    """Request model for incoming chat messages."""
    query: str

class ChatResponse(BaseModel):
    """Response model for outgoing chat messages."""
    answer: str
    namespace_used: str | None = None # Optionally return the namespace used

