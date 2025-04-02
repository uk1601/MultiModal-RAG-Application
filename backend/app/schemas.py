# app/schemas.py
from pydantic import BaseModel, Field
from typing import List

class ChatRequest(BaseModel):
    document_id: str = Field(..., description="Unique identifier for the document.")
    message: str = Field(..., description="The user's message to the assistant.")

    class Config:
        schema_extra = {
            "example": {
                "document_id": "doc123",
                "message": "What is the capital of France?"
            }
        }

class ChatResponse(BaseModel):
    role: str = Field(..., description="Role of the message sender (e.g., 'assistant').")
    content: str = Field(..., description="Markdown-formatted response from the assistant.")

    class Config:
        schema_extra = {
            "example": {
                "role": "assistant",
                "content": "# Assistant Response\n\n**You said:** What is the capital of France?\n\n![Sample Image](https://via.placeholder.com/150)\n\n*This is a dummy response in Markdown.*"
            }
        }

class ChatHistoryResponse(BaseModel):
    document_id: str = Field(..., description="Unique identifier for the document.")
    messages: List[ChatResponse] = Field(..., description="List of all chat messages exchanged.")

    class Config:
        schema_extra = {
            "example": {
                "document_id": "doc123",
                "messages": [
                    {
                        "role": "user",
                        "content": "What is the capital of France?"
                    },
                    {
                        "role": "assistant",
                        "content": "# Assistant Response\n\n**You said:** What is the capital of France?\n\n![Sample Image](https://via.placeholder.com/150)\n\n*This is a dummy response in Markdown.*"
                    }
                ]
            }
        }

class DocumentProcessRequest(BaseModel):
    document_id: str = Field(..., description="Unique identifier for the document to process.")

    class Config:
        schema_extra = {
            "example": {
                "document_id": "doc123"
            }
        }