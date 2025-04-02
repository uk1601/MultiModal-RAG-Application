import json
import logging
import os
from typing import List

from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO)

def setup_chat_histories():
    if not os.path.exists(CHAT_HISTORY_DIR):
        os.makedirs(CHAT_HISTORY_DIR)
CHAT_HISTORY_DIR = "chat_histories"




class ChatRequest(BaseModel):
    document_id: str = Field(..., description="Unique identifier for the chat session.")
    message: str = Field(..., description="The user's message to the assistant.")

    class Config:
        schema_extra = {
            "example": {
                "document_id": "doc123",
                "message": "Hello, how are you?"
            }
        }


class ChatResponse(BaseModel):
    role: str = Field(..., description="Role of the message sender (e.g., 'assistant').")
    content: str = Field(..., description="Markdown-formatted response from the assistant.")

    class Config:
        schema_extra = {
            "example": {
                "role": "assistant",
                "content": "# Assistant Response\n\n**You said:** Hello, how are you?\n\n![Dummy Image](https://via.placeholder.com/150)\n\n*This is a dummy response in Markdown.*"
            }
        }

class NotesRequest(BaseModel):
    filename: str = Field(..., description="Name of the PDF file to summarize.")

class ChatHistoryResponse(BaseModel):
    document_id: str = Field(..., description="Unique identifier for the chat session.")
    messages: List[ChatResponse] = Field(..., description="List of all chat messages exchanged.")

    class Config:
        schema_extra = {
            "example": {
                "document_id": "doc123",
                "messages": [
                    {
                        "role": "user",
                        "content": "Hello, how are you?"
                    },
                    {
                        "role": "assistant",
                        "content": "# Assistant Response\n\n**You said:** Hello, how are you?\n\n![Dummy Image](https://via.placeholder.com/150)\n\n*This is a dummy response in Markdown.*"
                    }
                ]
            }
        }


# Utility Functions

def get_chat_history_file(document_id: str) -> str:
    """
    Returns the file path for a given document ID's chat history.
    """
    cwd = os.getcwd()
    directory = cwd + "/chat_histories/assignment3/pdfs/"
    js = cwd + f"/chat_histories/{document_id}_chat.json"
    history_path = os.path.join(directory, f"{document_id}_chat.json")
    #    if nested directory is not present, create it
    if not os.path.exists(directory):
        logging.info(f"Creating directory: {directory}")
        os.makedirs(directory)
    # if json file does not exist, create it
    if not os.path.exists(js):
        logging.info(f"Creating chat history file: {js}")
        with open(js, "w") as f:
            json.dump({}, f)


    return js


def load_chat_history(document_id: str) -> List[dict]:
    """
    Loads the chat history for a given document ID.
    """
    filepath = get_chat_history_file(document_id)
    if not os.path.exists(filepath):
        return []
    with open(filepath, "r", encoding="utf-8") as f:
        try:
            data = json.load(f)
            return data.get("messages", [])
        except json.JSONDecodeError:
            logging.error(f"Invalid JSON in {filepath}.")
            return []


def save_chat_history(document_id: str, messages: List[dict]):
    """
    Saves the chat history for a given document ID.
    """
    filepath = get_chat_history_file(document_id)
    data = {
        "document_id": document_id,
        "messages": messages
    }
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


from markdown2 import Markdown
from weasyprint import HTML
import os


def markdown_to_pdf(markdown_content, output_filename):
    # Convert Markdown to HTML
    logging.info(f'Converting Markdown to HTML...{markdown_content}')
    markdowner = Markdown()
    html_content = markdowner.convert(markdown_content)
    logging.info(f"Converted Markdown to HTML:\n{html_content}")
    # Wrap the HTML content in a basic HTML structure
    full_html = f"""
    <html>
    <head>
        <meta charset="utf-8">
        <style>
            body {{ font-family: Arial, sans-serif; }}
            h1, h2, h3 {{ color: #333; }}
            code {{ background-color: #f0f0f0; padding: 2px 4px; border-radius: 4px; }}
        </style>
    </head>
    <body>
        {html_content}
    </body>
    </html>
    """

    # Create a PDF from the HTML
    html = HTML(string=full_html)
    pdf_file = html.write_pdf()
    output_filename = output_filename
    # Save the PDF file
    logging.info(f"Saving PDF to: {output_filename}")
    # Wrap this in a try-except block to handle any potential errors
    try:
        with open(output_filename, 'wb') as f:
            f.write(pdf_file)
    except Exception as e:
        logging.error(f"Failed to save PDF: {str(e)}")


    # Return the absolute path of the saved PDF
    p = os.path.abspath(output_filename)
    logging.info(f"PDF saved to: {p}")
    return p