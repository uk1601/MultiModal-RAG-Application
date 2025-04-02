# app/services/rag_service.py
import logging

from llama_index.core.base.llms.types import CompletionResponse
from llama_index.llms.nvidia import NVIDIA
from llama_index.llms.openai import OpenAI

from app.document_processors import get_pdf_documents
from app.services.pinecone_service import initialize_pinecone, setup_pinecone_index, store_in_pinecone, \
    load_stored_pages

from app.utils import download_pdf_from_s3, embed_model
from fastapi import HTTPException
import os

# Configuration
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "multimodalindex")
VECTOR_DIMENSION = int(os.getenv("VECTOR_DIMENSION", "768"))  # Ensure consistency
SIMILARITY_TOP_K = int(os.getenv("SIMILARITY_TOP_K", "10"))


def initialize_rag(document_id: str):
    """
    Initializes the RAG setup by downloading the document, processing it, and storing embeddings in Pinecone.
    """
    # Initialize Pinecone
    pinecone = initialize_pinecone()
    logging.info("Initialized Pinecone.")
    # Download PDF from S3
    pdf_path = download_pdf_from_s3(document_id)
    logging.info(f"Downloaded PDF from S3: {pdf_path}")
    if not pdf_path:
        raise HTTPException(status_code=404, detail=f"Document {document_id} not found in S3.")

    # # Process PDF to extract documents
    # with open(pdf_path, "rb") as pdf_file:
    #     documents = get_pdf_documents(pdf_file)

    # Setup Pinecone index
    pinecone_index = setup_pinecone_index(PINECONE_INDEX_NAME, VECTOR_DIMENSION, pinecone)

    # Load stored pages to avoid duplication
    # stored_pages = load_stored_pages(document_id)
    #
    # # Prepare parsed_data in the required format
    # parsed_data = []
    # for doc in documents:
    #     parsed_data.append({
    #         "pdf_name": doc.metadata.get("source", "Unknown PDF"),
    #         "pages": [
    #             {
    #                 "page_num": doc.metadata.get("page_num", 0),
    #                 "text": doc.text
    #             }
    #         ]
    #     })
    #
    # # Store data in Pinecone
    # store_in_pinecone(pinecone_index, parsed_data, stored_pages, document_id)
    return pinecone


def query_chat(document_id: str, message: str, is_sum = False) -> str:
    """
    Handle a chat query by retrieving relevant documents from Pinecone and generating a response.
    """
    # Initialize RAG (ensure embeddings are stored)

    pinecone = initialize_rag(document_id)

    # Setup Pinecone index
    pinecone_index = pinecone.Index(PINECONE_INDEX_NAME)

    # Generate embedding for the query
    query_embedding = embed_model._get_text_embedding(message)
    if not query_embedding:
        raise HTTPException(status_code=500, detail="Failed to generate embedding for the query.")

    # Query Pinecone for similar vectors
    try:
        results = pinecone_index.query(
            vector=query_embedding,
            top_k=SIMILARITY_TOP_K,
            include_metadata=True
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error querying Pinecone: {str(e)}")

    # Aggregate relevant text from the results
    relevant_text = ""
    for match in results.matches:
        relevant_text += match.metadata.get("text", "") + "\n"

    # Generate response using the LLM
    try:
        if is_sum:
            query_embedding = embed_model.get_text_embedding(relevant_text)
            results = pinecone_index.query(
                vector=query_embedding,
                top_k=SIMILARITY_TOP_K,
                include_metadata=True,
            )
            relevant_text = ""
            for match in results.matches:
                relevant_text += match.metadata.get("text", "") + "\n"
        response = generate_response(relevant_text, message)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating response: {str(e)}")

    return response.text


def summarize_document(document_id: str, message: str = None) -> str:
    """
    Summarizes the document based on the user's message.
    """
    logging.info(f"Summarizing document {document_id}")
    message =f"Give me summary of this pdf {document_id}. It must be well structured, markdown code. It must be markdown formatted only. Dont give references, conclusion, acknowledgements etc. Give me very elaborate summary of this pdf {document_id}. Each sub headings must have a large description."
    # The summarization logic can be similar to chat
    return query_chat(document_id, message, is_sum=True)


def generate_response(relevant_text: str, user_message: str):
    """
    Generates a Markdown-formatted response using the LLM based on relevant text and user message.
    """
    # Implement the logic to interact with the LLM
    # Example using the NVIDIA LLM
    # llm_model_name = os.getenv("LLM_MODEL_NAME", "meta/llama-3.1-70b-instruct")
    llm = OpenAI()
    # llm = NVIDIA()
    logging.info(f"Generating response for user message: {relevant_text}")
    prompt = f"Based on the following information:\n{relevant_text}\n\nUser Question: {user_message}\n\nProvide a detailed Markdown-formatted answer."
    # print(prompt)
    response = llm.complete(prompt)
    return response
