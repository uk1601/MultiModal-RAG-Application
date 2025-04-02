# app/services/pinecone_service.py
import json
import logging

import dotenv
import pinecone
import os
from typing import List, Dict
import uuid
from fastapi import HTTPException
from pinecone import Pinecone, ServerlessSpec

from app.utils import embed_model
from app.utils import load_chat_history, save_chat_history

dotenv.load_dotenv()
print(os.getenv("PINECONE_API_KEY"))
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
# Initialize Pinecone
def initialize_pinecone():
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    return pc


def setup_pinecone_index(index_name: str, embedding_dimension: int, pinecone) -> pinecone.Index:
    """
    Checks if the specified index_name exists in Pinecone.
    If it exists, connects to the index; if not, creates the index with specified parameters.
    """

    indexes = pinecone.list_indexes()

    # Check if the index_name exists in the list

    try:
        pinecone_index = pinecone.Index(index_name)
        # print(f"[INFO] Index '{index_name}' does not exist. Creating it.")
        # pinecone_index = pinecone.create_index(
        #     name='multimodalindex',
        #     dimension=1536,
        #     metric="cosine",
        #     spec=ServerlessSpec(
        #         cloud="aws",
        #         region="us-east-1"
        #     )
        # )
        print(f"[INFO] Index '{index_name}' created successfully.")
        return pinecone_index
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error setting up Pinecone index: {str(e)}")


def store_in_pinecone(index: pinecone.Index, parsed_data: List[Dict], stored_pages: set, document_id: str):
    """
    Stores parsed PDF data in Pinecone, checking if each page has already been stored based on pdf_name and page_num.
    """
    print("[INFO] Storing parsed data in Pinecone.")

    try:
        for page_data in parsed_data:
            pdf_name = page_data.get("pdf_name", "Unknown PDF")

            for page in page_data.get("pages", []):
                page_num = page.get("page_num")
                page_text = page.get("text", "")
                metadata = {
                    "page_num": page_num,
                    "pdf_name": pdf_name,
                    "text": page_text,
                    "document_id": document_id
                }

                # Check if this page is already tracked locally
                if (pdf_name, page_num) in stored_pages:
                    print(f"[INFO] Page {page_num} of '{pdf_name}' already exists in Pinecone. Skipping storage.")
                    continue

                # Embed and store the new page if not found in local cache
                embedding = embed_model.get_text_embedding(page_text)  # Replace with actual embedding method
                if not embedding:
                    print(f"[WARN] Failed to generate embedding for page {page_num} of '{pdf_name}'. Skipping.")
                    continue

                print(f"[INFO] Storing page {page_num} of '{pdf_name}' in Pinecone.")

                index.upsert([
                    {
                        "id": f"{pdf_name}_page_{page_num}_{uuid.uuid4()}",
                        "values": embedding,
                        "metadata": metadata
                    }
                ])

                # Add the page to the local cache and save to file
                stored_pages.add((pdf_name, page_num))
                save_stored_pages(document_id, stored_pages)
                print(f"[INFO] Successfully stored page {page_num} of '{pdf_name}' in Pinecone.")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error storing data in Pinecone: {str(e)}")


def load_stored_pages(document_id: str) -> set:
    """
    Loads the set of stored pages from a JSON file.
    """
    cwd = os.getcwd()
    directory = cwd + "/index_histories/"

    history_path = os.path.join(directory, f"{document_id}_stored_pages.json")
    #    if nested directory is not present, create it
    if not os.path.exists(directory):
        os.makedirs(directory)
    # if json file does not exist, create it
    if not os.path.exists(history_path):
        with open(history_path, "w") as f:
            json.dump([], f)

    logging.info(f"[INFO] Loading stored pages from {history_path}.")
    if not os.path.exists(history_path):
        return set()
    with open(history_path, "r") as f:
        try:
            stored_pages = set(tuple(page) for page in json.load(f))
            return stored_pages
        except json.JSONDecodeError:
            print(f"[WARN] Invalid JSON format in {history_path}. Returning empty set.")
            return set()


def save_stored_pages(document_id: str, stored_pages: set):
    """
    Saves the set of stored pages to a JSON file for persistence.
    """
    cwd = os.getcwd()
    directory = cwd + "/index_histories/"
    dir = directory + "assignment3/pdfs/"
    if not os.path.exists(dir):
        os.makedirs(dir)
    history_path = os.path.join(directory, f"{document_id}_stored_pages.json")
    #    if nested directory is not present, create it
    if not os.path.exists(directory):
        os.makedirs(directory)
    # if json file does not exist, create it
    if not os.path.exists(history_path):
        with open(history_path, "w") as f:
            json.dump([], f)


    with open(history_path, "w") as f:
            json.dump(list(stored_pages), f, indent=4)