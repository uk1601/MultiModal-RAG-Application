# components/services/s3_service.py
import os
import requests
import streamlit as st
from typing import List, Dict
from dotenv import load_dotenv

load_dotenv()

FASTAPI_BASE_URL = os.getenv("FASTAPI_BASE_URL")
PDF_PREFIX = os.getenv("PDF_PREFIX")
PDF_EXTENSION = os.getenv("PDF_EXTENSION")

def fetch_pdfs() -> List[Dict]:
    try:
        params = {"prefix": PDF_PREFIX, "file_extension": PDF_EXTENSION}
        response = requests.get(f"{FASTAPI_BASE_URL}/list", params=params)
        response.raise_for_status()
        return response.json().get("objects", [])
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching PDFs: {e}")
        return []

def get_presigned_url(key: str, download: bool = False) -> str:
    try:
        params = {'key': key, 'download': download}
        response = requests.get(f"{FASTAPI_BASE_URL}/url", params=params)
        response.raise_for_status()
        return response.json().get("url", "#")
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching URL for {key}: {e}")
        return "#"
