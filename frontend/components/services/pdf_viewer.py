# components/services/pdf_viewer.py
import requests
import streamlit as st
from streamlit_pdf_viewer import pdf_viewer

def fetch_pdf_content(url: str) -> bytes:
    if not url:
        return None  # Prevent fetching if URL is None
    try:
        response = requests.get(url)
        response.raise_for_status()
        return response.content
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching PDF content: {e}")
        return None

def display_pdf(content: bytes, width: int = 1200, height: int = 1000):
    if content:
        pdf_viewer(
            content,
            width=width,
            height=height,
            rendering="unwrap",
            render_text=True
        )
