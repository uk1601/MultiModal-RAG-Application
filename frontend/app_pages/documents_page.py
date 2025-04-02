#app_pages/documents_page.py
import streamlit as st
import requests
import logging
from PIL import Image
from io import BytesIO
import os
from components.services.pdf_viewer import fetch_pdf_content, display_pdf

API_BASE_URL = os.getenv("API_BASE_URL")

@st.cache_data(ttl=3600)
def fetch_publications(api_base_url: str, access_token: str, page: int = 1, per_page: int = 100):
    endpoint = f"{api_base_url}/publications"
    headers = {"Authorization": f"Bearer {access_token}"}
    params = {"page": page, "per_page": per_page}

    try:
        response = requests.get(endpoint, headers=headers, params=params)
        response.raise_for_status()
        return response.json()
    except Exception as err:
        logging.error(f"Error fetching publications: {err}")
        raise

@st.cache_data(ttl=3600)
def fetch_image(image_url: str):
    try:
        response = requests.get(image_url)
        response.raise_for_status()
        return BytesIO(response.content)
    except Exception as e:
        logging.error(f"Error loading image: {e}")
        return None

def process_document_action(action, doc_content):
    if action == "Summarize":
        return "Document summary will be displayed here..."
    elif action == "Query":
        query = st.text_input("Enter your query about the document:", key="query_input")
        if query:
            return f"Query results for: {query}"
    elif action == "Generate Report":
        report_type = st.selectbox("Select report type:", ["Executive Summary", "Detailed Analysis", "Key Findings"], key="report_type_select")
        if st.button("Generate", key="generate_report_btn"):
            return f"{report_type} will be generated..."
    return None

def display_documents_grid(publications):
    if not publications:
        st.info("No publications available.")
        return

    cols_per_row = 4
    num_rows = (len(publications) + cols_per_row - 1) // cols_per_row

    for row in range(num_rows):
        cols = st.columns(cols_per_row)
        for col_index, col in enumerate(cols):
            pub_index = row * cols_per_row + col_index
            if pub_index < len(publications):
                pub = publications[pub_index]
                with col:
                    image_data = fetch_image(pub['IMAGE_URL'])
                    if image_data:
                        image = Image.open(image_data)
                        st.image(image, use_column_width=True)
                    else:
                        st.image("https://via.placeholder.com/150", use_column_width=True)

                    st.caption(pub['TITLE'])





                    if st.button(f"View", key=f"view_{pub['ID']}"):
                        print(f"Viewing document: {pub['TITLE']}")
                        print(f"Document ID: {pub['ID']}")
                        st.session_state['selected_pdf'] = pub['PDF_URL']
                        st.session_state['selected_title'] = pub['TITLE']
                        st.session_state['current_page'] = "PDF Viewer"
                        st.session_state["PDF_URL"] = pub['PDF_URL']
                        st.rerun()

                    # with btn_col2:
                    #     if st.button(f"Select", key=f"select_{pub['ID']}"):
                    #         st.session_state['selected_pdf'] = pub['PDF_URL']
                    #         st.session_state['selected_title'] = pub['TITLE']
                    #         st.session_state['current_page'] = "Document Actions"
                    #         st.rerun()

def documents_page():
    st.title("Documents Library")

    access_token = st.session_state.get('access_token')
    if not access_token:
        st.error("Authentication token is missing.")
        return

    if 'publications' not in st.session_state or st.button("Refresh Publications"):
        with st.spinner("Fetching publications..."):
            try:
                data = fetch_publications(API_BASE_URL, access_token)
                st.session_state['publications'] = data.get('publications', [])
                st.session_state['total_count'] = data.get('total_count', 0)
            except Exception as e:
                st.error(f"Failed to fetch publications: {e}")
                return

    st.success(f"Total Publications: {st.session_state.get('total_count', 0)}")

    search_query = st.text_input("🔍 Search Publications", placeholder="Enter title to search...", key="search_input")

    filtered_publications = [
        pub for pub in st.session_state.get('publications', [])
        if not search_query or search_query.lower() in pub.get('TITLE', '').lower()
    ]

    display_documents_grid(filtered_publications)

def pdf_viewer_page():
    print("PDF Viewer Page loaded...")
    header_container = st.container()
    with header_container:
        st.header(st.session_state.get('selected_title', 'PDF Viewer'))

    pdf_url = st.session_state.get('selected_pdf')
    if not pdf_url:
        st.error("No PDF selected to view.")
        return

    with st.spinner("Loading PDF..."):
        try:
            pdf_content = fetch_pdf_content(pdf_url)
            if pdf_content:
                display_pdf(pdf_content, width=800, height=1000)
            else:
                st.error("Failed to load PDF content.")
        except Exception as e:
            st.error(f"An error occurred while fetching the PDF: {e}")
