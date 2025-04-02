# pages/pdf_gallery.py
import streamlit as st
from components.ui.card import pdf_card
from components.ui.buttons import view_button, download_button
from components.services.s3_service import fetch_pdfs, get_presigned_url
from components.services.pdf_viewer import fetch_pdf_content, display_pdf
from pathlib import Path

def pdf_gallery_page():
    st.title("📚 PDF Gallery")
    
    # Initialize session state
    if 'show_pdf_viewer' not in st.session_state:
        st.session_state.show_pdf_viewer = False
        st.session_state.current_pdf_url = None
        st.session_state.current_pdf_name = None

    def set_pdf_viewer(url: str, name: str):
        st.session_state.show_pdf_viewer = True
        st.session_state.current_pdf_url = url
        st.session_state.current_pdf_name = name

    def unset_pdf_viewer():
        st.session_state.show_pdf_viewer = False
        st.session_state.current_pdf_url = None
        st.session_state.current_pdf_name = None

    # Check if viewing a PDF
    if st.session_state.show_pdf_viewer and st.session_state.current_pdf_url:
        col1, col2 = st.columns([0.2, 0.8])
        
        with col1:
            view_button("← Back to Gallery", key="back_button", callback=unset_pdf_viewer)
            if st.session_state.current_pdf_name:
                st.markdown(f"""
                    <div class="pdf-card">
                        <h4>{st.session_state.current_pdf_name}</h4>
                    </div>
                """, unsafe_allow_html=True)
        
        with col2:
            pdf_content = fetch_pdf_content(st.session_state.current_pdf_url)
            display_pdf(pdf_content)
        return  # Exit to prevent rendering the gallery below

    # If not viewing, show the search bar and gallery
    search_query = st.text_input("🔍 Search PDFs by name:")

    # PDF Gallery Grid
    with st.spinner("Loading PDF Gallery..."):
        pdfs = fetch_pdfs()

    if not pdfs:
        st.info("📂 No PDFs found in the testing folder.")
        return

    # Filter PDFs based on search query
    if search_query:
        filtered_pdfs = [
            pdf for pdf in pdfs
            if search_query.lower() in Path(pdf['key']).name.lower()
        ]
        if not filtered_pdfs:
            st.warning(f"No PDFs found matching '{search_query}'.")
    else:
        filtered_pdfs = pdfs

    if not filtered_pdfs:
        st.info("📂 No PDFs to display.")
        return

    # Display PDF grid with enhanced styling
    num_columns = 3
    columns = st.columns(num_columns)
    
    for idx, pdf in enumerate(filtered_pdfs):
        with columns[idx % num_columns]:
            pdf_name = Path(pdf['key']).name
            view_url = get_presigned_url(pdf['key'], download=False)
            download_url = get_presigned_url(pdf['key'], download=True)

            pdf_card(
                title=pdf_name,
                description=f"📊 Size: {pdf['size'] / 1024:.2f} KB\n🕒 Modified: {pdf['last_modified']}"
            )

            col1, col2 = st.columns(2)
            with col1:
                view_button("🔍 View", key=f"view_{pdf['key']}", callback=set_pdf_viewer, url=view_url, name=pdf_name)
            
            with col2:
                download_button(download_url)
