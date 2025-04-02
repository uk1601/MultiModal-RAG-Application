# components/navbar.py
import streamlit as st

def render_navbar(current_page):
    """Render the sidebar navigation and return the selected page."""
    st.sidebar.title("Navigation")
    sidebar_pages = ["Home", "Documents", "Profile"]

    selected_sidebar_page = st.sidebar.radio(
        "Go to",
        sidebar_pages,
        index=sidebar_pages.index(current_page) if current_page in sidebar_pages else 0
    )

    return selected_sidebar_page
