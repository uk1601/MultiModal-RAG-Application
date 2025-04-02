# app.py
import streamlit as st
from services.session_store import session_store
from services.authentication import auth
import logging
import os
from dotenv import load_dotenv
from app_pages.home_page import home_page
from app_pages.documents_page import display_documents_grid, documents_page, pdf_viewer_page
from app_pages.document_actions_page import display_document_actions, document_actions_page, summerize_page, query_page, report_page, notes_page

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler()
    ]
)

# Backend API base URL
API_BASE_URL = os.getenv("API_BASE_URL")

# Session state defaults
session_defaults = {
    'display_login': True,
    'display_register': False,
    'current_page': 'Documents',
    'selected_pdf': None,
    'selected_title': None,
    'selected_document_id': None  # Added to track selected document

}

def initialize_session_state():
    for key, default in session_defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default

def clear_session_storage():
    logging.info("Clearing session storage")
    # Clear all session state
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    # Reinitialize with defaults
    initialize_session_state()

def login_page():
    # Added UI for the application title and features
    st.markdown(
        """
        <style>
        .container {
            text-align: center;
            font-family: Arial, sans-serif;
        }
        .title {
            font-size: 2.5em;
            color: #2E86C1;
            margin-bottom: 10px;
        }
        .features {
            background-color: #f5f5f5;
            padding: 15px;
            border-radius: 10px;
            margin-bottom: 20px;
        }
        .features h3 {
            color: #1A5276;
        }
        .features ul {
            text-align: center;
            list-style-type: none;
            padding-left: 0;
        }
        .features ul li {
            font-size: 1.1em;
            color: #333;
            margin-bottom: 10px;
            padding-left: 1.5em;
            text-indent: -1.5em;
        }
        .features ul li:before {
            content: '✓';
            margin-right: 10px;
            color: #1ABC9C;
        }
        </style>
        <div class="container">
            <div class="title">Document Query Platform</div>
            <div class="features">
                <h3>Features available:</h3>
                <ul>
                    <li>Secure Document Summarization and Querying</li>
                    <li>Access to Multiple Document Extractors (Open Source & Enterprise)</li>
                    <li>Seamless Integration with GPT Models for Text Extraction</li>
                    <li>View and Interact with Pre-Processed Documents</li>
                </ul>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    if session_store.get_value('display_login'):
        display_login_form()
    elif session_store.get_value('display_register'):
        display_register_form()

def display_login_form():
    st.subheader("Login")

    with st.form("login_form", clear_on_submit=True):
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")
        submit = st.form_submit_button("Login")

        if submit:
            if not email or not password:
                st.error("Please enter both email and password.")
                return

            try:
                auth.login(email, password)
                st.success("Logged in successfully!")
                st.session_state['current_page'] = 'Documents'  # Set to Home after login
                st.rerun()
            except Exception as e:
                st.error(f"Login failed: {str(e)}")

    if st.button("Register"):
        show_register_form()

def display_register_form():
    st.subheader("Register")
    with st.form("register_form", clear_on_submit=True):
        username = st.text_input("Username")
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")
        submit = st.form_submit_button("Register")

        if submit:
            if not username or not email or not password:
                st.error("Please fill in all fields.")
                return

            try:
                auth.register(username, email, password)
                st.success("Registered successfully! Please log in.")
                show_login_form()
            except Exception as e:
                st.error(f"Registration failed: {str(e)}")

    if st.button("Back to Login"):
        show_login_form()

def show_register_form():
    session_store.set_value('display_login', False)
    session_store.set_value('display_register', True)
    st.rerun()

def show_login_form():
    session_store.set_value('display_login', True)
    session_store.set_value('display_register', False)
    st.rerun()

def main():
    initialize_session_state()

    if not session_store.is_authenticated():
        login_page()
        return

    # documents_page()
    # Sidebar navigation
    pages = {
        "Documents": documents_page,
        "Document Actions": document_actions_page
    }

    current_page = st.session_state['current_page']
    print("Printing current page" + current_page)
    # Handle PDF Viewer as a special case
    if current_page == "PDF Viewer":
        pdf_viewer_page()
    elif current_page == "Documents":
        documents_page()
    elif current_page == "Home":
        documents_page()
    elif current_page == "Summerize":
        summerize_page()
    elif current_page == "Query":
        query_page()
    elif current_page == "Report":
        report_page()
    elif current_page == "Notes":
        notes_page()



        # pages[selected_page]()

    # Logout button in sidebar
    with st.sidebar:
        if st.button("Logout", use_container_width=True, type="secondary"):
            try:
                logging.info("Logging out user")
                auth.logout()
                clear_session_storage()
                st.success("Logged out successfully.")
                st.experimental_rerun()
            except Exception as e:
                logging.error(f"Error during logout: {e}")
                st.sidebar.error("Error during logout. Please try again.")

        print("Printing current page: " + current_page)
        if st.session_state["current_page"] != "Documents" and st.button("← Back to Documents", key="back_to_documents", use_container_width=True, type="secondary"):
            st.session_state['current_page'] = "Documents"
            st.rerun()

        st.markdown("---")
        if st.session_state['current_page'] != "Documents":
            if st.button("Summerize", key="summerize", use_container_width=True, type="primary"):
                st.session_state['current_page'] = "Summerize"
                st.rerun()
            if st.button("Query", key="query", use_container_width=True, type="primary"):
                st.session_state['current_page'] = "Query"
                st.rerun()
            if st.button("Notes", key="notes", use_container_width=True, type="primary"):
                st.session_state['current_page'] = "Notes"
                st.rerun()
            if st.button("Report Generation", key="Report", use_container_width=True, type="primary"):
                st.session_state['current_page'] = "Report"
                st.rerun()




if __name__ == "__main__":
    st.set_page_config(page_title="Document Query Platform", page_icon="📄", layout="wide")
    main()
