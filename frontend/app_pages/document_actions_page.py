# document_actions_page.py
import streamlit as st
import logging
from components.services.pdf_viewer import fetch_pdf_content, display_pdf
import requests
import os

def fetch_summary():
    endpoint = f"{os.getenv("API_BASE_URL")}/summarize"
    headers = {"Authorization": f"Bearer {st.session_state.get('access_token')}", "Content-Type": "application/json", "Accept": "application/json"}
    print("Title selected for summaize: " + str(st.session_state.get('selected_title')))
    try:
        # https://cfapublications.s3.amazonaws.com/assignment3/pdfs/Beyond Active and Passive Investing_ The Customization of Finance.pdf
        # This is an example of pdf url
        # i need the name stracted like assignment3/pdfs/Beyond Active and Passive Investing_ The Customization of Finance.pdf

        pdf_name =  str(st.session_state.get('PDF_URL'))
        pdf_name = pdf_name.split("/")[-1]
        pdf_name = "assignment3/pdfs/" + pdf_name
        print("PDF Name: " + pdf_name)
        response = requests.post(endpoint, headers=headers, json={"document_name": pdf_name})
        response.raise_for_status()
        print("Response: " + str(response.json()))
        return response.json()
    except Exception as err:
        logging.error(f"Error fetching publications: {err}")
        raise


def process_document_action(action, doc_content):
    if action == "Summarize":
        with st.spinner("Generating summary..."):
            # Implement actual summarization logic here
            return "Document summary will be displayed here..."

    elif action == "Query":
        query = st.text_input("Enter your query about the document:")
        if query:
            with st.spinner("Processing query..."):
                # Implement actual query logic here
                return f"Query results for: {query}"

    elif action == "Generate Report":
        report_type = st.selectbox(
            "Select report type:",
            ["Executive Summary", "Detailed Analysis", "Key Findings"]
        )
        if st.button("Generate"):
            with st.spinner(f"Generating {report_type}..."):
                # Implement actual report generation logic here
                return f"{report_type} will be generated..."
    return None

def display_document_actions(doc_title, doc_content):
    """Handle actions for a selected document"""
    st.subheader(f"Selected Document: {doc_title}")

    action = st.selectbox(
        "Choose an action:",
        ["Select an action...", "Summarize", "View", "Query", "Generate Report"]
    )

    if action != "Select an action...":
        if action == "View":
            st.session_state['current_page'] = "PDF Viewer"
            st.rerun()
        else:
            result = process_document_action(action, doc_content)
            if result:
                st.markdown("### Results")
                st.write(result)

def document_actions_page():
    st.title("Document Actions")

    # Check if a document is selected
    selected_title = st.session_state.get('selected_title')
    selected_pdf = st.session_state.get('selected_pdf')

    # Conditional rendering: Only display dropdown if a document is selected
    if not selected_title:
        st.warning("Please select a document from the Documents page.")

        # Button to redirect to Documents page if no document is selected
        if st.button("Go to Documents Page", key="go_to_docs"):
            st.session_state['current_page'] = "Documents"
            st.rerun()
        return  # Exit the function early if no document is selected

    # Display actions for the selected document
    display_document_actions(selected_title, selected_pdf)

    # Option to clear the selection
    if st.button("Clear Selection", key="clear_selection"):
        st.session_state.pop('selected_title', None)
        st.session_state.pop('selected_pdf', None)
        st.rerun()

def fetch_notes_list():
    endpoint = f"{os.getenv('API_BASE_URL')}/list-files"
    headers = {
        "Authorization": f"Bearer {st.session_state.get('access_token')}",
        "Content-Type": "application/json",
        "Accept": "application/json"
    }
    try:
        response = requests.get(endpoint, headers=headers)
        response.raise_for_status()
        return response.json().get("files", [])
    except Exception as err:
        logging.error(f"Error fetching notes list: {err}")
        raise

def download_note(filename):
    endpoint = f"{os.getenv('API_BASE_URL')}/download-file"
    headers = {
        "Authorization": f"Bearer {st.session_state.get('access_token')}",
        "Content-Type": "application/json",
        "Accept": "application/json"
    }
    try:
        response = requests.post(
            endpoint,
            headers=headers,
            json={"filename": filename},
            stream=True
        )
        response.raise_for_status()
        return response.content
    except Exception as err:
        logging.error(f"Error downloading note: {err}")
        raise

def notes_page():
    st.title("Notes")

    # Create two columns
    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Available Notes")
        try:
            notes_list = fetch_notes_list()

            # Initialize selected note in session state if not exists
            if 'selected_note' not in st.session_state:
                st.session_state.selected_note = None

            # Create a container for the list
            notes_container = st.container()
            with notes_container:
                for note in notes_list:
                    # Create a unique key for each button
                    if st.button(
                        note,
                        key=f"note_{note}",
                        help="Click to view this note",
                        use_container_width=True,
                        type="secondary" if st.session_state.selected_note != note else "primary"
                    ):
                        st.session_state.selected_note = note
                        st.rerun()

        except Exception as e:
            st.error(f"Error loading notes list: {str(e)}")

    with col2:
        if st.session_state.selected_note:
            st.subheader(f"Viewing: {st.session_state.selected_note}")

            try:
                # Download and display PDF
                pdf_content = download_note(st.session_state.selected_note)
                display_pdf(pdf_content)

            except Exception as e:
                st.error(f"Error loading PDF: {str(e)}")
        else:
            st.info("Select a note from the list to view its content")

    # Add a refresh button
    if st.button("Refresh Notes List", key="refresh_notes"):
        st.session_state.selected_note = None
        st.rerun()

def generate_report(pdf_name):
    endpoint = f"{os.getenv('API_BASE_URL')}/generate_report"
    headers = {
        "Authorization": f"Bearer {st.session_state.get('access_token')}",
        "Content-Type": "application/json",
        "Accept": "application/json"
    }
    try:
        response = requests.post(
            endpoint,
            headers=headers,
            json={"pdf_name": pdf_name},
            stream=True
        )
        response.raise_for_status()
        return response.content
    except Exception as err:
        logging.error(f"Error generating report: {err}")
        raise

def report_page():
    st.title("Report")

    # Check if a document is selected
    selected_pdf_url = st.session_state.get('PDF_URL')

    if not selected_pdf_url:
        st.warning("Please select a document from the Documents page first.")
        if st.button("Go to Documents Page"):
            st.session_state['current_page'] = "Documents"
            st.rerun()
        return

    # Extract PDF name from URL
    pdf_name = selected_pdf_url.split("/")[-1]
    pdf_name = "assignment3/pdfs/" + pdf_name

    # Create two columns
    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Report Information")
        st.write(f"Selected Document: {pdf_name.split('/')[-1]}")

    with col2:
        st.subheader("Report Preview")
        with st.spinner("Generating report..."):
            try:
                # Generate and display report automatically
                report_content = generate_report(pdf_name)
                display_pdf(report_content)
            except Exception as e:
                st.error(f"Error generating or displaying report: {str(e)}")

def summerize_page():
    st.title("Summary")
    response = fetch_summary()
    md = response.get("markdown")
    st.markdown(md)

def fetch_chat_response(message):
    endpoint = f"{os.getenv('API_BASE_URL')}/chat"
    headers = {
        "Authorization": f"Bearer {st.session_state.get('access_token')}",
        "Content-Type": "application/json",
        "Accept": "application/json"
    }
    try:
        pdf_name = str(st.session_state.get('PDF_URL')).split("/")[-1]
        pdf_name = "assignment3/pdfs/" + pdf_name

        response = requests.post(
            endpoint,
            headers=headers,
            json={
                "document_id": pdf_name,
                "message": message
            }
        )
        response.raise_for_status()
        return response.json()
    except Exception as err:
        logging.error(f"Error in chat response: {err}")
        raise

def query_page():
    st.title("Chat with Document")

    # Initialize chat history in session state if it doesn't exist
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    # Display chat history
    chat_container = st.container()
    with chat_container:
        for message in st.session_state.chat_history:
            if message["role"] == "user":
                st.warning(f'You: {message["content"]}', icon="🧑")
            else:
                st.info(f'🤖 Assistant: {message["content"]}')

    # Chat input
    with st.container():
        # Create a form for the chat input
        with st.form(key='chat_form', clear_on_submit=True):
            user_input = st.text_area("Type your message:", key='user_input', height=100)
            submit_button = st.form_submit_button("Send")

            if submit_button and user_input:
                # Add user message to chat history
                st.session_state.chat_history.append({
                    "role": "user",
                    "content": user_input
                })

                # Get bot response
                with st.spinner("Thinking..."):
                    try:
                        response = fetch_chat_response(user_input)
                        bot_message = response.get("content", "Sorry, I couldn't process your request.")

                        # Add bot response to chat history
                        st.session_state.chat_history.append({
                            "role": "assistant",
                            "content": bot_message
                        })
                    except Exception as e:
                        st.error(f"An error occurred: {str(e)}")

                # Rerun to update the chat display
                st.rerun()

    # Add a button to clear chat history
    if st.button("Clear Chat History"):
        st.session_state.chat_history = []
        st.rerun()
