import streamlit as st

class SessionStore:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(SessionStore, cls).__new__(cls, *args, **kwargs)
            cls._instance.defaults = {
                'is_authenticated': False,
                'access_token': None,
                'refresh_token': None,
                'display_login': True,
                'display_register': False,
                'user_email': None,
                'current_page': 'Home',
                'pppages': ['Home', 'Summary', 'Querying']
            }
            cls._instance.initialize_session()  # Moved here after defaults are set
        return cls._instance

    def __init__(self):
        self.defaults = {
            'is_authenticated': False,
            'access_token': None,
            'refresh_token': None,
            'display_login': True,
            'display_register': False,
            'user_email': None,
            'current_page': 'Home',
            'selected_file': "Select a PDF document",
            'model_type': "Open Source Extractor",
            'operation': "Summarize",
            'gpt_model': "gpt-4o-mini",
            'query_text': None
        }

    def initialize_session(self):
        """ Initialize session state with default values if not already set """
        for key, default in self.defaults.items():
            if key not in st.session_state:
                st.session_state[key] = default

    def clear_session(self):
        """ Clear the session state to defaults """
        for key in self.defaults.keys():
            st.session_state[key] = self.defaults[key]

    def set_value(self, key, value):
        """ Set a specific session value """
        st.session_state[key] = value

    def get_value(self, key):
        """ Get a specific session value """
        return st.session_state.get(key)

    def is_authenticated(self):
        """ Check if the user is authenticated """
        if 'is_authenticated' not in st.session_state:
            st.session_state['is_authenticated'] = False
        return st.session_state['is_authenticated']

    def get_current_page(self):
        """ Get the current page """
        return st.session_state['current_page']

    def set_current_page(self, page):
        """ Set the current page """
        st.session_state['current_page'] = page

    def get_user_email(self):
        """ Get the logged-in user's email """
        return st.session_state['user_email']

# Create a globally accessible instance
session_store = SessionStore()
