import streamlit as st

# from services.utils import authenticate_user
from services.utils import authenticate_user, register_user, logout_user, refresh_access_token
from services.session_store import session_store  # Import the globally accessible instance

class Authentication:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(Authentication, cls).__new__(cls, *args, **kwargs)
        return cls._instance

    def __init__(self):
        self.session_store = session_store

    def login(self, email, password):
        """ Process login and set session state upon success """
        if email and password:
            with st.spinner("Authenticating..."):
                success, error_message = authenticate_user(email, password)
            if success:
                st.success("Login successful!")
                self.session_store.set_value('is_authenticated', True)
                self.session_store.set_value('user_email', email)
                self.session_store.set_value('display_login', False)
                st.rerun()  # Re-run the app to reflect login
            else:
                st.error(error_message)
        else:
            st.warning("Please enter both email and password.")

    def register(self, username, email, password):
        """ Process user registration """
        if username and email and password:
            with st.spinner("Registering..."):
                success, error_message = register_user(username, email, password)
            if success:
                st.success("Registration successful! Please log in.")
                self.session_store.set_value('display_register', False)
                self.session_store.set_value('display_login', True)
                st.rerun()  # Re-run the app to show login page
            else:
                st.error(error_message)
        else:
            st.warning("Please fill all fields.")

    def logout(self):
        """ Logout the user and clear session """
        logout_user()
        st.success("Logged out successfully.")
        self.session_store.clear_session()
        st.rerun()

    def check_access(self):
        """ Ensure the user is authenticated to view pppages """
        if not self.session_store.is_authenticated():
            st.warning("You need to log in to access this page.")
            st.stop()

    def refresh_token(self):
        """ Refresh the access token if needed """
        access_token = self.session_store.get_value('access_token')
        refresh_token = self.session_store.get_value('refresh_token')
        if refresh_token:
            new_access_token = refresh_access_token(refresh_token)
            self.session_store.set_value('access_token', new_access_token)

# Create a globally accessible instance
auth = Authentication()
