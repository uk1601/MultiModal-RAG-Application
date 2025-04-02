import streamlit as st
import requests
import os

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

API_BASE_URL = os.getenv("API_BASE_URL")

def authenticate_user(email, password):
    login_url = f"{API_BASE_URL}/auth/login"
    try:
        response = requests.post(login_url, data={"email": email, "password": password})
        if response.status_code == 200:
            data = response.json()
            st.session_state['access_token'] = data.get("access_token")
            st.session_state['refresh_token'] = data.get("refresh_token")
            return True, None
        else:
            error_detail = parse_error_response(response)
            return False, error_detail
    except Exception as e:
        return False, f"An error occurred: {e}"

def register_user(username, email, password):
    register_url = f"{API_BASE_URL}/auth/register"
    try:
        response = requests.post(register_url, data={"username": username, "email": email, "password": password})
        if response.status_code == 200 or response.status_code == 201:
            return True, None
        else:
            error_detail = parse_error_response(response)
            return False, error_detail
    except Exception as e:
        return False, f"An error occurred: {e}"

def parse_error_response(response):
    try:
        error_json = response.json()
        if isinstance(error_json, dict):
            detail = error_json.get("detail")
            if isinstance(detail, list):
                # Pydantic validation errors
                messages = [f"{err['loc'][-1]}: {err['msg']}" for err in detail]
                return " ; ".join(messages)
            elif isinstance(detail, dict):
                # Custom error response
                error_message = detail.get("error", "Error")
                error_details = detail.get("details", "")
                return f"{error_message}: {error_details}"
            else:
                return detail or "An error occurred."
        else:
            return "An error occurred."
    except ValueError:
        return "An error occurred."

def logout_user():
    st.session_state['is_authenticated'] = False
    st.session_state['access_token'] = None
    st.session_state['refresh_token'] = None
    st.session_state['user_email'] = None
    st.session_state['current_page'] = 'Home'

def get_headers():
    return {"Authorization": f"Bearer {st.session_state['access_token']}"}

def refresh_access_token():
    refresh_url = f"{API_BASE_URL}/auth/refresh"
    try:
        response = requests.post(refresh_url, data={"refresh_token": st.session_state['refresh_token']})
        if response.status_code == 200:
            data = response.json()
            st.session_state['access_token'] = data.get("access_token")
            return True
        else:
            st.error("Session expired. Please log in again.")
            logout_user()
            return False
    except Exception as e:
        st.error(f"An error occurred: {e}")
        logout_user()
        return False
