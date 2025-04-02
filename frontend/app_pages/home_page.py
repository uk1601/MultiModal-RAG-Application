# home_page.py
import streamlit as st

def home_page():
    st.title("Welcome to Document Query Platform")
    st.write("Access and analyze your documents securely.")

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Documents Available", "1,000+")
    with col2:
        st.metric("Active Users", "500+")

    st.markdown("""
    ### Key Features:
    - 📄 Secure Document Management
    - 🔍 Advanced Search Capabilities
    - 📊 Document Analytics
    - 🤖 AI-Powered Insights
    """)
