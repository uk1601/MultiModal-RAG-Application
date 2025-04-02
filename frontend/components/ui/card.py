# components/ui/card.py
import streamlit as st

def pdf_card(title: str, description: str, height: int = 320):
    st.markdown(f"""
        <div class="pdf-card" style="height: {height}px;">
            <h3>{title}</h3>
            <p>{description}</p>
        </div>
    """, unsafe_allow_html=True)
