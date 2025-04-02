# components/ui/buttons.py
import streamlit as st
from functools import partial

def view_button(label: str, key: str, callback, *args, **kwargs):
    st.button(label, key=key, use_container_width=True, on_click=partial(callback, *args, **kwargs))

def download_button(url: str, label: str = "📥 Download"):
    st.markdown(
        f'<a href="{url}" class="download-btn" target="_blank">{label}</a>',
        unsafe_allow_html=True
    )
