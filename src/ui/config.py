"""
config.py — App-wide constants and Streamlit page configuration.
"""

import streamlit as st

API_BASE = "http://localhost:8000"


def setup_page():
    st.set_page_config(
        page_title="AI-DJ | Hit Predictor",
        page_icon="🎧",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
