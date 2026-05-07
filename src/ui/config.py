"""
config.py — App-wide constants and Streamlit page configuration.
"""

import streamlit as st

API_BASE = "https://ai-dj-api-343602206157.europe-west4.run.app"


def setup_page():
    st.set_page_config(
        page_title="AI-DJ | Hit Predictor",
        page_icon="🎧",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
