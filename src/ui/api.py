"""
api.py — HTTP client functions that talk to the FastAPI backend.
"""

import requests
import streamlit as st
from config import API_BASE


def call_predict(query: str) -> dict:
    try:
        r = requests.post(f"{API_BASE}/predict", json={"query": query}, timeout=30)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.ConnectionError:
        st.error("❌ Cannot connect to the API. Make sure the FastAPI server is running on port 8000.")
        return {}
    except Exception as e:
        st.error(f"❌ API error: {e}")
        return {}


def call_predict_manual(features: dict) -> dict:
    try:
        r = requests.post(f"{API_BASE}/predict_manual", json=features, timeout=30)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.ConnectionError:
        st.error("❌ Cannot connect to the API.")
        return {}
    except Exception as e:
        st.error(f"❌ API error: {e}")
        return {}


def call_history() -> list:
    try:
        r = requests.get(f"{API_BASE}/past_predictions", timeout=30)
        return r.json() if r.ok else []
    except Exception:
        return []