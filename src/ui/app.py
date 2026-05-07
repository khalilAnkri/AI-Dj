"""
INFO9023 - Machine Learning Systems Design - Spotify Hit Predictor
Streamlit Frontend — AI-DJ
"""

import streamlit as st

from config import setup_page
from styles import inject_css
from api import call_predict, call_predict_manual, call_history
from ui import render_header, render_footer, render_result, render_manual_inputs, render_tabs, top_nav
from eda import render_eda_page

setup_page()
inject_css()

page = top_nav()  # returns "home" or "eda"

# ─────────────────────────────────────────────────────────────────────────────
# EDA Page
# ─────────────────────────────────────────────────────────────────────────────
if page == "eda":
    render_eda_page()

# ─────────────────────────────────────────────────────────────────────────────
# Home Page
# ─────────────────────────────────────────────────────────────────────────────
else:
    render_header()

    _, nav_col, _ = st.columns([1.5, 1, 1.5])

    with nav_col:
        active_tab = render_tabs()

    if active_tab == "ANALYZE":
        st.markdown('<div style="color:#4a7060;font-family:\'JetBrains Mono\',monospace;'
                    'font-size:9px;letter-spacing:3px;text-transform:uppercase;'
                    'margin-bottom:14px;">Paste a Spotify track URL to analyze</div>',
                    unsafe_allow_html=True)
        col_input, col_btn = st.columns([5, 1])
        with col_input:
            query = st.text_input("", placeholder="https://open.spotify.com/track/...",
                                  key="url_input", label_visibility="collapsed")
        with col_btn:
            analyze = st.button("▶ ANALYZE", key="analyze_btn")
        if analyze and query:
            with st.spinner("Analyzing track..."):
                result = call_predict(query)
            if result:
                st.session_state["last_result"] = result
        if "last_result" in st.session_state:
            render_result(st.session_state["last_result"])

    elif active_tab == "MANUAL INPUT":
        features = render_manual_inputs()
        if features:
            with st.spinner("Running prediction..."):
                result = call_predict_manual(features)
            if result:
                st.session_state["last_manual_result"] = result
        if "last_manual_result" in st.session_state:
            render_result(st.session_state["last_manual_result"])

    elif active_tab == "HISTORY":
        st.markdown('<div style="color:#4a7060;font-family:\'JetBrains Mono\',monospace;'
                    'font-size:9px;letter-spacing:3px;text-transform:uppercase;'
                    'margin-bottom:16px;">Past Predictions</div>', unsafe_allow_html=True)
        if st.button("↻ REFRESH", key="refresh_btn"):
            st.session_state["history"] = call_history()
        if "history" not in st.session_state:
            st.session_state["history"] = call_history()
        history = st.session_state.get("history", [])
        if not history:
            st.markdown('<div style="color:#4a7060;text-align:center;padding:60px 0;'
                        'font-family:\'JetBrains Mono\',monospace;font-size:12px;'
                        'letter-spacing:2px;text-transform:uppercase;">NO PREDICTIONS YET</div>',
                        unsafe_allow_html=True)
        else:
            for item in reversed(history):
                prediction = item.get("prediction", "")
                color = "#2ecc71" if prediction == "Hit" else "#e8365d"
                track = item.get("track_name", item.get("source", "Manual Input"))
                artist = item.get("artist", "")
                conf = item.get("confidence", "")
                ts = item.get("predicted_at", "")[:19].replace("T", " ")
                st.markdown(f"""<div style="background:#0e1e1a;border:1px solid #1e3530;
                    border-radius:8px;padding:14px 18px;margin-bottom:8px;
                    display:flex;justify-content:space-between;align-items:center;">
                    <div><span style="font-weight:600;font-size:14px;color:#d4ede5;
                         font-family:'DM Sans',sans-serif;">{track}</span>
                    <span style="color:#4a7060;font-size:13px;margin-left:8px;">{artist}</span></div>
                    <div style="display:flex;gap:16px;align-items:center;">
                    <span style="font-family:'JetBrains Mono',monospace;font-size:12px;color:#4a7060;">{ts}</span>
                    <span style="color:{color};font-family:'JetBrains Mono',monospace;font-size:12px;letter-spacing:2px;">{prediction}</span>
                    <span style="font-family:'JetBrains Mono',monospace;font-size:13px;color:{color};">{conf}</span>
                    </div></div>""", unsafe_allow_html=True)

render_footer()