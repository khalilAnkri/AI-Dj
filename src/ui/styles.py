"""
styles.py — Global CSS for the dark-teal AI-DJ theme.
Pixel-perfect match to the reference HTML design spec.
"""

import streamlit as st


def inject_css():
    st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=DM+Sans:wght@300;400;500;600&family=JetBrains+Mono:wght@400;600;700&display=swap');

:root {
    --bg:           #0b1a17;
    --bg2:          #0f211d;
    --card:         #0e1e1a;
    --card2:        #112019;
    --border:       #1e3530;
    --green:        #2ecc71;
    --green-dim:    #1a9e52;
    --green-glow:   rgba(46,204,113,0.25);
    --red:          #e8365d;
    --red-glow:     rgba(232,54,93,0.25);
    --muted:        #4a7060;
    --text:         #d4ede5;
    --label:        #6aaa8a;
    --amber:        #e8a836;
    --track-bg:     #1a2e28;
}

@keyframes twinkle-a { 0%,100%{opacity:0} 50%{opacity:0.85} }
@keyframes twinkle-b { 0%,100%{opacity:0} 50%{opacity:0.60} }

[data-testid="stAppViewContainer"]::before {
    content:'';position:fixed;inset:0;z-index:0;pointer-events:none;
    background-image:
        radial-gradient(circle 1px at  4%  6%, #7fffd4 0%, transparent 100%),
        radial-gradient(circle 1px at 11% 14%, #7fffd4 0%, transparent 100%),
        radial-gradient(circle 1px at 19%  3%, #ffffff 0%, transparent 100%),
        radial-gradient(circle 1px at 27% 22%, #7fffd4 0%, transparent 100%),
        radial-gradient(circle 1px at 35%  9%, #7fffd4 0%, transparent 100%),
        radial-gradient(circle 1px at 43% 18%, #ffffff 0%, transparent 100%),
        radial-gradient(circle 1px at 52%  5%, #7fffd4 0%, transparent 100%),
        radial-gradient(circle 1px at 61% 13%, #7fffd4 0%, transparent 100%),
        radial-gradient(circle 1px at 70%  2%, #ffffff 0%, transparent 100%),
        radial-gradient(circle 1px at 78% 20%, #7fffd4 0%, transparent 100%),
        radial-gradient(circle 1px at 86%  8%, #7fffd4 0%, transparent 100%),
        radial-gradient(circle 1px at 93% 16%, #ffffff 0%, transparent 100%),
        radial-gradient(circle 1px at 97%  4%, #7fffd4 0%, transparent 100%),
        radial-gradient(circle 1.5px at  7% 35%, #7fffd4 0%, transparent 100%),
        radial-gradient(circle 1px   at 15% 42%, #7fffd4 0%, transparent 100%),
        radial-gradient(circle 1px   at 23% 55%, #ffffff 0%, transparent 100%),
        radial-gradient(circle 1.5px at 31% 48%, #7fffd4 0%, transparent 100%),
        radial-gradient(circle 1px   at 39% 38%, #7fffd4 0%, transparent 100%),
        radial-gradient(circle 1px   at 47% 62%, #7fffd4 0%, transparent 100%),
        radial-gradient(circle 1.5px at 55% 44%, #ffffff 0%, transparent 100%),
        radial-gradient(circle 1px   at 63% 57%, #7fffd4 0%, transparent 100%),
        radial-gradient(circle 1px   at 72% 33%, #7fffd4 0%, transparent 100%),
        radial-gradient(circle 1.5px at 80% 50%, #ffffff 0%, transparent 100%),
        radial-gradient(circle 1px   at 88% 40%, #7fffd4 0%, transparent 100%),
        radial-gradient(circle 1px   at 95% 60%, #7fffd4 0%, transparent 100%),
        radial-gradient(circle 1px at  3% 75%, #7fffd4 0%, transparent 100%),
        radial-gradient(circle 1px at 10% 82%, #ffffff 0%, transparent 100%),
        radial-gradient(circle 1px at 18% 70%, #7fffd4 0%, transparent 100%),
        radial-gradient(circle 1px at 26% 88%, #7fffd4 0%, transparent 100%),
        radial-gradient(circle 1px at 34% 78%, #ffffff 0%, transparent 100%),
        radial-gradient(circle 1px at 42% 92%, #7fffd4 0%, transparent 100%),
        radial-gradient(circle 1px at 50% 72%, #7fffd4 0%, transparent 100%),
        radial-gradient(circle 1px at 58% 85%, #ffffff 0%, transparent 100%),
        radial-gradient(circle 1px at 66% 68%, #7fffd4 0%, transparent 100%),
        radial-gradient(circle 1px at 74% 90%, #7fffd4 0%, transparent 100%),
        radial-gradient(circle 1px at 82% 76%, #ffffff 0%, transparent 100%),
        radial-gradient(circle 1px at 90% 83%, #7fffd4 0%, transparent 100%),
        radial-gradient(circle 1px at 96% 95%, #7fffd4 0%, transparent 100%);
    animation: twinkle-a 5s ease-in-out infinite alternate;
    opacity: 0.75;
}
[data-testid="stAppViewContainer"]::after {
    content:'';position:fixed;inset:0;z-index:0;pointer-events:none;
    background-image:
        radial-gradient(circle 2px at  8% 28%, #7fffd4 0%, transparent 100%),
        radial-gradient(circle 2px at 22% 65%, #7fffd4 0%, transparent 100%),
        radial-gradient(circle 2px at 45% 30%, #ffffff 0%, transparent 100%),
        radial-gradient(circle 2px at 67% 72%, #7fffd4 0%, transparent 100%),
        radial-gradient(circle 2px at 84% 45%, #7fffd4 0%, transparent 100%),
        radial-gradient(circle 2px at 13% 90%, #ffffff 0%, transparent 100%),
        radial-gradient(circle 2px at 57% 15%, #7fffd4 0%, transparent 100%),
        radial-gradient(circle 2px at 91% 55%, #7fffd4 0%, transparent 100%);
    animation: twinkle-b 7s 2s ease-in-out infinite alternate;
    opacity: 0.5;
}

html, body,
[data-testid="stAppViewContainer"],
[data-testid="stMain"] {
    background-color: var(--bg) !important;
    color: var(--text) !important;
    font-family: 'DM Sans', sans-serif !important;
}
.main .block-container {
    background-color: var(--bg) !important;
    max-width: 1100px !important;
    margin: 0 auto !important;
    padding: 0 20px !important;
}
[data-testid="stHeader"]  { background: transparent !important; }
[data-testid="stSidebar"] { background: rgba(11,26,23,0.97) !important; }

#MainMenu, footer, header    { visibility: hidden; }
[data-testid="stToolbar"]    { display: none; }
[data-testid="stDecoration"] { display: none; }

h1, h2, h3 {
    font-family: 'Bebas Neue', sans-serif !important;
    letter-spacing: 3px;
    color: #fff !important;
}

[data-testid="stVerticalBlockBorderWrapper"] {
    background: var(--card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    backdrop-filter: blur(6px) !important;
    -webkit-backdrop-filter: blur(6px) !important;
}

[data-testid="stTextInput"] input {
    background: #0a1812 !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    color: var(--text) !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 14px !important;
    padding: 12px 16px !important;
}
[data-testid="stTextInput"] input:focus {
    border-color: var(--green) !important;
    box-shadow: 0 0 0 2px var(--green-glow) !important;
}
[data-testid="stTextInput"] input::placeholder { color: var(--muted) !important; }

.stButton > button {
    background: var(--green) !important;
    color: #000 !important;
    font-family: 'Bebas Neue', sans-serif !important;
    font-size: 18px !important;
    letter-spacing: 2px !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 10px 32px !important;
    transition: all 0.2s ease !important;
    width: 100%;
}
.stButton > button:hover {
    background: #3de077 !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 20px var(--green-glow) !important;
}

.card-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 9px;
    letter-spacing: 3px;
    color: var(--muted);
    text-transform: uppercase;
    margin-bottom: 10px;
}
.track-name {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 28px;
    letter-spacing: 1px;
    line-height: 1.1;
    color: #fff;
}
.track-artist { font-size: 12px; color: var(--muted); margin-top: 2px; }

.badge-hit {
    display: inline-flex; align-items: center; gap: 6px;
    background: rgba(46,204,113,0.10); border: 1px solid var(--green);
    color: var(--green); font-family: 'JetBrains Mono', monospace;
    font-size: 11px; letter-spacing: 2px; padding: 6px 14px; border-radius: 20px;
}
.badge-flop {
    display: inline-flex; align-items: center; gap: 6px;
    background: rgba(232,54,93,0.10); border: 1px solid var(--red);
    color: var(--red); font-family: 'JetBrains Mono', monospace;
    font-size: 11px; letter-spacing: 2px; padding: 6px 14px; border-radius: 20px;
}

.prob-bar-bg {
    background: #1a2e28; border-radius: 3px;
    height: 6px; overflow: hidden; margin-top: 4px;
}
.prob-bar-fill-hit {
    background: var(--green); height: 100%; border-radius: 3px;
    transition: width 0.8s cubic-bezier(0.4,0,0.2,1);
}
.prob-bar-fill-flop {
    background: var(--red); height: 100%; border-radius: 3px;
    transition: width 0.8s cubic-bezier(0.4,0,0.2,1);
}

.explanation-text {
    font-family: 'DM Sans', sans-serif; font-size: 12px; line-height: 1.65;
    color: var(--label); font-style: italic;
    border-left: 2px solid var(--green-dim); padding-left: 10px; margin-top: 10px;
}

.radar-feat-row {
    display: flex; justify-content: space-between;
    font-family: 'JetBrains Mono', monospace;
    font-size: 9px; letter-spacing: 1px; color: var(--muted); margin-bottom: 3px;
}
.radar-feat-row span:last-child { color: var(--green); }

[data-testid="stSlider"] > div > div > div > div { background: var(--green) !important; }
[data-testid="stSlider"] [data-testid="stTickBar"] { display: none !important; }
[data-testid="stSlider"] div[role="slider"] {
    background: var(--green) !important;
    border: 2px solid var(--green) !important;
    box-shadow: 0 0 6px var(--green-glow) !important;
}
[data-testid="stSlider"] > div > div > div { background: var(--border) !important; }
[data-testid="stSlider"] label {
    font-family: 'DM Sans', sans-serif !important;
    font-size: 13px !important; color: var(--text) !important;
}
[data-testid="stSlider"] [data-testid="stThumbValue"],
[data-testid="stSlider"] p {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 11px !important; color: var(--green) !important;
}

[data-testid="stSelectbox"] label {
    font-family: 'DM Sans', sans-serif !important;
    font-size: 13px !important; color: var(--text) !important;
}
[data-testid="stSelectbox"] > div > div {
    background: #0a1812 !important; border: 1px solid var(--border) !important;
    border-radius: 6px !important; color: var(--text) !important;
    font-family: 'JetBrains Mono', monospace !important; font-size: 12px !important;
}
[data-testid="stSelectbox"] > div > div:focus-within {
    border-color: var(--green) !important; box-shadow: 0 0 0 1px var(--green) !important;
}
[data-testid="stSelectbox"] svg { fill: var(--muted) !important; }

[data-testid="stNumberInput"] label {
    font-family: 'DM Sans', sans-serif !important;
    font-size: 13px !important; color: var(--text) !important;
}
[data-testid="stNumberInput"] input {
    background: #0a1812 !important; border: 1px solid var(--border) !important;
    border-radius: 6px !important; color: var(--text) !important;
    font-family: 'JetBrains Mono', monospace !important; font-size: 12px !important;
}
[data-testid="stNumberInput"] input:focus {
    border-color: var(--green) !important; box-shadow: 0 0 0 1px var(--green) !important;
}
[data-testid="stNumberInput"] button {
    background: var(--border) !important; border: none !important;
    color: var(--text) !important; border-radius: 4px !important;
}
[data-testid="stNumberInput"] button:hover {
    background: var(--green) !important; color: #000 !important;
}

.js-plotly-plot,
.js-plotly-plot .plotly,
.js-plotly-plot .svg-container { background: transparent !important; }

[data-testid="stSpinner"] p {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 11px !important; letter-spacing: 2px !important;
    color: var(--label) !important;
}

::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }

hr { border-color: var(--border) !important; }

[data-testid="stAlert"] {
    background: rgba(14,30,26,0.8) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 12px !important;
}
</style>
""", unsafe_allow_html=True)

    # Second injection — loads AFTER Streamlit's internal theme so it wins
    st.markdown("""
<style>
div[data-testid="stSegmentedControl"] {
    display: flex !important;
    justify-content: center !important;
    background: transparent !important;
    border: none !important;
    padding: 0 !important;
    margin-bottom: 24px !important;
}
div[data-testid="stSegmentedControl"] > div:first-child {
    background: transparent !important;
    border: none !important;
    padding: 4px !important;
    gap: 8px !important;
    display: flex !important;
}
div[data-testid="stSegmentedControl"] [data-testid="stSegmentedControlActiveThumb"] {
    display: none !important;
}
div[data-testid="stSegmentedControl"] button {
    background: transparent !important;
    border: 1px solid #1e3530 !important;
    color: #4a7060 !important;
    border-radius: 24px !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 13px !important;
    letter-spacing: 2px !important;
    text-transform: uppercase !important;
    padding: 12px 32px !important;
    min-width: 150px !important;
    text-align: center !important;
    transition: all 0.2s ease !important;
    box-shadow: none !important;
    outline: none !important;
    margin: 0 !important;
}
div[data-testid="stSegmentedControl"] button:hover {
    color: #d4ede5 !important;
    border-color: #1a9e52 !important;
    background: transparent !important;
}
div[data-testid="stSegmentedControl"] button[aria-checked="true"] {
    background: rgba(46,204,113,0.12) !important;
    border: 1px solid #2ecc71 !important;
    color: #2ecc71 !important;
    box-shadow: none !important;
}
div[data-testid="stSegmentedControl"] button[aria-checked="false"] {
    background: transparent !important;
    border: 1px solid #1e3530 !important;
    color: #4a7060 !important;
}
div[data-testid="stSegmentedControl"] button p,
div[data-testid="stSegmentedControl"] button span {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 13px !important;
    letter-spacing: 2px !important;
    text-transform: uppercase !important;
    margin: 0 !important;
}
div[data-testid="stSegmentedControl"] button[aria-checked="true"] p,
div[data-testid="stSegmentedControl"] button[aria-checked="true"] span {
    color: #2ecc71 !important;
}
div[data-testid="stSegmentedControl"] button[aria-checked="false"] p,
div[data-testid="stSegmentedControl"] button[aria-checked="false"] span {
    color: #4a7060 !important;
}
</style>
""", unsafe_allow_html=True)