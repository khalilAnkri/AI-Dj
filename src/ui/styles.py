"""
styles.py — Global CSS for the dark-teal AI-DJ theme.
Pixel-perfect match to the reference HTML design spec.
"""

import base64
import os
import streamlit as st


def _bg_css():
    """Read Background.png from the same folder and return a CSS block."""
    img_path = os.path.join(os.path.dirname(__file__), "assets", "Background.png")
    try:
        with open(img_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        return f"""
<style>
body,
.stApp,
[data-testid="stApp"],
[data-testid="stAppViewContainer"] {{
    background-image: url("data:image/png;base64,{b64}") !important;
    background-size: cover !important;
    background-position: center top !important;
    background-attachment: fixed !important;
    background-repeat: no-repeat !important;
    background-color: transparent !important;
}}
html {{
    background-image: url("data:image/png;base64,{b64}") !important;
    background-size: cover !important;
    background-position: center top !important;
    background-repeat: no-repeat !important;
}}
</style>
"""
    except FileNotFoundError:
        return ""  # silently skip if image not found


def inject_css():
    st.markdown(_bg_css(), unsafe_allow_html=True)
    st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=DM+Sans:wght@300;400;500;600&family=JetBrains+Mono:wght@400;600;700&display=swap');

/* ── Design tokens ──────────────────────────────────────────
   Single source of truth — all components reference these.
   ---------------------------------------------------------- */
:root {
    /* Backgrounds */
    --bg:           #0b1a17;
    --bg-1:         #0f211d;
    --bg-2:         #0e1e1a;
    --card:         #0a1410;
    --card2:        #0a1410;

    /* Borders / lines */
    --border:       #1e3530;
    --line:         #1e3530;
    --line-soft:    #162c26;
    --track-bg:     #1a2e28;

    /* Accent / green */
    --accent:       #2ecc71;
    --green:        #2ecc71;
    --green-dim:    #1a9e52;
    --green-glow:   rgba(46, 204, 113, 0.25);

    /* Red */
    --red:          #e8365d;
    --red-glow:     rgba(232, 54, 93, 0.25);

    /* Amber */
    --amber:        #e8a836;

    /* Foreground scale */
    --fg-0:         #d4ede5;   /* primary text   */
    --fg-1:         #6aaa8a;   /* label / dimmed  */
    --fg-2:         #4a7060;   /* muted           */
    --fg-3:         #2a5040;   /* very muted      */

    /* Legacy aliases kept for backward compat */
    --text:         #d4ede5;
    --muted:        #4a7060;
    --label:        #6aaa8a;

    /* hue angle used by oklch accents (≈ green 142°) */
    --ah:           142;
}


/* ── Base layout  ─────────────────────────────────────────── */
html, body,
[data-testid="stAppViewContainer"],
[data-testid="stMain"] {
    background-color: transparent !important;
    color: var(--fg-0) !important;
    font-family: 'DM Sans', sans-serif !important;
}
.main .block-container {
    background-color: transparent !important;
    max-width: 100% !important;
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

/* ── Top nav ─────────────────────────────────────────────── */
.top-nav {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding-bottom: 14px;
    border-bottom: 1px solid var(--line-soft);
    margin-bottom: 32px;
}
.brand {
    display: flex;
    align-items: center;
    gap: 12px;
}
.brand-mark {
    width: 36px;
    height: 36px;
    display: grid;
    place-items: center;
    background: var(--bg-1);
    border: 1px solid var(--line);
    border-radius: 50%;
    color: var(--accent);
}
.brand-name {
    font-family: 'JetBrains Mono', ui-monospace, monospace;
    font-size: 12px;
    letter-spacing: 0.18em;
    color: var(--fg-0);
    display: block;
}
.brand-sub {
    font-family: 'JetBrains Mono', ui-monospace, monospace;
    font-size: 10.5px;
    color: var(--fg-3);
    letter-spacing: 0.08em;
    margin-top: 3px;
    display: block;
}
.meta-strip {
    display: flex;
    gap: 22px;
    font-family: 'JetBrains Mono', ui-monospace, monospace;
    font-size: 11px;
    color: var(--fg-2);
    letter-spacing: 0.06em;
}
.meta-strip span {
    display: inline-flex;
    align-items: center;
    gap: 8px;
}
.meta-strip i {
    width: 6px;
    height: 6px;
    background: var(--accent);
    border-radius: 50%;
    box-shadow: 0 0 0 3px oklch(0.78 0.18 142 / 0.18);
    font-style: normal;
}
.meta-strip span:nth-child(2) i,
.meta-strip span:nth-child(3) i {
    background: var(--fg-3);
    box-shadow: none;
}

/* ── Cards ───────────────────────────────────────────────── */
[data-testid="stVerticalBlockBorderWrapper"],
[data-testid="stVerticalBlockBorderWrapper"] > div,
[data-testid="stVerticalBlockBorderWrapper"] > div > div,
div[data-testid="stVerticalBlock"] > div[class*="block-container"],
div.stVerticalBlock > div[data-testid="stVerticalBlockBorderWrapper"] {
    background: #0a1410 !important;
    background-color: #0a1410 !important;
    border: 1px solid #1a2e26 !important;
    border-radius: 12px !important;
    backdrop-filter: blur(6px) !important;
    -webkit-backdrop-filter: blur(6px) !important;
}

/* ── Text input ──────────────────────────────────────────── */
[data-testid="stTextInput"] input {
    background: #0a1812 !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    color: var(--fg-0) !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 14px !important;
    padding: 12px 16px !important;
}
[data-testid="stTextInput"] input:focus {
    border-color: var(--green) !important;
    box-shadow: 0 0 0 2px var(--green-glow) !important;
}
[data-testid="stTextInput"] input::placeholder { color: var(--muted) !important; }

/* ── Buttons ─────────────────────────────────────────────── */
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

/* ── Result card components ──────────────────────────────── */
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
.track-artist {
    font-size: 12px;
    color: var(--muted);
    margin-top: 2px;
}
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
    background: var(--track-bg); border-radius: 3px;
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

/* ── Sliders ─────────────────────────────────────────────── */
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
    font-size: 13px !important; color: var(--fg-0) !important;
}
[data-testid="stSlider"] [data-testid="stThumbValue"],
[data-testid="stSlider"] p {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 11px !important; color: var(--green) !important;
}

/* ── Selectbox ───────────────────────────────────────────── */
[data-testid="stSelectbox"] label {
    font-family: 'DM Sans', sans-serif !important;
    font-size: 13px !important; color: var(--fg-0) !important;
}
[data-testid="stSelectbox"] > div > div {
    background: #0a1812 !important; border: 1px solid var(--border) !important;
    border-radius: 6px !important; color: var(--fg-0) !important;
    font-family: 'JetBrains Mono', monospace !important; font-size: 12px !important;
}
[data-testid="stSelectbox"] > div > div:focus-within {
    border-color: var(--green) !important; box-shadow: 0 0 0 1px var(--green) !important;
}
[data-testid="stSelectbox"] svg { fill: var(--muted) !important; }

/* ── Number input ────────────────────────────────────────── */
[data-testid="stNumberInput"] label {
    font-family: 'DM Sans', sans-serif !important;
    font-size: 13px !important; color: var(--fg-0) !important;
}
[data-testid="stNumberInput"] input {
    background: #0a1812 !important; border: 1px solid var(--border) !important;
    border-radius: 6px !important; color: var(--fg-0) !important;
    font-family: 'JetBrains Mono', monospace !important; font-size: 12px !important;
}
[data-testid="stNumberInput"] input:focus {
    border-color: var(--green) !important; box-shadow: 0 0 0 1px var(--green) !important;
}
[data-testid="stNumberInput"] button {
    background: var(--border) !important; border: none !important;
    color: var(--fg-0) !important; border-radius: 4px !important;
}
[data-testid="stNumberInput"] button:hover {
    background: var(--green) !important; color: #000 !important;
}

/* ── Plotly ──────────────────────────────────────────────── */
.js-plotly-plot,
.js-plotly-plot .plotly,
.js-plotly-plot .svg-container { background: transparent !important; }

/* ── Misc ────────────────────────────────────────────────── */
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
    border: 1px solid var(--border) !important;
    color: var(--muted) !important;
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
    color: var(--fg-0) !important;
    border-color: var(--green-dim) !important;
    background: transparent !important;
}
div[data-testid="stSegmentedControl"] button[aria-checked="true"] {
    background: rgba(46,204,113,0.12) !important;
    border: 1px solid var(--green) !important;
    color: var(--green) !important;
    box-shadow: none !important;
}
div[data-testid="stSegmentedControl"] button[aria-checked="false"] {
    background: transparent !important;
    border: 1px solid var(--border) !important;
    color: var(--muted) !important;
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
    color: var(--green) !important;
}
div[data-testid="stSegmentedControl"] button[aria-checked="false"] p,
div[data-testid="stSegmentedControl"] button[aria-checked="false"] span {
    color: var(--muted) !important;
}
</style>
""", unsafe_allow_html=True)