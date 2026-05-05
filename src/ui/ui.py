"""
ui.py — All reusable UI components matching the HTML design spec.
  • render_header()        — page title + subtitle with Bebas Neue
  • render_footer()        — bottom status bar
  • render_manual_inputs() — native Streamlit 3-col audio feature input grid
  • render_result()        — 3-column result cards (track analysis / gauge / radar)
"""

import streamlit as st
from charts import make_gauge, make_radar_chart

_GREEN = "#2ecc71"
_RED   = "#e8365d"
_AMBER = "#e8a836"
_MUTED = "#4a7060"
_LABEL = "#6aaa8a"
_BG    = "#0e1e1a"
_BORDER= "#1e3530"


# ─────────────────────────────────────────────────────────────────────────────
# Page chrome
# ─────────────────────────────────────────────────────────────────────────────

def render_header():
    st.markdown("""
<div style="text-align:center;padding:42px 0 24px;position:relative;z-index:1;">
    <h1 style="
        font-family:'Bebas Neue',sans-serif;
        font-size:clamp(32px,4.5vw,52px);
        letter-spacing:3px;
        margin:0;line-height:1;
        color:#fff;
        text-transform:uppercase;">
        Predict the Next <span style="color:#2ecc71;">Chart-Topper.</span>
    </h1>
    <p style="
        color:#4a7060;
        font-size:11px;
        margin-top:10px;
        font-family:'JetBrains Mono',monospace;
        letter-spacing:3px;
        text-transform:uppercase;">
        Powered by Random Forest &nbsp;·&nbsp; Spotify Audio Features &nbsp;·&nbsp; Musicae API
    </p>
</div>
""", unsafe_allow_html=True)


def render_footer():
    st.markdown("""
<div style="
    margin-top:60px;
    border-top:1px solid #1e3530;
    padding:16px 0;
    display:flex;
    justify-content:space-between;
    align-items:center;
    position:relative;z-index:1;">
    <span style="
        font-family:'JetBrains Mono',monospace;
        font-size:11px;color:#4a7060;letter-spacing:1px;">
        🟢 API: localhost:8080 &nbsp;·&nbsp; MODEL: RandomForest
        &nbsp;·&nbsp; FEATURES: Musicae API
    </span>
    <span style="
        font-family:'JetBrains Mono',monospace;
        font-size:11px;color:#4a7060;letter-spacing:1px;">
        INFO9023 · Team AI-DJ
    </span>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Manual input — native Streamlit widgets, styled via CSS
# Returns a dict of features when PREDICT is clicked, else None
# ─────────────────────────────────────────────────────────────────────────────

def render_manual_inputs() -> dict | None:
    """
    3-column audio feature input grid using native Streamlit widgets.
    Returns the feature dict when the user clicks PREDICT, else None.
    """
    _label = lambda txt: st.markdown(
        f'<div style="font-family:\'DM Sans\',sans-serif;font-size:13px;'
        f'color:#d4ede5;margin-bottom:2px;">{txt}</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div style="font-family:\'JetBrains Mono\',monospace;font-size:9px;'
        'letter-spacing:3px;color:#4a7060;text-transform:uppercase;'
        'margin-bottom:14px;">Manually Input Audio Features</div>',
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns(3, gap="medium")

    # ── Col 1 ──────────────────────────────────────────────────────────────
    with col1:
        with st.container(border=True):
            danceability = st.slider("Danceability",    0.0, 1.0, 0.70, 0.01, key="m_dance")
            energy       = st.slider("Energy",          0.0, 1.0, 0.70, 0.01, key="m_energy")
            valence      = st.slider("Valence",         0.0, 1.0, 0.60, 0.01, key="m_valence")
            speechiness  = st.slider("Speechiness",     0.0, 1.0, 0.05, 0.01, key="m_speech")

    # ── Col 2 ──────────────────────────────────────────────────────────────
    with col2:
        with st.container(border=True):
            acousticness     = st.slider("Acousticness",     0.0, 1.0, 0.10, 0.01, key="m_acoustic")
            instrumentalness = st.slider("Instrumentalness", 0.0, 1.0, 0.00, 0.01, key="m_instru")
            liveness         = st.slider("Liveness",         0.0, 1.0, 0.08, 0.01, key="m_live")
            mode_label = st.selectbox("Mode", ["Major", "Minor"], key="m_mode")
            mode = 1 if mode_label == "Major" else 0

    # ── Col 3 ──────────────────────────────────────────────────────────────
    with col3:
        with st.container(border=True):
            loudness = st.slider("Loudness (dB)", -60.0, 0.0, -9.4, 0.1, key="m_loud")
            tempo    = st.slider("Tempo (BPM)",    50.0, 210.0, 110.0, 0.5, key="m_tempo")
            key_names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
            key_label = st.selectbox("Key", key_names, key="m_key")
            key_val   = key_names.index(key_label)
            duration_ms   = st.number_input("Duration [ms]", 30000, 600000, 200000, 1000, key="m_dur")
            time_signature = st.selectbox("Time Signature", [3, 4, 5, 6, 7], index=1, key="m_timesig")

    # ── PREDICT button ──────────────────────────────────────────────────────
    st.markdown("<div style='margin-top:4px'>", unsafe_allow_html=True)
    clicked = st.button("▶ PREDICT", key="manual_predict_btn")
    st.markdown("</div>", unsafe_allow_html=True)

    if clicked:
        return {
            "danceability":     danceability,
            "energy":           energy,
            "valence":          valence,
            "speechiness":      speechiness,
            "acousticness":     acousticness,
            "instrumentalness": instrumentalness,
            "liveness":         liveness,
            "mode":             mode,
            "loudness":         loudness,
            "tempo":            tempo,
            "key":              key_val,
            "duration_ms":      int(duration_ms),
            "time_signature":   int(time_signature),
        }
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Result renderer — 3-column layout
# ─────────────────────────────────────────────────────────────────────────────

def render_result(result: dict):
    prediction   = result.get("prediction", "Unknown")
    confidence   = result.get("confidence", "0%")
    hit_prob     = result.get("hit_probability", "0%")
    flop_prob    = result.get("flop_probability", "0%")
    track_name   = result.get("track_name", "Unknown")
    artist       = result.get("artist", "Unknown")
    thumbnail    = result.get("thumbnail", "")
    explanation  = result.get("explanation", "")
    top_features = result.get("top_features", {})
    recs         = result.get("if_you_liked_this", [])

    conf_val = float(str(confidence).replace("%", ""))
    hit_val  = float(str(hit_prob).replace("%", ""))
    flop_val = float(str(flop_prob).replace("%", ""))
    is_hit   = prediction == "Hit"

    st.markdown('<div style="position:relative;z-index:1;margin-top:8px"></div>',
                unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 1.25, 1])

    with col1:
        _render_track_analysis(
            track_name, artist, thumbnail, is_hit,
            hit_prob, hit_val, flop_prob, flop_val
        )
    with col2:
        _render_prediction_insights(conf_val, prediction, is_hit, explanation)
    with col3:
        _render_audio_radar(top_features, prediction)

    if recs:
        st.markdown("<br>", unsafe_allow_html=True)
        _render_recommendations(recs)


# ─────────────────────────────────────────────────────────────────────────────
# Private sub-renderers
# ─────────────────────────────────────────────────────────────────────────────

def _render_track_analysis(track_name, artist, thumbnail, is_hit,
                            hit_prob, hit_val, flop_prob, flop_val):
    emoji       = "🔥" if is_hit else "💀"
    badge_class = "badge-hit" if is_hit else "badge-flop"
    label       = "CERTIFIED HIT" if is_hit else "CERTIFIED FLOP"
    color       = _GREEN if is_hit else _RED

    with st.container(border=True):
        st.markdown('<div class="card-label">Track Analysis</div>',
                    unsafe_allow_html=True)
        if thumbnail:
            st.image(thumbnail, width=88)
        st.markdown(
            f'<div class="track-name">{track_name}</div>'
            f'<div class="track-artist">{artist}</div>',
            unsafe_allow_html=True
        )
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(
            f'<span class="{badge_class}">{label} {emoji}</span>',
            unsafe_allow_html=True
        )
        st.markdown("<br><br>", unsafe_allow_html=True)
        st.markdown(f"""
<div style="font-family:'JetBrains Mono',monospace;font-size:9px;
            letter-spacing:2px;color:{_MUTED};text-transform:uppercase;
            margin-bottom:4px;">Hit Probability ▾</div>
<div class="prob-bar-bg">
    <div class="prob-bar-fill-hit" style="width:{hit_val}%"></div>
</div>
<div style="font-family:'JetBrains Mono',monospace;font-size:11px;
            color:{_GREEN};margin-top:2px;">{hit_prob}</div>

<div style="font-family:'JetBrains Mono',monospace;font-size:9px;
            letter-spacing:2px;color:{_MUTED};text-transform:uppercase;
            margin-top:12px;margin-bottom:4px;">Flop Probability ▾</div>
<div class="prob-bar-bg">
    <div class="prob-bar-fill-flop" style="width:{flop_val}%"></div>
</div>
<div style="font-family:'JetBrains Mono',monospace;font-size:11px;
            color:{_RED};margin-top:2px;">{flop_prob}</div>
""", unsafe_allow_html=True)


def _render_prediction_insights(conf_val, prediction, is_hit, explanation):
    color   = _GREEN if is_hit else _RED
    emoji   = "🔥" if is_hit else "💀"
    verdict = "CERTIFIED HIT" if is_hit else "CERTIFIED FLOP"

    with st.container(border=True):
        st.markdown('<div class="card-label">Prediction &amp; Insights</div>',
                    unsafe_allow_html=True)
        st.plotly_chart(
            make_gauge(conf_val, prediction),
            use_container_width=True,
            config={"displayModeBar": False},
        )
        st.markdown(f"""
<div style="display:flex;justify-content:space-between;
            font-family:'JetBrains Mono',monospace;font-size:9px;
            letter-spacing:2px;margin-top:-6px;padding:0 4px;">
    <span style="color:{_RED}">FLOP</span>
    <span style="color:{_GREEN}">CHART-TOPPER</span>
</div>
<div style="text-align:center;font-family:'JetBrains Mono',monospace;
            font-size:12px;letter-spacing:3px;text-transform:uppercase;
            color:{color};margin-top:10px;">
    {verdict} {emoji}
</div>
""", unsafe_allow_html=True)
        if explanation:
            st.markdown(
                f'<div class="explanation-text">{explanation}</div>',
                unsafe_allow_html=True
            )


def _render_audio_radar(top_features, prediction):
    is_hit = prediction == "Hit"
    color  = _GREEN if is_hit else _RED

    with st.container(border=True):
        st.markdown('<div class="card-label">Audio Profile Radar</div>',
                    unsafe_allow_html=True)
        if top_features:
            st.plotly_chart(
                make_radar_chart(top_features, prediction),
                use_container_width=True,
                config={"displayModeBar": False},
            )
            # Feature table below radar
            for feat, val in top_features.items():
                display_key = feat.replace("_", " ").upper()
                try:
                    display_val = f"{float(val):.3f}"
                except (ValueError, TypeError):
                    display_val = str(val)
                st.markdown(f"""
<div class="radar-feat-row">
    <span>{display_key}</span>
    <span style="color:{color}">{display_val}</span>
</div>""", unsafe_allow_html=True)
        else:
            st.markdown(
                f'<div style="color:{_MUTED};font-family:\'JetBrains Mono\','
                f'monospace;font-size:11px;text-align:center;padding:40px 0;">'
                f'NO FEATURE DATA</div>',
                unsafe_allow_html=True
            )


def _render_recommendations(recs):
    with st.container(border=True):
        st.markdown(
            '<div class="card-label">🎵 If You Liked This — You Might Also Love</div>',
            unsafe_allow_html=True
        )
        rec_cols = st.columns(len(recs))
        for i, rec in enumerate(recs):
            with rec_cols[i]:
                if rec.get("thumbnail"):
                    st.image(rec["thumbnail"], use_container_width=True)
                st.markdown(f"""
<div style="margin-top:8px;">
    <div style="font-family:'DM Sans',sans-serif;font-size:13px;
                font-weight:600;color:#d4ede5;">{rec.get("track_name","")}</div>
    <div style="font-size:11px;color:{_MUTED};margin-top:2px;
                font-family:'DM Sans',sans-serif;">{rec.get("artist","")}</div>
    <a href="{rec.get('spotify_url','#')}" target="_blank"
       style="font-size:11px;color:{_GREEN};text-decoration:none;
              font-family:'JetBrains Mono',monospace;
              letter-spacing:1px;margin-top:4px;display:block;">
        ▶ OPEN IN SPOTIFY
    </a>
</div>""", unsafe_allow_html=True)
                
def render_tabs() -> str:
    if "active_tab" not in st.session_state:
        st.session_state["active_tab"] = "ANALYZE"

    tabs = ["ANALYZE", "MANUAL INPUT", "HISTORY"]

    choice = st.segmented_control(
        label="Navigation",
        options=tabs,
        default=st.session_state["active_tab"],
        label_visibility="collapsed",
        key="nav_bar"
    )

    if choice and choice != st.session_state["active_tab"]:
        st.session_state["active_tab"] = choice
        st.rerun()

    return st.session_state["active_tab"]