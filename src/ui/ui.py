"""
ui.py — All reusable UI components matching the HTML design spec.
  • top_nav()             — top navigation bar
  • render_header()       — page title + subtitle
  • render_footer()       — bottom status bar
  • render_manual_inputs()— native Streamlit 3-col audio feature input grid
  • render_result()       — single-row 3-column result layout:
                             [Track Analysis | Key Drivers | Audio Profile]
"""

import streamlit as st
from charts import make_radar_chart

_GREEN = "#2ecc71"
_RED = "#e8365d"
_AMBER = "#e8a836"
_MUTED = "#4a7060"
_LABEL = "#6aaa8a"
_BG = "#0e1e1a"
_BORDER = "#1e3530"


# ─────────────────────────────────────────────────────────────────────────────
# Page chrome
# ─────────────────────────────────────────────────────────────────────────────


def top_nav():
    if "page" not in st.session_state:
        st.session_state["page"] = "home"

    current_page = st.session_state["page"]

    st.markdown(
        """
    <div class="top-nav">
    <div class="brand">
        <div class="brand-mark">
        <svg viewBox="0 0 24 24" width="20" height="20">
            <circle cx="12" cy="12" r="10" fill="none" stroke="currentColor" stroke-width="1.4"/>
            <circle cx="12" cy="12" r="3.5" fill="none" stroke="currentColor" stroke-width="1.4"/>
            <circle cx="12" cy="12" r="0.9" fill="currentColor"/>
            <path d="M12 2 V 5.5 M12 18.5 V 22 M2 12 H 5.5 M18.5 12 H 22"
                stroke="currentColor" stroke-width="1.4" stroke-linecap="round"/>
        </svg>
        </div>
        <div>
        <span class="brand-name">Ai-Dj/STUDIO</span>
        <span class="brand-sub">audio forecasting · v2.4</span>
        </div>
    </div>
    <div class="meta-strip">
        <span><i></i>Model: RandomForest</span>
        <span><i></i>Features: 13</span>
        <span><i></i>Live</span>
    </div>
    </div>
""",
        unsafe_allow_html=True,
    )

    col1, col2, *_ = st.columns([1, 1, 8])
    with col1:
        if st.button(
            "Home",
            key="nav_home",
            type="primary" if current_page == "home" else "secondary",
        ):
            st.session_state["page"] = "home"
            st.rerun()
    with col2:
        if st.button(
            "EDA",
            key="nav_eda",
            type="primary" if current_page == "eda" else "secondary",
        ):
            st.session_state["page"] = "eda"
            st.rerun()

    return st.session_state["page"]


def render_header():
    st.markdown(
        """
<div style="
    display:flex;
    align-items:center;
    justify-content:space-between;
    background:#0a1410;
    border:1px solid #1a2e26;
    border-radius:12px;
    padding:40px 48px;
    margin-bottom:32px;
    gap:32px;
    overflow:hidden;
    position:relative;
">
  <div style="flex:1.4;min-width:0;">
    <div style="
        display:flex;align-items:center;gap:8px;
        font-family:'JetBrains Mono',monospace;
        font-size:11px;letter-spacing:3px;
        color:#2ecc71;text-transform:uppercase;
        margin-bottom:20px;">
      <span style="width:7px;height:7px;border-radius:50%;background:#2ecc71;display:inline-block;"></span>
      HIT FORECASTING ENGINE
    </div>
    <div style="
        font-family:'Inter','DM Sans',sans-serif;
        font-size:clamp(40px,5.5vw,72px);
        font-weight:800;
        line-height:1.05;
        color:#ffffff;
        letter-spacing:-1.5px;
        margin-bottom:24px;">
      Predict the next<br/>
      <em style="font-style:italic;color:#2ecc71;font-weight:800;">chart-topper.</em>
    </div>
    <p style="
        font-family:'Inter','DM Sans',sans-serif;
        font-size:14px;color:#4a7060;line-height:1.7;
        max-width:420px;margin:0;">
      Tune a track's audio DNA — danceability, energy, key, tempo
      — and see whether the model thinks it has hit potential. Built
      on Spotify-style audio features and an ensemble classifier.
    </p>
  </div>

  <div style="
      flex:0.7;display:flex;flex-direction:column;
      align-items:flex-end;justify-content:center;
      gap:12px;min-width:220px;">
    <div style="align-self:flex-end;margin-bottom:4px;">
      <svg viewBox="0 0 100 100" width="80" height="80">
        <circle cx="50" cy="50" r="46" fill="none" stroke="#2ecc71" stroke-width="1.2" opacity="0.5"/>
        <circle cx="50" cy="50" r="32" fill="none" stroke="#2ecc71" stroke-width="1.2" opacity="0.4"/>
        <circle cx="50" cy="50" r="18" fill="none" stroke="#2ecc71" stroke-width="1.2" opacity="0.6"/>
        <circle cx="50" cy="50" r="5"  fill="none" stroke="#2ecc71" stroke-width="1.4"/>
        <circle cx="50" cy="50" r="1.8" fill="#2ecc71"/>
      </svg>
    </div>
    <svg viewBox="0 0 260 90" width="260" height="90" xmlns="http://www.w3.org/2000/svg">
      <defs>
        <linearGradient id="wg" x1="0" y1="0" x2="1" y2="0">
          <stop offset="0%"   stop-color="#2ecc71" stop-opacity="0.35"/>
          <stop offset="50%"  stop-color="#2ecc71" stop-opacity="1"/>
          <stop offset="100%" stop-color="#2ecc71" stop-opacity="0.5"/>
        </linearGradient>
      </defs>
      <rect x="0"   y="36" width="4" height="18" rx="2" fill="url(#wg)"/>
      <rect x="8"   y="28" width="4" height="34" rx="2" fill="url(#wg)"/>
      <rect x="16"  y="20" width="4" height="50" rx="2" fill="url(#wg)"/>
      <rect x="24"  y="30" width="4" height="30" rx="2" fill="url(#wg)"/>
      <rect x="32"  y="15" width="4" height="60" rx="2" fill="url(#wg)"/>
      <rect x="40"  y="32" width="4" height="26" rx="2" fill="url(#wg)"/>
      <rect x="48"  y="22" width="4" height="46" rx="2" fill="url(#wg)"/>
      <rect x="56"  y="10" width="4" height="70" rx="2" fill="url(#wg)"/>
      <rect x="64"  y="25" width="4" height="40" rx="2" fill="url(#wg)"/>
      <rect x="72"  y="18" width="4" height="54" rx="2" fill="url(#wg)"/>
      <rect x="80"  y="33" width="4" height="24" rx="2" fill="url(#wg)"/>
      <rect x="88"  y="8"  width="4" height="74" rx="2" fill="url(#wg)"/>
      <rect x="96"  y="28" width="4" height="34" rx="2" fill="url(#wg)"/>
      <rect x="104" y="20" width="4" height="50" rx="2" fill="url(#wg)"/>
      <rect x="112" y="35" width="4" height="20" rx="2" fill="url(#wg)"/>
      <rect x="120" y="12" width="4" height="66" rx="2" fill="url(#wg)"/>
      <rect x="128" y="26" width="4" height="38" rx="2" fill="url(#wg)"/>
      <rect x="136" y="18" width="4" height="54" rx="2" fill="url(#wg)"/>
      <rect x="144" y="30" width="4" height="30" rx="2" fill="url(#wg)"/>
      <rect x="152" y="22" width="4" height="46" rx="2" fill="url(#wg)"/>
      <rect x="160" y="14" width="4" height="62" rx="2" fill="url(#wg)"/>
      <rect x="168" y="32" width="4" height="26" rx="2" fill="url(#wg)"/>
      <rect x="176" y="24" width="4" height="42" rx="2" fill="url(#wg)"/>
      <rect x="184" y="6"  width="4" height="78" rx="2" fill="url(#wg)"/>
      <rect x="192" y="28" width="4" height="34" rx="2" fill="url(#wg)"/>
      <rect x="200" y="20" width="4" height="50" rx="2" fill="url(#wg)"/>
      <rect x="208" y="34" width="4" height="22" rx="2" fill="url(#wg)"/>
      <rect x="216" y="16" width="4" height="58" rx="2" fill="url(#wg)"/>
      <rect x="224" y="28" width="4" height="34" rx="2" fill="url(#wg)"/>
      <rect x="232" y="10" width="4" height="70" rx="2" fill="url(#wg)"/>
      <rect x="240" y="30" width="4" height="30" rx="2" fill="url(#wg)"/>
      <rect x="248" y="22" width="4" height="46" rx="2" fill="url(#wg)"/>
      <rect x="256" y="38" width="4" height="14" rx="2" fill="url(#wg)"/>
    </svg>
    <div style="font-family:'JetBrains Mono',monospace;font-size:10px;
                letter-spacing:2px;color:#2a5040;text-transform:uppercase;
                align-self:flex-start;margin-top:4px;">
      TRK &nbsp;·&nbsp; 04:21
    </div>
    <div style="font-family:'JetBrains Mono',monospace;font-size:9px;
                letter-spacing:2px;color:#1e3a2e;text-transform:uppercase;
                align-self:flex-start;">
      WAVEFORM &nbsp;·&nbsp; SAMPLE 0024
    </div>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )


def render_footer():
    st.markdown(
        """
<div style="
    margin-top:60px;border-top:1px solid #1e3530;padding:16px 0;
    display:flex;justify-content:space-between;align-items:center;
    position:relative;z-index:1;">
    <span style="font-family:'JetBrains Mono',monospace;
                 font-size:11px;color:#4a7060;letter-spacing:1px;">
        🟢 API: Cloud Run &nbsp;·&nbsp; MODEL: RandomForest
        &nbsp;·&nbsp; FEATURES: Musicae API
    </span>
    <span style="font-family:'JetBrains Mono',monospace;
                 font-size:11px;color:#4a7060;letter-spacing:1px;">
        INFO9023 · Team AI-DJ
    </span>
</div>
""",
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Manual input — native Streamlit widgets, styled via CSS
# Returns a dict of features when PREDICT is clicked, else None
# ─────────────────────────────────────────────────────────────────────────────


def render_manual_inputs() -> dict | None:
    """
    3-column audio feature input grid using native Streamlit widgets.
    Returns the feature dict when the user clicks PREDICT, else None.
    """
    st.markdown(
        """
<style>
div[data-testid="stVerticalBlockBorderWrapper"] > div[data-testid="stVerticalBlock"],
div[data-testid="stVerticalBlockBorderWrapper"] {
    background-color: #0a1410 !important;
    background: #0a1410 !important;
    border: 1px solid #1a2e26 !important;
    border-radius: 12px !important;
}
</style>
""",
        unsafe_allow_html=True,
    )

    st.markdown(
        "<div style=\"font-family:'JetBrains Mono',monospace;font-size:9px;"
        "letter-spacing:3px;color:#4a7060;text-transform:uppercase;"
        'margin-bottom:14px;">Manually Input Audio Features</div>',
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns(3, gap="medium")

    with col1:
        with st.container(border=True):
            danceability = st.slider(
                "Danceability", 0.0, 1.0, 0.70, 0.01, key="m_dance"
            )
            energy = st.slider("Energy", 0.0, 1.0, 0.70, 0.01, key="m_energy")
            valence = st.slider("Valence", 0.0, 1.0, 0.60, 0.01, key="m_valence")
            speechiness = st.slider("Speechiness", 0.0, 1.0, 0.05, 0.01, key="m_speech")

    with col2:
        with st.container(border=True):
            acousticness = st.slider(
                "Acousticness", 0.0, 1.0, 0.10, 0.01, key="m_acoustic"
            )
            instrumentalness = st.slider(
                "Instrumentalness", 0.0, 1.0, 0.00, 0.01, key="m_instru"
            )
            liveness = st.slider("Liveness", 0.0, 1.0, 0.08, 0.01, key="m_live")
            mode_label = st.selectbox("Mode", ["Major", "Minor"], key="m_mode")
            mode = 1 if mode_label == "Major" else 0

    with col3:
        with st.container(border=True):
            loudness = st.slider("Loudness (dB)", -60.0, 0.0, -9.4, 0.1, key="m_loud")
            tempo = st.slider("Tempo (BPM)", 50.0, 210.0, 110.0, 0.5, key="m_tempo")
            key_names = [
                "C",
                "C#",
                "D",
                "D#",
                "E",
                "F",
                "F#",
                "G",
                "G#",
                "A",
                "A#",
                "B",
            ]
            key_label = st.selectbox("Key", key_names, key="m_key")
            key_val = key_names.index(key_label)
            duration_ms = st.number_input(
                "Duration [ms]", 30000, 600000, 200000, 1000, key="m_dur"
            )
            time_signature = st.selectbox(
                "Time Signature", [3, 4, 5, 6, 7], index=1, key="m_timesig"
            )

    st.markdown("<div style='margin-top:4px'>", unsafe_allow_html=True)
    clicked = st.button("▶ PREDICT", key="manual_predict_btn")
    st.markdown("</div>", unsafe_allow_html=True)

    if clicked:
        return {
            "danceability": danceability,
            "energy": energy,
            "valence": valence,
            "speechiness": speechiness,
            "acousticness": acousticness,
            "instrumentalness": instrumentalness,
            "liveness": liveness,
            "mode": mode,
            "loudness": loudness,
            "tempo": tempo,
            "key": key_val,
            "duration_ms": int(duration_ms),
            "time_signature": int(time_signature),
        }
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Result renderer — single row, 3 equal columns
# ─────────────────────────────────────────────────────────────────────────────


def render_result(result: dict):
    prediction = result.get("prediction", "Unknown")
    track_name = result.get("track_name", "Unknown")
    artist = result.get("artist", "Unknown")
    thumbnail = result.get("thumbnail", "")
    explanation = result.get("explanation", "")
    top_features = result.get("top_features", {})
    recs = result.get("if_you_liked_this", [])

    is_hit = prediction == "Hit"

    st.markdown(
        '<div style="position:relative;z-index:1;margin-top:8px"></div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        """
<style>
[data-testid="stHorizontalBlock"] > [data-testid="stColumn"] > div > div >
[data-testid="stVerticalBlockBorderWrapper"] {
    height: 100%;
}
[data-testid="stHorizontalBlock"] > [data-testid="stColumn"] > div > div >
[data-testid="stVerticalBlockBorderWrapper"] > div[data-testid="stVerticalBlock"] {
    height: 100%;
}
[data-testid="stHorizontalBlock"] > [data-testid="stColumn"] {
    display: flex;
    flex-direction: column;
}
[data-testid="stHorizontalBlock"] > [data-testid="stColumn"] > div {
    flex: 1;
    display: flex;
    flex-direction: column;
}
[data-testid="stHorizontalBlock"] > [data-testid="stColumn"] > div > div {
    flex: 1;
    display: flex;
    flex-direction: column;
}
</style>
""",
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns([1, 1, 1])

    with col1:
        _render_track_analysis(track_name, artist, thumbnail, is_hit, explanation)
    with col2:
        _render_key_drivers(top_features, prediction)
    with col3:
        _render_audio_profile(top_features, prediction)

    if recs:
        st.markdown("<br>", unsafe_allow_html=True)
        _render_recommendations(recs)


# ─────────────────────────────────────────────────────────────────────────────
# Private sub-renderers
# ─────────────────────────────────────────────────────────────────────────────


def _render_track_analysis(track_name, artist, thumbnail, is_hit, explanation=""):
    emoji = "🔥" if is_hit else "💀"
    badge_class = "badge-hit" if is_hit else "badge-flop"
    label = "CERTIFIED HIT" if is_hit else "CERTIFIED FLOP"

    with st.container(border=True):
        st.markdown(
            '<div class="card-label">Track Analysis</div>', unsafe_allow_html=True
        )
        if thumbnail:
            st.image(thumbnail, width=88)
        st.markdown(
            f'<div class="track-name">{track_name}</div>'
            f'<div class="track-artist">{artist}</div>',
            unsafe_allow_html=True,
        )
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(
            f'<span class="{badge_class}">{label} {emoji}</span>',
            unsafe_allow_html=True,
        )
        if explanation:
            st.markdown(
                f'<div class="explanation-text" style="margin-top:16px;">{explanation}</div>',
                unsafe_allow_html=True,
            )


def _render_key_drivers(top_features: dict, prediction: str):
    is_hit = prediction == "Hit"
    color = _GREEN if is_hit else _RED

    with st.container(border=True):
        st.markdown('<div class="card-label">Key Drivers</div>', unsafe_allow_html=True)
        st.markdown(
            "<div style=\"font-family:'JetBrains Mono',monospace;font-size:22px;"
            'font-weight:700;color:#ffffff;margin-bottom:16px;letter-spacing:-0.5px;">'
            "What's pushing the verdict</div>",
            unsafe_allow_html=True,
        )

        if top_features:
            max_abs = max(abs(float(v)) for v in top_features.values()) or 1.0
            for i, (feat, val) in enumerate(top_features.items(), start=1):
                try:
                    fval = float(val)
                except (ValueError, TypeError):
                    continue
                bar_pct = int(round(abs(fval) / max_abs * 100))
                disp_key = feat.replace("_", " ").title()
                feat_low = feat.lower()
                if "loudness" in feat_low:
                    disp_val = f"{fval:.1f} dB"
                elif "tempo" in feat_low:
                    disp_val = f"{int(fval)} BPM"
                elif "duration" in feat_low:
                    disp_val = f"{int(fval / 1000)}s"
                else:
                    disp_val = f"{fval:.2f}"
                st.markdown(
                    f"""
<div style="display:flex;align-items:center;gap:10px;margin-bottom:11px;">
    <span style="font-family:'JetBrains Mono',monospace;font-size:9px;
                 color:#444;min-width:18px;letter-spacing:1px;">{i:02d}</span>
    <span style="font-family:'JetBrains Mono',monospace;font-size:11px;
                 color:#bbbbbb;min-width:90px;letter-spacing:0.5px;">{disp_key}</span>
    <div style="flex:1;background:#1a2e26;border-radius:3px;height:5px;overflow:hidden;">
        <div style="width:{bar_pct}%;height:100%;background:{color};
                    border-radius:3px;transition:width 0.4s ease;"></div>
    </div>
    <span style="font-family:'JetBrains Mono',monospace;font-size:11px;
                 color:{color};min-width:58px;text-align:right;
                 letter-spacing:0.5px;">{disp_val}</span>
</div>""",
                    unsafe_allow_html=True,
                )
        else:
            st.markdown(
                f"<div style=\"color:{_MUTED};font-family:'JetBrains Mono',monospace;"
                f'font-size:11px;text-align:center;padding:40px 0;">NO FEATURE DATA</div>',
                unsafe_allow_html=True,
            )


def _render_audio_profile(top_features: dict, prediction: str):
    is_hit = prediction == "Hit"
    color = _GREEN if is_hit else _RED

    with st.container(border=True):
        st.markdown(
            '<div class="card-label">Audio Profile</div>', unsafe_allow_html=True
        )
        st.markdown(
            "<div style=\"font-family:'JetBrains Mono',monospace;font-size:22px;"
            'font-weight:700;color:#ffffff;margin-bottom:4px;letter-spacing:-0.5px;">'
            "Feature radar</div>",
            unsafe_allow_html=True,
        )

        if top_features:
            st.plotly_chart(
                make_radar_chart(top_features, prediction),
                use_container_width=True,
                config={"displayModeBar": False},
            )

            feat_items = []
            for feat, val in top_features.items():
                feat_low = feat.lower()
                disp_key = feat.replace("_", " ").title()
                try:
                    fval = float(val)
                    if "loudness" in feat_low:
                        disp_val = f"{fval:.1f} dB"
                    elif "tempo" in feat_low:
                        disp_val = f"{int(fval)} BPM"
                    elif "duration" in feat_low:
                        disp_val = f"{int(fval / 1000)}s"
                    else:
                        disp_val = f"{fval:.2f}"
                except (ValueError, TypeError):
                    disp_val = str(val)
                feat_items.append((disp_key, disp_val))

            mid = (len(feat_items) + 1) // 2
            left_feats = feat_items[:mid]
            right_feats = feat_items[mid:]
            gc1, gc2 = st.columns(2)

            row_style = "display:flex;justify-content:space-between;align-items:baseline;margin-bottom:14px;"
            key_style = "font-family:'DM Sans',sans-serif;font-size:12px;color:#888;letter-spacing:0.3px;"
            val_style = f"font-family:'JetBrains Mono',monospace;font-size:13px;font-weight:700;color:{color};"

            for gcol, items in ((gc1, left_feats), (gc2, right_feats)):
                with gcol:
                    for k, v in items:
                        st.markdown(
                            f"""
<div style="{row_style}">
    <span style="{key_style}">{k}</span>
    <span style="{val_style}">{v}</span>
</div>""",
                            unsafe_allow_html=True,
                        )
        else:
            st.markdown(
                f"<div style=\"color:{_MUTED};font-family:'JetBrains Mono',monospace;"
                f'font-size:11px;text-align:center;padding:40px 0;">NO FEATURE DATA</div>',
                unsafe_allow_html=True,
            )


def _render_recommendations(recs):
    with st.container(border=True):
        st.markdown(
            '<div class="card-label">🎵 If You Liked This — You Might Also Love</div>',
            unsafe_allow_html=True,
        )
        rec_cols = st.columns(len(recs))
        for i, rec in enumerate(recs):
            with rec_cols[i]:
                if rec.get("thumbnail"):
                    st.image(rec["thumbnail"], use_container_width=True)
                st.markdown(
                    f"""
<div style="margin-top:8px;">
    <div style="font-family:'DM Sans',sans-serif;font-size:13px;
                font-weight:600;color:#d4ede5;">{rec.get("track_name", "")}</div>
    <div style="font-size:11px;color:{_MUTED};margin-top:2px;
                font-family:'DM Sans',sans-serif;">{rec.get("artist", "")}</div>
    <a href="{rec.get("spotify_url", "#")}" target="_blank"
       style="font-size:11px;color:{_GREEN};text-decoration:none;
              font-family:'JetBrains Mono',monospace;
              letter-spacing:1px;margin-top:4px;display:block;">
        ▶ OPEN IN SPOTIFY
    </a>
</div>""",
                    unsafe_allow_html=True,
                )


def render_tabs() -> str:
    if "active_tab" not in st.session_state:
        st.session_state["active_tab"] = "ANALYZE"

    tabs = ["ANALYZE", "MANUAL INPUT", "HISTORY"]

    choice = st.segmented_control(
        label="Navigation",
        options=tabs,
        default=st.session_state["active_tab"],
        label_visibility="collapsed",
        key="nav_bar",
    )

    if choice and choice != st.session_state["active_tab"]:
        st.session_state["active_tab"] = choice
        st.rerun()

    return st.session_state["active_tab"]
