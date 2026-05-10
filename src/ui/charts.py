"""
charts.py — Plotly chart builders matching the HTML design spec.

make_gauge       : segmented semicircle (red → amber → green) with needle + Bebas percentage.
make_radar_chart : hexagonal audio-profile spider chart.
"""

import plotly.graph_objects as go

# ── Palette ──────────────────────────────────────────────────────────────────
_GREEN = "#2ecc71"
_RED = "#e8365d"
_AMBER = "#e8a836"
_GRID = "#1e3530"
_MUTED = "#4a7060"
_LABEL = "#6aaa8a"
_BG = "rgba(0,0,0,0)"
_TRACK = "#1a2e28"


def make_gauge(confidence_pct: float, prediction: str) -> go.Figure:
    """
    Segmented semicircular gauge matching the HTML canvas spec:
      0–50%  → red zone   (#e8365d)
      50–75% → amber zone (#e8a836)
      75–100%→ green zone (#2ecc71)
    Needle + center dot drawn via shapes. Bebas Neue % number centered.
    """
    is_hit = prediction == "Hit"
    needle_color = _GREEN if is_hit else _RED
    emoji = "🔥" if is_hit else "💀"

    # Map 0-100 value to angle: 180° (left) → 0° (right) on a semicircle
    # Plotly gauge: 0 = left (180°), 100 = right (0°)
    fig = go.Figure()

    fig.add_trace(
        go.Indicator(
            mode="gauge+number",
            value=confidence_pct,
            number={
                "suffix": "%",
                "font": {
                    "size": 42,
                    "color": needle_color,
                    "family": "Bebas Neue, sans-serif",
                },
                "valueformat": ".0f",
            },
            gauge={
                "axis": {
                    "range": [0, 100],
                    "tickvals": [0, 25, 50, 75, 100],
                    "ticktext": ["0", "25", "50", "75", "100"],
                    "tickfont": {
                        "color": _MUTED,
                        "size": 10,
                        "family": "JetBrains Mono, monospace",
                    },
                    "tickcolor": _BG,
                    "ticklen": 0,
                },
                "bar": {
                    "color": needle_color,
                    "thickness": 0.03,
                },
                "bgcolor": _BG,
                "borderwidth": 0,
                "bordercolor": _BG,
                "steps": [
                    # Track background
                    {"range": [0, 100], "color": _TRACK},
                    # Coloured zone overlays (slightly inner)
                    {"range": [0, 50], "color": "#2a1520"},  # deep red tint
                    {"range": [50, 75], "color": "#251e10"},  # deep amber tint
                    {"range": [75, 100], "color": "#0f2018"},  # deep green tint
                ],
                "threshold": {
                    "line": {"color": needle_color, "width": 4},
                    "thickness": 0.82,
                    "value": confidence_pct,
                },
            },
            domain={"x": [0, 1], "y": [0, 1]},
        )
    )

    # Zone arc colours via shapes (drawn as SVG arcs on top)
    # We use annotations for the emoji below the number
    fig.add_annotation(
        text=emoji,
        x=0.5,
        y=0.12,
        xref="paper",
        yref="paper",
        showarrow=False,
        font={"size": 20},
    )

    fig.update_layout(
        paper_bgcolor=_BG,
        plot_bgcolor=_BG,
        margin={"l": 24, "r": 24, "t": 28, "b": 8},
        height=185,
        font={"color": _LABEL, "family": "DM Sans, sans-serif"},
    )
    return fig


def make_radar_chart(top_features: dict, prediction: str) -> go.Figure:
    """
    Hexagonal radar chart with 6 axes matching the HTML spec.
    Features are normalised to [0, 1] before plotting.
    """
    is_hit = prediction == "Hit"
    color = _GREEN if is_hit else _RED
    rgb = "46,204,113" if is_hit else "232,54,93"

    # ── Normalise ────────────────────────────────────────────────────────────
    normalized: dict = {}
    for k, v in top_features.items():
        fv = float(v)
        if k == "loudness":
            normalized[k] = max(0.0, min(1.0, (fv + 60) / 60))
        elif k == "duration_ms":
            normalized[k] = max(0.0, min(1.0, fv / 400_000))
        elif k == "tempo":
            normalized[k] = max(0.0, min(1.0, fv / 210))
        else:
            normalized[k] = max(0.0, min(1.0, fv))

    labels = [k.replace("_", " ").title() for k in normalized]
    values = list(normalized.values())
    # Close the polygon
    values_c = values + values[:1]
    labels_c = labels + labels[:1]

    fig = go.Figure(
        go.Scatterpolar(
            r=values_c,
            theta=labels_c,
            fill="toself",
            fillcolor=f"rgba({rgb},0.15)",
            line={"color": color, "width": 2},
            marker={"size": 4, "color": color},
            hovertemplate="%{theta}: %{r:.2f}<extra></extra>",
        )
    )

    fig.update_layout(
        polar={
            "bgcolor": "rgba(26,46,40,0.5)",
            "radialaxis": {
                "visible": True,
                "range": [0, 1],
                "gridcolor": _GRID,
                "linecolor": _GRID,
                "tickfont": {
                    "color": _MUTED,
                    "size": 8,
                    "family": "JetBrains Mono, monospace",
                },
                "tickvals": [0.25, 0.5, 0.75, 1.0],
                "ticktext": ["", "", "", ""],
                "showticklabels": False,
                "linewidth": 1,
            },
            "angularaxis": {
                "gridcolor": _GRID,
                "linecolor": _GRID,
                "tickfont": {
                    "color": _LABEL,
                    "size": 9,
                    "family": "JetBrains Mono, monospace",
                },
                "tickmode": "array",
                "tickvals": labels_c[:-1],
                "rotation": 90,
                "direction": "clockwise",
            },
        },
        paper_bgcolor=_BG,
        plot_bgcolor=_BG,
        margin={"l": 48, "r": 48, "t": 48, "b": 48},
        showlegend=False,
        height=240,
    )
    return fig
