"""Reusable Streamlit UI components for the PCB GenAI application."""

from __future__ import annotations

from typing import Any

import streamlit as st


STATUS_COLORS = {
    "PASS": ("#166534", "#dcfce7", "#86efac"),
    "FAIL": ("#991b1b", "#fee2e2", "#fca5a5"),
}

SEVERITY_COLORS = {
    "LOW": ("#365314", "#ecfccb", "#bef264"),
    "MEDIUM": ("#92400e", "#fef3c7", "#fcd34d"),
    "HIGH": ("#9a3412", "#ffedd5", "#fdba74"),
    "CRITICAL": ("#991b1b", "#fee2e2", "#fca5a5"),
}

DIFFICULTY_COLORS = {
    "EASY": ("#166534", "#dcfce7", "#86efac"),
    "MODERATE": ("#92400e", "#fef3c7", "#fcd34d"),
    "HARD": ("#9a3412", "#ffedd5", "#fdba74"),
    "EXPERT": ("#991b1b", "#fee2e2", "#fca5a5"),
}


def inject_global_styles() -> None:
    """Add compact CSS used by badges and cards."""

    st.markdown(
        """
        <style>
        .pcb-badge {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            min-width: 88px;
            padding: 0.35rem 0.7rem;
            border-radius: 999px;
            border: 1px solid;
            font-weight: 700;
            letter-spacing: 0;
            line-height: 1.2;
            white-space: nowrap;
        }
        .pcb-metric-row {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
            gap: 0.75rem;
            margin: 0.75rem 0 1rem 0;
        }
        .pcb-metric {
            border: 1px solid rgba(148, 163, 184, 0.35);
            border-radius: 8px;
            padding: 0.75rem;
            background: rgba(248, 250, 252, 0.72);
        }
        .pcb-metric-label {
            color: #475569;
            font-size: 0.82rem;
            margin-bottom: 0.2rem;
        }
        .pcb-metric-value {
            color: #0f172a;
            font-size: 1.15rem;
            font-weight: 700;
        }
        .severity-chip {
            display: inline-flex;
            padding: 0.2rem 0.55rem;
            border-radius: 999px;
            border: 1px solid;
            font-size: 0.78rem;
            font-weight: 700;
            line-height: 1.2;
        }
        .dashboard-hero {
            border: 1px solid rgba(148, 163, 184, 0.32);
            border-radius: 8px;
            padding: 1rem;
            background: #f8fafc;
            margin-bottom: 1rem;
        }
        .dashboard-hero h3 {
            margin: 0 0 0.35rem 0;
            color: #0f172a;
        }
        .dashboard-hero p {
            margin: 0;
            color: #475569;
        }
        .dashboard-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
            gap: 0.75rem;
            margin: 0.75rem 0 1rem 0;
        }
        .dashboard-card {
            border: 1px solid rgba(148, 163, 184, 0.35);
            border-radius: 8px;
            padding: 0.85rem;
            background: #ffffff;
            min-height: 92px;
        }
        .dashboard-card-label {
            color: #64748b;
            font-size: 0.8rem;
            margin-bottom: 0.3rem;
        }
        .dashboard-card-value {
            color: #0f172a;
            font-size: 1.35rem;
            font-weight: 750;
            line-height: 1.15;
            overflow-wrap: anywhere;
        }
        .dashboard-card-help {
            color: #64748b;
            font-size: 0.78rem;
            margin-top: 0.35rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_badge(label: str, palette: tuple[str, str, str] | None = None) -> None:
    """Render a colored pill badge."""

    color, background, border = palette or ("#334155", "#f1f5f9", "#cbd5e1")
    st.markdown(
        f"""
        <span class="pcb-badge" style="color:{color}; background:{background}; border-color:{border};">
            {escape_html(label)}
        </span>
        """,
        unsafe_allow_html=True,
    )


def render_dashboard_hero(title: str, description: str) -> None:
    """Render the dashboard introduction panel."""

    st.markdown(
        f"""
        <div class="dashboard-hero">
            <h3>{escape_html(title)}</h3>
            <p>{escape_html(description)}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_dashboard_metric(label: str, value: Any, help_text: str = "") -> None:
    """Render one compact dashboard metric card."""

    help_markup = f'<div class="dashboard-card-help">{escape_html(help_text)}</div>' if help_text else ""
    st.markdown(
        f"""
        <div class="dashboard-card">
            <div class="dashboard-card-label">{escape_html(label)}</div>
            <div class="dashboard-card-value">{escape_html(value)}</div>
            {help_markup}
        </div>
        """,
        unsafe_allow_html=True,
    )


def open_dashboard_grid() -> None:
    """Open a responsive dashboard card grid."""

    st.markdown('<div class="dashboard-grid">', unsafe_allow_html=True)


def close_dashboard_grid() -> None:
    """Close a responsive dashboard card grid."""

    st.markdown("</div>", unsafe_allow_html=True)


def render_status_badge(status: str) -> None:
    """Render PASS/FAIL status."""

    normalized = str(status).upper()
    render_badge(normalized, STATUS_COLORS.get(normalized))


def render_difficulty_badge(difficulty: str) -> None:
    """Render reconstruction difficulty."""

    normalized = str(difficulty).upper()
    render_badge(normalized, DIFFICULTY_COLORS.get(normalized))


def render_quality_score(score: int | float) -> None:
    """Render a bounded quality score progress bar."""

    try:
        numeric_score = int(float(score))
    except (TypeError, ValueError):
        numeric_score = 0
    numeric_score = max(0, min(100, numeric_score))
    st.metric("Quality score", f"{numeric_score}/100")
    st.progress(numeric_score / 100)


def render_detection_summary(result: dict[str, Any]) -> None:
    """Render top-level fault-detection summary."""

    col_status, col_quality = st.columns([1, 2])
    with col_status:
        st.caption("Overall status")
        render_status_badge(str(result.get("overall_status", "FAIL")))
    with col_quality:
        render_quality_score(result.get("quality_score", 0))

    confidence = result.get("confidence_score", 0.0)
    try:
        confidence_text = f"{float(confidence) * 100:.1f}%"
    except (TypeError, ValueError):
        confidence_text = "Unavailable"

    st.markdown(
        f"""
        <div class="pcb-metric-row">
            <div class="pcb-metric">
                <div class="pcb-metric-label">Model confidence</div>
                <div class="pcb-metric-value">{escape_html(confidence_text)}</div>
            </div>
            <div class="pcb-metric">
                <div class="pcb-metric-label">Defects found</div>
                <div class="pcb-metric-value">{len(result.get("defects_found", []))}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.write(result.get("summary", "No summary provided."))


def render_defect_cards(defects: list[dict[str, Any]]) -> None:
    """Render defects as expandable cards with severity coding."""

    st.subheader("Defects")
    if not defects:
        st.success("No defects were reported by the AI inspection.")
        return

    for index, defect in enumerate(defects, start=1):
        severity = str(defect.get("severity", "LOW")).upper()
        color, background, border = SEVERITY_COLORS.get(severity, SEVERITY_COLORS["LOW"])
        title = f"{index}. {defect.get('defect_type', 'Defect')} - {severity}"
        with st.expander(title, expanded=index == 1):
            st.markdown(
                f"""
                <span class="severity-chip" style="color:{color}; background:{background}; border-color:{border};">
                    {escape_html(severity)}
                </span>
                """,
                unsafe_allow_html=True,
            )
            st.write(f"**Location:** {defect.get('location', 'Not specified')}")
            st.write(f"**Description:** {defect.get('description', 'Not provided')}")
            st.write(f"**Recommended action:** {defect.get('recommended_action', 'Inspect and repair as needed.')}")


def render_component_analysis_table(component_analysis: dict[str, Any]) -> None:
    """Render component analysis as a two-column table."""

    st.subheader("Component Analysis")
    rows = [
        {"Area": "Solder joints", "Assessment": component_analysis.get("solder_joints", "Not assessed.")},
        {"Area": "Traces", "Assessment": component_analysis.get("traces", "Not assessed.")},
        {"Area": "Components", "Assessment": component_analysis.get("components", "Not assessed.")},
        {"Area": "Substrate", "Assessment": component_analysis.get("substrate", "Not assessed.")},
    ]
    st.dataframe(rows, hide_index=True, use_container_width=True)


def render_reconstruction_header(result: dict[str, Any]) -> None:
    """Render board type and repair difficulty."""

    board_type = result.get("board_type_identified", "Unknown PCB type")
    st.caption("Board type identified")
    st.markdown(f"### {board_type}")
    st.caption("Estimated repair difficulty")
    render_difficulty_badge(str(result.get("estimated_repair_difficulty", "MODERATE")))
    st.write(result.get("summary", "No summary provided."))


def render_reconstruction_steps(steps: list[dict[str, Any]]) -> None:
    """Render reconstruction steps as a numbered checklist."""

    st.subheader("Reconstruction Checklist")
    if not steps:
        st.info("No reconstruction steps were returned.")
        return

    for step in sorted(steps, key=lambda item: item.get("step_number", 0)):
        number = step.get("step_number", 0)
        label = f"Step {number}: {step.get('area', 'Unspecified area')}"
        checked = st.checkbox(label, key=f"reconstruction_step_{number}_{hash(label)}")
        with st.expander("Details", expanded=not checked):
            st.write(f"**Action:** {step.get('action', 'No action specified.')}")
            st.write(f"**Expected result:** {step.get('expected_result', 'Not specified.')}")
            st.write(f"**Tools needed:** {step.get('tools_needed', 'Standard PCB repair tools')}")


def render_healthy_board_table(description: dict[str, Any]) -> None:
    """Render healthy board expectations as a comparison table."""

    st.subheader("Healthy Board Description")
    rows = [
        {"Aspect": "Solder joints", "Healthy condition": description.get("solder_joints", "Not described.")},
        {"Aspect": "Traces", "Healthy condition": description.get("traces", "Not described.")},
        {"Aspect": "Components", "Healthy condition": description.get("components", "Not described.")},
        {"Aspect": "Surface", "Healthy condition": description.get("surface", "Not described.")},
    ]
    st.dataframe(rows, hide_index=True, use_container_width=True)


def render_reference_prompt(prompt: str) -> None:
    """Render reference image prompt in a copy-friendly text area."""

    st.subheader("Reference Image Prompt")
    st.text_area(
        "Copy this prompt into an image generation tool",
        value=prompt,
        height=220,
        label_visibility="visible",
    )


def render_raw_response_debug(raw_response: str | None) -> None:
    """Render raw service text for JSON troubleshooting."""

    if not raw_response:
        return
    with st.expander("Raw AI response for debugging"):
        st.code(raw_response, language="json")


def escape_html(value: Any) -> str:
    """Small HTML escaper for badge text."""

    text = str(value)
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&#x27;")
    )
