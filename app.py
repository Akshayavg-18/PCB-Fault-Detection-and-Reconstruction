"""Streamlit app for PCB fault detection and healthy PCB reconstruction."""

from __future__ import annotations

import os
from datetime import datetime
from typing import Any, Callable

import streamlit as st
from dotenv import load_dotenv
from google.api_core.exceptions import GoogleAPIError

from detector import detect_pcb_faults
from gemini_client import (
    DEFAULT_MODEL_NAME,
    IMAGE_MODEL_NAME,
    SUPPORTED_MODELS,
    GeminiConfigurationError,
    GeminiImageGenerationError,
    GeminiResponseParseError,
    create_client,
)
from image_utils import (
    create_local_reconstruction_preview,
    human_readable_size,
    preprocess_image,
    validate_image_bytes,
)
from reconstructor import generate_reconstruction_guide
from ui_components import (
    inject_global_styles,
    render_dashboard_hero,
    render_dashboard_metric,
    render_component_analysis_table,
    render_defect_cards,
    render_detection_summary,
    render_difficulty_badge,
    render_healthy_board_table,
    render_raw_response_debug,
    render_reconstruction_header,
    render_reconstruction_steps,
    render_reference_prompt,
    render_status_badge,
)


load_dotenv()


MODEL_DISPLAY_NAMES = {
    "gemini-2.5-flash": "Fast Vision Model",
    "gemini-2.5-pro": "Advanced Vision Model",
    "gemini-2.0-flash": "Fast Vision Model v2",
    "gemini-2.0-flash-lite": "Lite Vision Model",
    "gemini-flash-latest": "Latest Fast Vision Model",
    "gemini-pro-latest": "Latest Advanced Vision Model",
    "gemini-1.5-flash": "Legacy Fast Vision Model",
    "gemini-1.5-pro": "Legacy Advanced Vision Model",
}


def initialize_session_state() -> None:
    """Initialize Streamlit session keys used across tabs."""

    defaults = {
        "uploaded_image_bytes": None,
        "uploaded_image_name": None,
        "processed_image_bytes": None,
        "processed_image_pil": None,
        "processed_enhance_contrast": None,
        "upload_metadata": None,
        "detection_result": None,
        "reconstruction_result": None,
        "reconstructed_image_bytes": None,
        "reconstructed_image_source": None,
        "inspection_history": [],
        "last_error_raw_response": None,
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def configure_page() -> None:
    """Configure page title, layout, and shared styles."""

    st.set_page_config(
        page_title="PCB Fault Detection and Reconstruction",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    inject_global_styles()


def render_sidebar() -> tuple[str, str, bool]:
    """Render API and preprocessing controls."""

    st.sidebar.header("AI Settings")
    env_key = os.getenv("GEMINI_API_KEY", "")
    api_key = st.sidebar.text_input(
        "API key",
        value=env_key,
        type="password",
        help="Stored only in this Streamlit session unless you set the local environment variable.",
    )
    model_name = st.sidebar.selectbox(
        "Analysis model",
        options=list(SUPPORTED_MODELS),
        index=list(SUPPORTED_MODELS).index(DEFAULT_MODEL_NAME),
        format_func=lambda value: MODEL_DISPLAY_NAMES.get(value, "Vision Model"),
        help="Choose a fast model for routine checks or an advanced model for more detailed inspection.",
    )

    st.sidebar.header("Preprocessing")
    enhance_contrast = st.sidebar.checkbox(
        "Enhance contrast with CLAHE",
        value=True,
        help="Applies contrast enhancement on the LAB L channel before analysis.",
    )
    return api_key, model_name, enhance_contrast


def handle_upload(enhance_contrast: bool) -> None:
    """Handle image upload, validation, and preprocessing."""

    uploaded_file = st.file_uploader(
        "Upload PCB image",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=False,
        help="Upload a JPG or PNG image up to 10MB.",
    )

    if uploaded_file is None:
        return

    image_bytes = uploaded_file.getvalue()
    validation = validate_image_bytes(image_bytes, uploaded_file.name)
    if not validation.is_valid:
        st.error(validation.message)
        return

    image_changed = (
        st.session_state.uploaded_image_bytes != image_bytes
        or st.session_state.uploaded_image_name != uploaded_file.name
        or st.session_state.processed_enhance_contrast != enhance_contrast
    )

    if image_changed or st.session_state.processed_image_bytes is None:
        try:
            processed = preprocess_image(image_bytes, enhance_contrast=enhance_contrast)
        except ValueError as exc:
            st.error(str(exc))
            return

        st.session_state.uploaded_image_bytes = image_bytes
        st.session_state.uploaded_image_name = uploaded_file.name
        st.session_state.processed_image_bytes = processed.image_bytes
        st.session_state.processed_image_pil = processed.pil_image
        st.session_state.processed_enhance_contrast = enhance_contrast
        st.session_state.upload_metadata = {
            "name": uploaded_file.name,
            "format": validation.format,
            "width": validation.width,
            "height": validation.height,
            "size": human_readable_size(validation.size_bytes),
            "processed_width": processed.processed_size[0],
            "processed_height": processed.processed_size[1],
            "contrast_enhanced": enhance_contrast,
        }
        st.session_state.detection_result = None
        st.session_state.reconstruction_result = None
        st.session_state.reconstructed_image_bytes = None
        st.session_state.reconstructed_image_source = None
        st.session_state.last_error_raw_response = None

        st.success(
            "Image ready: "
            f"{validation.width}x{validation.height}, {validation.format}, "
            f"{human_readable_size(validation.size_bytes)}."
        )


def get_client_or_show_error(api_key: str, model_name: str):
    """Create a Gemini client and render configuration errors."""

    try:
        return create_client(api_key=api_key, model_name=model_name)
    except GeminiConfigurationError as exc:
        st.error(sanitize_visible_error(str(exc)))
        return None


def sanitize_visible_error(message: str) -> str:
    """Remove provider-specific wording from UI-facing errors."""

    replacements = {
        "Gemini": "AI service",
        "gemini": "vision-model",
        "GEMINI_API_KEY": "API key",
        "Google": "AI provider",
        "google": "AI provider",
    }
    sanitized = message
    for source, target in replacements.items():
        sanitized = sanitized.replace(source, target)
    return sanitized


def render_uploaded_image_panel() -> None:
    """Render the uploaded or preprocessed PCB image in the left column."""

    st.subheader("Uploaded PCB")
    if st.session_state.processed_image_pil is None:
        st.info("Upload a JPG or PNG PCB image to begin.")
        return

    st.image(
        st.session_state.processed_image_pil,
        caption="Preprocessed image used for analysis",
        use_container_width=True,
    )
    st.caption(f"File: {st.session_state.uploaded_image_name}")


def add_history_event(event_type: str, result: dict[str, Any]) -> None:
    """Record a compact dashboard history item for the current session."""

    history = st.session_state.inspection_history
    entry = {
        "time": datetime.now().strftime("%H:%M:%S"),
        "event": event_type,
        "file": st.session_state.uploaded_image_name or "No file",
        "status": result.get("overall_status")
        or result.get("estimated_repair_difficulty")
        or result.get("status")
        or "Done",
    }
    history.insert(0, entry)
    st.session_state.inspection_history = history[:8]


def count_defect_severities(defects: list[dict[str, Any]]) -> dict[str, int]:
    """Count defect severity values for dashboard display."""

    counts = {"LOW": 0, "MEDIUM": 0, "HIGH": 0, "CRITICAL": 0}
    for defect in defects:
        severity = str(defect.get("severity", "LOW")).upper()
        if severity in counts:
            counts[severity] += 1
    return counts


def render_dashboard_tab(api_key: str, model_name: str, enhance_contrast: bool) -> None:
    """Render a high-level inspection dashboard."""

    render_dashboard_hero(
        "PCB Quality Dashboard",
        "Monitor upload readiness, AI configuration, inspection status, and reconstruction progress from one screen.",
    )

    metadata = st.session_state.upload_metadata or {}
    detection = st.session_state.detection_result or {}
    reconstruction = st.session_state.reconstruction_result or {}
    reconstructed_image = st.session_state.reconstructed_image_bytes
    defects = detection.get("defects_found", []) if detection else []
    severity_counts = count_defect_severities(defects)

    metric_cols = st.columns(4)
    with metric_cols[0]:
        render_dashboard_metric(
            "Image",
            "Ready" if st.session_state.processed_image_bytes else "Waiting",
            metadata.get("name", "Upload a JPG or PNG PCB image."),
        )
    with metric_cols[1]:
        render_dashboard_metric(
            "AI engine",
            "Configured" if api_key else "API key needed",
            MODEL_DISPLAY_NAMES.get(model_name, "Vision Model"),
        )
    with metric_cols[2]:
        quality = f"{detection.get('quality_score', '--')}/100" if detection else "--"
        render_dashboard_metric("Quality", quality, "Available after fault detection.")
    with metric_cols[3]:
        render_dashboard_metric(
            "Reconstructed image",
            "Ready" if reconstructed_image else "Not generated",
            "Generated after the reconstruction guide.",
        )

    left, right = st.columns([1.05, 1.25], gap="large")
    with left:
        st.subheader("Current PCB")
        if st.session_state.processed_image_pil is not None:
            st.image(st.session_state.processed_image_pil, use_container_width=True)
            st.dataframe(
                [
                    {"Property": "File", "Value": metadata.get("name", "-")},
                    {
                        "Property": "Original size",
                        "Value": f"{metadata.get('width', '-')} x {metadata.get('height', '-')}",
                    },
                    {
                        "Property": "Processed size",
                        "Value": f"{metadata.get('processed_width', '-')} x {metadata.get('processed_height', '-')}",
                    },
                    {"Property": "Format", "Value": metadata.get("format", "-")},
                    {"Property": "File size", "Value": metadata.get("size", "-")},
                    {"Property": "CLAHE contrast", "Value": "On" if enhance_contrast else "Off"},
                ],
                hide_index=True,
                use_container_width=True,
            )
        else:
            st.info("Upload a PCB image to populate the dashboard.")

        if reconstructed_image:
            st.subheader("Reconstructed PCB")
            st.image(
                reconstructed_image,
                caption=st.session_state.reconstructed_image_source or "Generated healthy PCB reference",
                use_container_width=True,
            )
            st.download_button(
                "Download Reconstructed Image",
                data=reconstructed_image,
                file_name="reconstructed_healthy_pcb.png",
                mime="image/png",
                use_container_width=True,
            )

    with right:
        st.subheader("Inspection Status")
        status_cols = st.columns(2)
        with status_cols[0]:
            st.caption("Fault detection")
            if detection:
                render_status_badge(str(detection.get("overall_status", "FAIL")))
                st.progress(max(0, min(100, int(detection.get("quality_score", 0)))) / 100)
            else:
                st.info("Not analyzed yet.")
        with status_cols[1]:
            st.caption("Reconstruction")
            if reconstruction:
                render_difficulty_badge(str(reconstruction.get("estimated_repair_difficulty", "MODERATE")))
                st.write(reconstruction.get("board_type_identified", "Board type unavailable."))
            else:
                st.info("No guide generated yet.")

        st.subheader("Defect Severity Breakdown")
        if detection:
            severity_rows = [{"Severity": key, "Count": value} for key, value in severity_counts.items()]
            st.bar_chart(
                {
                    "LOW": [severity_counts["LOW"]],
                    "MEDIUM": [severity_counts["MEDIUM"]],
                    "HIGH": [severity_counts["HIGH"]],
                    "CRITICAL": [severity_counts["CRITICAL"]],
                }
            )
            st.dataframe(severity_rows, hide_index=True, use_container_width=True)
        else:
            st.info("Run fault detection to see severity counts.")

    st.subheader("Session Activity")
    if st.session_state.inspection_history:
        st.dataframe(st.session_state.inspection_history, hide_index=True, use_container_width=True)
    else:
        st.info("Analysis and reconstruction events will appear here during this session.")


def generate_reconstructed_image_action() -> bytes | None:
    """
    Generate reconstructed PCB image using local OpenCV reconstruction only.
    """

    if not st.session_state.processed_image_bytes:
        st.warning("Upload PCB image first.")
        return None

    with st.spinner("Generating reconstructed PCB image..."):

        try:

            reconstructed = create_local_reconstruction_preview(
                st.session_state.processed_image_bytes
            )

            st.session_state.reconstructed_image_source = (
                "AI-enhanced local PCB reconstruction"
            )

            return reconstructed

        except Exception as exc:

            st.error(
                f"Reconstruction failed: {exc}"
            )

            return None


def run_gemini_action(
    action_label: str,
    spinner_text: str,
    api_key: str,
    model_name: str,
    action: Callable[[bytes, Any], dict[str, Any]],
) -> dict[str, Any] | None:
    """Run a Gemini image action with consistent error handling."""

    if st.session_state.processed_image_bytes is None:
        st.warning("Upload a valid PCB image first.")
        return None

    client = get_client_or_show_error(api_key, model_name)
    if client is None:
        return None

    st.session_state.last_error_raw_response = None
    with st.spinner(spinner_text):
        try:
            return action(st.session_state.processed_image_bytes, client)
        except GeminiResponseParseError as exc:
            st.error(f"{action_label} failed because the AI service did not return valid JSON.")
            st.session_state.last_error_raw_response = exc.raw_response
            render_raw_response_debug(exc.raw_response)
        except GoogleAPIError as exc:
            st.error(f"AI service error: {sanitize_visible_error(str(exc))}")
        except ValueError as exc:
            st.error(str(exc))
        except Exception as exc:
            st.error(f"Unexpected error during {action_label.lower()}: {exc}")
    return None


def render_fault_detection_tab(api_key: str, model_name: str) -> None:
    """Render Module 1 UI."""

    left, right = st.columns([1, 1.35], gap="large")
    with left:
        render_uploaded_image_panel()

    with right:
        st.subheader("Fault Detection")
        analyze = st.button(
            "Analyze PCB",
            type="primary",
            disabled=st.session_state.processed_image_bytes is None,
            use_container_width=True,
        )
        if analyze:
            result = run_gemini_action(
                "Fault detection",
                "Inspecting PCB image with the vision model...",
                api_key,
                model_name,
                detect_pcb_faults,
            )
            if result is not None:
                st.session_state.detection_result = result
                add_history_event("Fault detection", result)

        if st.session_state.detection_result:
            result = st.session_state.detection_result
            render_detection_summary(result)
            render_defect_cards(result.get("defects_found", []))
            render_component_analysis_table(result.get("component_analysis", {}))
        else:
            st.info("Run analysis to see PASS/FAIL status, defects, and component-level findings.")


def render_reconstruction_tab(api_key: str, model_name: str) -> None:
    """Render Module 2 UI."""

    left, right = st.columns([1, 1.35], gap="large")
    with left:
        render_uploaded_image_panel()

    with right:
        st.subheader("Healthy PCB Reconstruction")
        generate = st.button(
            "Generate Reconstruction Guide",
            type="primary",
            disabled=st.session_state.processed_image_bytes is None,
            use_container_width=True,
        )
        if generate:
            result = run_gemini_action(
                "Reconstruction guide",
                "Generating healthy board reconstruction guidance with the vision model...",
                api_key,
                model_name,
                generate_reconstruction_guide,
            )
            if result is not None:
                st.session_state.reconstruction_result = result
                add_history_event("Reconstruction guide", result)

        if st.session_state.reconstruction_result:
            result = st.session_state.reconstruction_result
            render_reconstruction_header(result)
            render_reconstruction_steps(result.get("reconstruction_steps", []))
            render_healthy_board_table(result.get("healthy_board_description", {}))
            render_reference_prompt(result.get("reference_image_prompt", ""))

            st.subheader("Reconstructed PCB Image")
            image_cols = st.columns([1, 1])
            with image_cols[0]:
                generate_image = st.button(
                    "Generate Reconstructed Image",
                    type="primary",
                    use_container_width=True,
                )
            with image_cols[1]:
                clear_image = st.button(
                    "Clear Image",
                    disabled=st.session_state.reconstructed_image_bytes is None,
                    use_container_width=True,
                )

            if generate_image:
                image_bytes = generate_reconstructed_image_action(api_key, model_name)
                if image_bytes is not None:
                    st.session_state.reconstructed_image_bytes = image_bytes
                    add_history_event("Reconstructed image", {"status": "Ready"})

            if clear_image:
                st.session_state.reconstructed_image_bytes = None
                st.session_state.reconstructed_image_source = None

            if st.session_state.reconstructed_image_bytes:
                st.image(
                    st.session_state.reconstructed_image_bytes,
                    caption=st.session_state.reconstructed_image_source or "Generated healthy PCB reference image",
                    use_container_width=True,
                )
                st.download_button(
                    "Download Reconstructed Image",
                    data=st.session_state.reconstructed_image_bytes,
                    file_name="reconstructed_healthy_pcb.png",
                    mime="image/png",
                    use_container_width=True,
                )
            else:
                st.info("Generate the reconstructed image after creating the guide.")
        else:
            st.info("Generate a guide to view repair steps and a healthy-board reference prompt.")


def main() -> None:
    """Run the Streamlit application."""

    configure_page()
    initialize_session_state()

    st.title("Generative AI based analysis of fault behaviour in electronic circuit (PCB) and reconstruction Guidance")
    st.caption(
        "Upload a PCB image, inspect likely faults, and generate a practical healthy-board reconstruction guide."
    )

    api_key, model_name, enhance_contrast = render_sidebar()
    handle_upload(enhance_contrast)

    tab_dashboard, tab_detection, tab_reconstruction = st.tabs(
        ["Dashboard", "Fault Detection", "Reconstruction Guide"]
    )
    with tab_dashboard:
        render_dashboard_tab(api_key, model_name, enhance_contrast)
    with tab_detection:
        render_fault_detection_tab(api_key, model_name)
    with tab_reconstruction:
        render_reconstruction_tab(api_key, model_name)


if __name__ == "__main__":
    main()
