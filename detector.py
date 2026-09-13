"""PCB fault detection logic powered by Gemini Vision."""

from __future__ import annotations

from typing import Any

from gemini_client import GeminiClient


DETECTION_PROMPT = """
You are an expert PCB (Printed Circuit Board) quality control inspector with 20+ years of experience.

Analyze the provided PCB image and return your response as a valid JSON object with exactly this structure:

{
  "overall_status": "PASS" or "FAIL",
  "confidence_score": <float 0.0 to 1.0>,
  "defects_found": [
    {
      "defect_type": "<type>",
      "severity": "LOW" | "MEDIUM" | "HIGH" | "CRITICAL",
      "location": "<describe location on board>",
      "description": "<what the defect looks like>",
      "recommended_action": "<fix suggestion>"
    }
  ],
  "component_analysis": {
    "solder_joints": "<assessment>",
    "traces": "<assessment>",
    "components": "<assessment>",
    "substrate": "<assessment>"
  },
  "quality_score": <integer 0 to 100>,
  "summary": "<2-3 sentence overall summary>"
}

Common defect types to look for: solder bridge, cold solder joint, missing component, lifted pad, 
PCB crack, burnt component, misaligned component, excess solder, insufficient solder, 
oxidation, delamination, open circuit trace, short circuit.

Return ONLY valid JSON. No markdown, no explanation outside the JSON.
""".strip()


REQUIRED_TOP_LEVEL_KEYS = {
    "overall_status",
    "confidence_score",
    "defects_found",
    "component_analysis",
    "quality_score",
    "summary",
}


def detect_pcb_faults(image_bytes: bytes, client: GeminiClient) -> dict[str, Any]:
    """Analyze a PCB image and return normalized fault-detection JSON."""

    raw_result = client.send_image_prompt(image_bytes=image_bytes, system_prompt=DETECTION_PROMPT)
    return normalize_detection_result(raw_result)


def normalize_detection_result(result: dict[str, Any]) -> dict[str, Any]:
    """Fill harmless defaults and clamp scores so UI rendering is resilient."""

    normalized: dict[str, Any] = dict(result)
    missing = REQUIRED_TOP_LEVEL_KEYS - normalized.keys()
    if missing:
        raise ValueError(f"Detection response is missing required keys: {', '.join(sorted(missing))}")

    status = str(normalized.get("overall_status", "FAIL")).upper()
    normalized["overall_status"] = "PASS" if status == "PASS" else "FAIL"
    normalized["confidence_score"] = clamp_float(normalized.get("confidence_score"), 0.0, 1.0, default=0.0)
    normalized["quality_score"] = int(clamp_float(normalized.get("quality_score"), 0, 100, default=0))

    defects = normalized.get("defects_found")
    if not isinstance(defects, list):
        normalized["defects_found"] = []
    else:
        normalized["defects_found"] = [normalize_defect(item) for item in defects if isinstance(item, dict)]

    component_analysis = normalized.get("component_analysis")
    if not isinstance(component_analysis, dict):
        component_analysis = {}
    normalized["component_analysis"] = {
        "solder_joints": str(component_analysis.get("solder_joints", "Not assessed.")),
        "traces": str(component_analysis.get("traces", "Not assessed.")),
        "components": str(component_analysis.get("components", "Not assessed.")),
        "substrate": str(component_analysis.get("substrate", "Not assessed.")),
    }

    normalized["summary"] = str(normalized.get("summary", "No summary provided."))
    return normalized


def normalize_defect(defect: dict[str, Any]) -> dict[str, str]:
    """Normalize a single defect object for display."""

    severity = str(defect.get("severity", "LOW")).upper()
    if severity not in {"LOW", "MEDIUM", "HIGH", "CRITICAL"}:
        severity = "LOW"

    return {
        "defect_type": str(defect.get("defect_type", "Unspecified defect")),
        "severity": severity,
        "location": str(defect.get("location", "Location not specified")),
        "description": str(defect.get("description", "No description provided.")),
        "recommended_action": str(defect.get("recommended_action", "Inspect and repair as needed.")),
    }


def clamp_float(value: Any, minimum: float, maximum: float, default: float) -> float:
    """Convert to float and clamp to an inclusive range."""

    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return max(minimum, min(maximum, number))
