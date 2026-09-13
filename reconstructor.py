"""Healthy PCB reconstruction guide generation powered by Gemini Vision."""

from __future__ import annotations

from typing import Any

from gemini_client import GeminiClient


RECONSTRUCTION_PROMPT = """
You are an expert PCB restoration engineer and technical writer.

Given a faulty PCB image, your task is to describe in detail what the healthy, defect-free version of this PCB should look like, as if you are guiding a reconstruction process.

Return a JSON object with this exact structure:

{
  "board_type_identified": "<type of PCB, e.g. single-layer, double-layer, Arduino clone, etc.>",
  "reconstruction_steps": [
    {
      "step_number": <int>,
      "area": "<which area of the board>",
      "action": "<what to do>",
      "expected_result": "<what it should look like after>",
      "tools_needed": "<list of tools or materials>"
    }
  ],
  "healthy_board_description": {
    "solder_joints": "<what perfect solder joints look like on this board>",
    "traces": "<what healthy traces look like>",
    "components": "<correct component placement and orientation>",
    "surface": "<substrate and surface finish condition>"
  },
  "reference_image_prompt": "<A detailed text-to-image prompt describing what the fully healthy version of this PCB should look like, suitable for an image generation model>",
  "estimated_repair_difficulty": "EASY" | "MODERATE" | "HARD" | "EXPERT",
  "summary": "<3-4 sentence reconstruction summary>"
}

Return ONLY valid JSON. No markdown, no text outside the JSON.
""".strip()


REQUIRED_TOP_LEVEL_KEYS = {
    "board_type_identified",
    "reconstruction_steps",
    "healthy_board_description",
    "reference_image_prompt",
    "estimated_repair_difficulty",
    "summary",
}


def generate_reconstruction_guide(image_bytes: bytes, client: GeminiClient) -> dict[str, Any]:
    """Generate a healthy-PCB reconstruction guide from a faulty PCB image."""

    raw_result = client.send_image_prompt(image_bytes=image_bytes, system_prompt=RECONSTRUCTION_PROMPT)
    return normalize_reconstruction_result(raw_result)


def normalize_reconstruction_result(result: dict[str, Any]) -> dict[str, Any]:
    """Normalize Gemini reconstruction JSON for predictable Streamlit rendering."""

    normalized: dict[str, Any] = dict(result)
    missing = REQUIRED_TOP_LEVEL_KEYS - normalized.keys()
    if missing:
        raise ValueError(f"Reconstruction response is missing required keys: {', '.join(sorted(missing))}")

    normalized["board_type_identified"] = str(
        normalized.get("board_type_identified", "Unknown PCB type")
    )

    difficulty = str(normalized.get("estimated_repair_difficulty", "MODERATE")).upper()
    if difficulty not in {"EASY", "MODERATE", "HARD", "EXPERT"}:
        difficulty = "MODERATE"
    normalized["estimated_repair_difficulty"] = difficulty

    steps = normalized.get("reconstruction_steps")
    if not isinstance(steps, list):
        normalized["reconstruction_steps"] = []
    else:
        normalized["reconstruction_steps"] = [
            normalize_reconstruction_step(index + 1, step)
            for index, step in enumerate(steps)
            if isinstance(step, dict)
        ]

    description = normalized.get("healthy_board_description")
    if not isinstance(description, dict):
        description = {}
    normalized["healthy_board_description"] = {
        "solder_joints": str(description.get("solder_joints", "Not described.")),
        "traces": str(description.get("traces", "Not described.")),
        "components": str(description.get("components", "Not described.")),
        "surface": str(description.get("surface", "Not described.")),
    }

    normalized["reference_image_prompt"] = str(
        normalized.get("reference_image_prompt", "No reference prompt provided.")
    )
    normalized["summary"] = str(normalized.get("summary", "No summary provided."))
    return normalized


def normalize_reconstruction_step(fallback_number: int, step: dict[str, Any]) -> dict[str, Any]:
    """Normalize a single reconstruction step."""

    try:
        step_number = int(step.get("step_number", fallback_number))
    except (TypeError, ValueError):
        step_number = fallback_number

    return {
        "step_number": step_number,
        "area": str(step.get("area", "Unspecified area")),
        "action": str(step.get("action", "No action specified.")),
        "expected_result": str(step.get("expected_result", "Expected result not specified.")),
        "tools_needed": str(step.get("tools_needed", "Standard PCB repair tools")),
    }
