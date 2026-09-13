"""Gemini API client utilities for PCB image analysis."""

from __future__ import annotations

import io
import json
import os
import re
from dataclasses import dataclass
from typing import Any

import google.generativeai as genai
from google.api_core.exceptions import GoogleAPIError
from google import genai as image_genai
from google.genai import errors as image_errors
from google.genai import types
from PIL import Image, UnidentifiedImageError


DEFAULT_MODEL_NAME = "gemini-2.5-flash"
IMAGE_MODEL_NAME = "gemini-2.5-flash-image"
SUPPORTED_MODELS = (
    "gemini-2.5-flash",
    "gemini-2.5-pro",
    "gemini-2.0-flash",
    "gemini-2.0-flash-lite",
    "gemini-flash-latest",
    "gemini-pro-latest",
    "gemini-1.5-flash",
    "gemini-1.5-pro",
)


class GeminiConfigurationError(RuntimeError):
    """Raised when Gemini cannot be configured for an API call."""


class GeminiResponseParseError(ValueError):
    """Raised when Gemini returns text that cannot be parsed as JSON."""

    def __init__(self, message: str, raw_response: str) -> None:
        super().__init__(message)
        self.raw_response = raw_response


class GeminiImageGenerationError(RuntimeError):
    """Raised when the image model does not return image bytes."""


@dataclass(frozen=True)
class GeminiClient:
    """Thin wrapper around the Google Generative AI multimodal API."""

    api_key: str
    model_name: str = DEFAULT_MODEL_NAME

    def __post_init__(self) -> None:
        if not self.api_key or not self.api_key.strip():
            raise GeminiConfigurationError(
                "An API key is required. Enter one in the sidebar or set the local API key environment variable."
            )
        if self.model_name not in SUPPORTED_MODELS:
            raise GeminiConfigurationError(
                f"Unsupported model '{self.model_name}'. Choose one of: {', '.join(SUPPORTED_MODELS)}."
            )

        genai.configure(api_key=self.api_key.strip())

    @property
    def model(self) -> genai.GenerativeModel:
        return genai.GenerativeModel(self.model_name)

    def send_image_prompt(self, image_bytes: bytes, system_prompt: str) -> dict[str, Any]:
        """Send an image and prompt to Gemini and return parsed JSON."""

        if not image_bytes:
            raise ValueError("Image bytes are empty.")
        if not system_prompt or not system_prompt.strip():
            raise ValueError("System prompt cannot be empty.")

        try:
            image = Image.open(io.BytesIO(image_bytes))
            image.load()
        except UnidentifiedImageError as exc:
            raise ValueError("Uploaded file is not a valid image.") from exc

        try:
            response = self.model.generate_content([system_prompt, image])
            raw_text = getattr(response, "text", "") or ""
        except GoogleAPIError:
            raise
        except Exception as exc:
            raise RuntimeError(f"Gemini request failed: {exc}") from exc

        return parse_json_response(raw_text)

    def generate_reconstructed_image(
        self,
        reference_prompt: str,
        image_model_name: str = IMAGE_MODEL_NAME,
    ) -> bytes:
        """Generate a reconstructed healthy PCB image from a text prompt."""

        if not reference_prompt or not reference_prompt.strip():
            raise ValueError("A reconstruction image prompt is required.")

        client = image_genai.Client(api_key=self.api_key.strip())
        prompt = build_reconstruction_image_prompt(reference_prompt)
        try:
            response = client.models.generate_content(
                model=image_model_name,
                contents=prompt,
                config=types.GenerateContentConfig(response_modalities=["TEXT", "IMAGE"]),
            )
        except image_errors.APIError as exc:
            message = str(exc)
            if "RESOURCE_EXHAUSTED" in message or "429" in message or "quota" in message.lower():
                raise GeminiImageGenerationError(
                    "Image reconstruction is currently unavailable because the image-generation quota for this API key is exhausted. Try again later or use an API key with image-generation quota."
                ) from exc
            raise GeminiImageGenerationError(f"Image reconstruction request failed: {exc}") from exc

        image_bytes = extract_first_inline_image(response)
        if not image_bytes:
            raise GeminiImageGenerationError(
                "The AI image model returned no image. Try making the reconstruction prompt more visual and specific."
            )
        return normalize_generated_image_bytes(image_bytes)


def create_client(api_key: str | None = None, model_name: str = DEFAULT_MODEL_NAME) -> GeminiClient:
    """Create a GeminiClient from an explicit key or GEMINI_API_KEY."""

    resolved_key = api_key or os.getenv("GEMINI_API_KEY", "")
    return GeminiClient(api_key=resolved_key, model_name=model_name)


def strip_markdown_code_fence(raw_text: str) -> str:
    """Remove common Markdown code fences from a model response."""

    text = (raw_text or "").strip()
    if not text.startswith("```"):
        return text

    match = re.match(r"^```(?:json|JSON)?\s*(.*?)\s*```$", text, flags=re.DOTALL)
    if match:
        return match.group(1).strip()

    parts = text.split("```")
    if len(parts) >= 2:
        candidate = parts[1].strip()
        if candidate.lower().startswith("json"):
            candidate = candidate[4:].strip()
        return candidate
    return text


def extract_json_candidate(raw_text: str) -> str:
    """Return the most likely JSON object from a raw text response."""

    text = strip_markdown_code_fence(raw_text)
    if text.startswith("{") and text.endswith("}"):
        return text

    first = text.find("{")
    last = text.rfind("}")
    if first != -1 and last != -1 and last > first:
        return text[first : last + 1].strip()
    return text


def parse_json_response(raw_text: str) -> dict[str, Any]:
    """Parse Gemini JSON, preserving raw text in errors for Streamlit debugging."""

    candidate = extract_json_candidate(raw_text)
    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError as exc:
        raise GeminiResponseParseError(
            f"Gemini returned invalid JSON: {exc.msg} at line {exc.lineno}, column {exc.colno}.",
            raw_response=raw_text,
        ) from exc

    if not isinstance(parsed, dict):
        raise GeminiResponseParseError(
            "Gemini returned JSON, but the root value was not an object.",
            raw_response=raw_text,
        )
    return parsed


def build_reconstruction_image_prompt(reference_prompt: str) -> str:
    """Create a focused prompt for producing a healthy PCB reference image."""

    return (
        "Generate a clean, realistic top-down technical reference image of the fully repaired, "
        "healthy PCB described below. Preserve the same board type, approximate layout, component "
        "density, copper traces, solder pads, component orientation, and board color. Remove all "
        "visible defects such as burn marks, solder bridges, corrosion, cracks, lifted pads, missing "
        "parts, and trace damage. The output should look like a defect-free PCB inspection reference, "
        "not a diagram, not a 3D render, and not an artistic illustration.\n\n"
        f"Healthy PCB description:\n{reference_prompt.strip()}"
    )


def extract_first_inline_image(response: Any) -> bytes | None:
    """Extract the first inline image payload from a google-genai response."""

    candidates = getattr(response, "candidates", None) or []
    for candidate in candidates:
        content = getattr(candidate, "content", None)
        parts = getattr(content, "parts", None) or []
        for part in parts:
            inline_data = getattr(part, "inline_data", None)
            if inline_data is None:
                continue
            data = getattr(inline_data, "data", None)
            if isinstance(data, bytes):
                return data
            if isinstance(data, str):
                import base64

                return base64.b64decode(data)

    parts = getattr(response, "parts", None) or []
    for part in parts:
        inline_data = getattr(part, "inline_data", None)
        if inline_data is None:
            continue
        data = getattr(inline_data, "data", None)
        if isinstance(data, bytes):
            return data
        if isinstance(data, str):
            import base64

            return base64.b64decode(data)
    return None


def normalize_generated_image_bytes(image_bytes: bytes) -> bytes:
    """Validate generated image bytes and normalize them to PNG."""

    buffer = io.BytesIO(image_bytes)
    try:
        image = Image.open(buffer)
        image.load()
    except UnidentifiedImageError as exc:
        raise GeminiImageGenerationError("The AI service returned data that was not a valid image.") from exc

    output = io.BytesIO()
    image.convert("RGB").save(output, format="PNG", optimize=True)
    return output.getvalue()


def send_image_prompt(image_bytes: bytes, system_prompt: str) -> dict[str, Any]:
    """Compatibility function matching the requested usage pattern."""

    client = create_client()
    return client.send_image_prompt(image_bytes, system_prompt)
