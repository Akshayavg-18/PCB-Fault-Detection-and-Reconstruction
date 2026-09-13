from __future__ import annotations

import base64
import io
from dataclasses import dataclass
from typing import Literal

import cv2
import numpy as np
from PIL import Image, UnidentifiedImageError


MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024
ALLOWED_FORMATS = {"JPEG", "PNG"}
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png"}


@dataclass(frozen=True)
class ImageValidationResult:
    is_valid: bool
    message: str
    format: str | None = None
    size_bytes: int = 0
    width: int | None = None
    height: int | None = None


@dataclass(frozen=True)
class PreprocessedImage:
    image_bytes: bytes
    pil_image: Image.Image
    original_size: tuple[int, int]
    processed_size: tuple[int, int]
    mime_type: Literal["image/jpeg", "image/png"]


def validate_image_bytes(
    image_bytes: bytes,
    filename: str | None = None
) -> ImageValidationResult:

    size = len(image_bytes or b"")

    if size == 0:
        return ImageValidationResult(
            False,
            "The uploaded file is empty."
        )

    if size > MAX_FILE_SIZE_BYTES:
        return ImageValidationResult(
            False,
            "Image exceeds 10MB limit."
        )

    try:

        with Image.open(io.BytesIO(image_bytes)) as image:
            image.verify()

        with Image.open(io.BytesIO(image_bytes)) as image:

            image_format = image.format
            width, height = image.size

    except UnidentifiedImageError:

        return ImageValidationResult(
            False,
            "Invalid image."
        )

    return ImageValidationResult(
        True,
        "Image valid.",
        format=image_format,
        size_bytes=size,
        width=width,
        height=height
    )


def pil_to_cv_rgb(
    image: Image.Image
) -> np.ndarray:

    return np.array(
        image.convert("RGB")
    )


def resize_if_larger(
    image_rgb: np.ndarray,
    max_size: int = 1024
) -> np.ndarray:

    h, w = image_rgb.shape[:2]

    if max(h, w) <= max_size:
        return image_rgb

    scale = max_size / float(max(h, w))

    nw = int(w * scale)
    nh = int(h * scale)

    return cv2.resize(
        image_rgb,
        (nw, nh),
        interpolation=cv2.INTER_AREA
    )


def encode_rgb_image(
    image_rgb: np.ndarray,
    output_format: Literal["JPEG", "PNG"] = "PNG"
) -> bytes:

    pil_image = Image.fromarray(
        image_rgb.astype(np.uint8)
    )

    buffer = io.BytesIO()

    pil_image.save(
        buffer,
        format=output_format
    )

    return buffer.getvalue()


def preprocess_image(
    image_bytes: bytes,
    max_size: int = 1024,
    enhance_contrast: bool = True,
    output_format: Literal["JPEG", "PNG"] = "JPEG",
) -> PreprocessedImage:

    validation = validate_image_bytes(image_bytes)

    if not validation.is_valid:
        raise ValueError(validation.message)

    with Image.open(io.BytesIO(image_bytes)) as image:

        original_size = image.size

        rgb = pil_to_cv_rgb(image)

    rgb = resize_if_larger(
        rgb,
        max_size=max_size
    )

    if enhance_contrast:

        lab = cv2.cvtColor(
            rgb,
            cv2.COLOR_RGB2LAB
        )

        l_channel, a_channel, b_channel = cv2.split(lab)

        clahe = cv2.createCLAHE(
            clipLimit=2.0,
            tileGridSize=(8, 8)
        )

        enhanced_l = clahe.apply(l_channel)

        enhanced_lab = cv2.merge(
            (enhanced_l, a_channel, b_channel)
        )

        rgb = cv2.cvtColor(
            enhanced_lab,
            cv2.COLOR_LAB2RGB
        )

    encoded = encode_rgb_image(
        rgb,
        output_format=output_format
    )

    processed_pil = Image.open(
        io.BytesIO(encoded)
    ).convert("RGB")

    mime_type: Literal["image/jpeg", "image/png"]

    if output_format == "JPEG":
        mime_type = "image/jpeg"
    else:
        mime_type = "image/png"

    return PreprocessedImage(
        image_bytes=encoded,
        pil_image=processed_pil,
        original_size=original_size,
        processed_size=processed_pil.size,
        mime_type=mime_type,
    )


def reconstruct_pcb_traces(
    image_rgb: np.ndarray
) -> np.ndarray:
    """
    Hybrid PCB reconstruction using:
    - burn detection
    - patch replacement
    - morphology repair
    """

    reconstructed = image_rgb.copy()

    hsv = cv2.cvtColor(
        reconstructed,
        cv2.COLOR_RGB2HSV
    )

    value = hsv[:, :, 2]

    # Detect burnt/dark regions
    burn_mask = cv2.inRange(
        value,
        0,
        70
    )

    kernel = np.ones((5,5), np.uint8)

    burn_mask = cv2.morphologyEx(
        burn_mask,
        cv2.MORPH_CLOSE,
        kernel,
        iterations=2
    )

    contours, _ = cv2.findContours(
        burn_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    for contour in contours:

        area = cv2.contourArea(contour)

        if area < 80:
            continue

        x, y, w, h = cv2.boundingRect(contour)

        pad = 25

        x0 = max(0, x - pad)
        y0 = max(0, y - pad)

        x1 = min(
            reconstructed.shape[1],
            x + w + pad
        )

        y1 = min(
            reconstructed.shape[0],
            y + h + pad
        )

        region_w = x1 - x0
        region_h = y1 - y0

        # Select nearby healthy region
        source_x = max(0, x0 - region_w)

        source_patch = reconstructed[
            y0:y0+region_h,
            source_x:source_x+region_w
        ]

        if source_patch.size == 0:
            continue

        source_patch = cv2.resize(
            source_patch,
            (region_w, region_h)
        )

        # Replace burnt region
        reconstructed[
            y0:y1,
            x0:x1
        ] = source_patch

    # Trace reconstruction
    gray = cv2.cvtColor(
        reconstructed,
        cv2.COLOR_RGB2GRAY
    )

    _, thresh = cv2.threshold(
        gray,
        120,
        255,
        cv2.THRESH_BINARY_INV
    )

    repaired = cv2.morphologyEx(
        thresh,
        cv2.MORPH_CLOSE,
        np.ones((3,3), np.uint8),
        iterations=1
    )

    repaired = cv2.dilate(
        repaired,
        np.ones((2,2), np.uint8),
        iterations=1
    )

    repaired_rgb = cv2.cvtColor(
        repaired,
        cv2.COLOR_GRAY2RGB
    )

    repaired_rgb = cv2.bitwise_not(
        repaired_rgb
    )

    # Blend reconstructed traces
    final = cv2.addWeighted(
        reconstructed,
        0.75,
        repaired_rgb,
        0.25,
        0
    )

    return final

def create_local_reconstruction_preview(
    image_bytes: bytes
) -> bytes:

    processed = preprocess_image(
        image_bytes
    )

    rgb = np.array(
        processed.pil_image
    )

    reconstructed = reconstruct_pcb_traces(
        rgb
    )

    return encode_rgb_image(
        reconstructed,
        output_format="PNG"
    )


def image_bytes_to_base64(
    image_bytes: bytes
) -> str:

    return base64.b64encode(
        image_bytes
    ).decode("ascii")


def human_readable_size(
    size_bytes: int
) -> str:

    if size_bytes < 1024:
        return f"{size_bytes} B"

    if size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"

    return f"{size_bytes / (1024 * 1024):.2f} MB"