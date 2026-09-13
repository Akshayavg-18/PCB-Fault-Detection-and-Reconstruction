import cv2
import numpy as np
from PIL import Image

import torch

from diffusers import StableDiffusionInpaintPipeline


pipe = StableDiffusionInpaintPipeline.from_pretrained(
    "runwayml/stable-diffusion-inpainting",
    torch_dtype=torch.float32
)

pipe = pipe.to("cpu")


def reconstruct_pcb_ai(image_path):

    image = Image.open(image_path).convert("RGB")

    image_np = np.array(image)

    hsv = cv2.cvtColor(
        image_np,
        cv2.COLOR_RGB2HSV
    )

    value = hsv[:, :, 2]

    # Detect dark burnt regions
    mask = cv2.inRange(
        value,
        0,
        70
    )

    kernel = np.ones((5,5), np.uint8)

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        kernel,
        iterations=2
    )

    mask_pil = Image.fromarray(mask)

    prompt = (
        "healthy green PCB circuit board "
        "with clean copper traces "
        "high quality electronics board"
    )

    result = pipe(
        prompt=prompt,
        image=image,
        mask_image=mask_pil
    ).images[0]

    return result