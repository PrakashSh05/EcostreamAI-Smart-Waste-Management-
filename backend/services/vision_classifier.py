"""
backend/services/vision_classifier.py
Uses Groq's Llama 4 Scout (multimodal) to accurately classify waste materials
from images. This supplements the YOLO model which has training bias issues.
"""

import base64
import logging
import os

from groq import Groq

logger = logging.getLogger(__name__)

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
VISION_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"

VALID_MATERIALS = ["plastic", "metal", "food_waste", "paper", "glass", "cardboard"]

CLASSIFY_PROMPT = """You are a waste material classifier. Look at this image and identify what waste material(s) are visible.

You MUST respond with ONLY a comma-separated list from these exact categories:
plastic, metal, food_waste, paper, glass, cardboard

Rules:
- If you see a plastic bottle, bag, wrapper, container → "plastic"
- If you see metal cans, tins, foil, utensils, steel bottle → "metal"  
- If you see food scraps, leftovers, fruit peels → "food_waste"
- If you see paper, newspaper, tissues, notebooks → "paper"
- If you see glass bottles, jars, broken glass → "glass"
- If you see cardboard boxes, packaging → "cardboard"
- If you see a pen, pencil → "plastic"
- If you see multiple materials, list all of them separated by commas
- If you cannot identify any waste material, respond with "unknown"
- Do NOT add any explanation, just the material name(s)

Respond with ONLY the material name(s), nothing else."""


async def classify_waste_image(image_bytes: bytes) -> list[str]:
    """
    Send image to Groq Llama 4 Scout for waste material classification.
    Returns a list of detected material names.
    """
    if not GROQ_API_KEY:
        logger.warning("GROQ_API_KEY not set, skipping vision classification")
        return []

    try:
        # Encode image to base64
        b64_image = base64.b64encode(image_bytes).decode("utf-8")

        # Determine image type (assume JPEG for camera captures)
        image_url = f"data:image/jpeg;base64,{b64_image}"

        client = Groq(api_key=GROQ_API_KEY)

        response = client.chat.completions.create(
            model=VISION_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": image_url},
                        },
                        {
                            "type": "text",
                            "text": CLASSIFY_PROMPT,
                        },
                    ],
                }
            ],
            max_tokens=50,
            temperature=0.1,
        )

        raw_answer = response.choices[0].message.content.strip().lower()
        logger.info("Vision classifier raw answer: %r", raw_answer)

        # Parse the response — extract valid material names
        materials = []
        for mat in VALID_MATERIALS:
            if mat in raw_answer:
                materials.append(mat)

        # If no valid materials found, check for common aliases
        if not materials:
            aliases = {
                "steel": "metal", "aluminum": "metal", "aluminium": "metal",
                "tin": "metal", "iron": "metal", "can": "metal",
                "bottle": "plastic", "wrapper": "plastic", "bag": "plastic",
                "newspaper": "paper", "tissue": "paper", "notebook": "paper",
                "food": "food_waste", "fruit": "food_waste", "vegetable": "food_waste",
                "box": "cardboard", "carton": "cardboard",
                "jar": "glass",
            }
            for alias, mat in aliases.items():
                if alias in raw_answer and mat not in materials:
                    materials.append(mat)

        if not materials and "unknown" not in raw_answer:
            # Last resort — return the raw answer if it looks like a material
            logger.warning("Could not parse materials from: %r", raw_answer)

        logger.info("Vision classifier result: %s", materials)
        return materials

    except Exception as e:
        logger.error("Vision classification failed: %s", e)
        return []
