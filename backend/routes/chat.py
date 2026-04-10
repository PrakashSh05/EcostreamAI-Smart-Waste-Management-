"""
backend/routes/chat.py
Chat endpoint for the citizen chatbot.
Uses RAG pipeline to answer waste disposal questions.
Chat is NOT saved to the database.
"""

import logging
import time

from fastapi import APIRouter, File, Form, UploadFile

router = APIRouter(tags=["chat"])
logger = logging.getLogger(__name__)

# Known materials in the YOLO model / waste domain
_KNOWN_MATERIALS = [
    "plastic bottles", "plastic bags", "plastic waste", "plastic",
    "metal", "food waste", "food", "paper", "glass", "cardboard",
    "e-waste", "hazardous waste", "construction debris",
    "sanitary waste", "wet waste", "dry waste", "mixed household waste",
    "solid waste", "used cooking oil", "old newspapers",
    "tetra pak cartons", "vegetable peels", "vending-site waste",
    "kitchen waste",
]

# Known cities
_KNOWN_CITIES = {
    "bangalore": "Bangalore", "bengaluru": "Bangalore",
    "mumbai": "Mumbai", "delhi": "Delhi",
    "chennai": "Chennai", "india": "India",
}


def _extract_materials(text: str) -> list[str]:
    """Extract known material names from user's message."""
    lower = text.lower()
    found = []
    sorted_materials = sorted(_KNOWN_MATERIALS, key=len, reverse=True)
    for mat in sorted_materials:
        if mat in lower:
            found.append(mat)
            lower = lower.replace(mat, "", 1)
    return found


def _extract_city(text: str, default: str = "Bangalore") -> str:
    """Extract city name from user's message."""
    lower = text.lower()
    for key, name in _KNOWN_CITIES.items():
        if key in lower:
            return name
    return default


def _chat_with_rag(question: str, materials: list[str], city: str) -> str:
    """
    Query the RAG chain directly with the user's actual question.
    Falls back to a helpful answer if the RAG chain can't answer.
    """
    try:
        from rag.query import _get_rag_chain, _get_city_aware_docs, _clean_context_chunk

        _get_rag_chain()

        # Import the chain and retriever
        from rag.query import _rag_chain

        material_text = ", ".join(materials) if materials else "waste"

        # Retrieve relevant documents
        docs = _get_city_aware_docs(question, city, material_text)
        context_chunks = [_clean_context_chunk(d.page_content) for d in docs]

        logger.info(
            "RAG retrieved %d docs, context length: %d chars",
            len(docs),
            sum(len(c) for c in context_chunks),
        )

        # Invoke RAG chain with the user's ACTUAL question
        answer = _rag_chain.invoke({
            "question": question,
            "city": city,
            "material_text": material_text,
        })

        # If the answer is useful, return it
        if answer and "I don't know" not in answer:
            return answer.strip()

        # Fallback: try with a reformulated question
        reformulated = f"How should I dispose of {material_text} in {city}?"
        answer2 = _rag_chain.invoke({
            "question": reformulated,
            "city": city,
            "material_text": material_text,
        })

        if answer2 and "I don't know" not in answer2:
            return answer2.strip()

    except Exception as e:
        logger.error("RAG chain error: %s", e)

    return ""  # empty = will use fallback


# City-specific disposal tips for common materials
_FALLBACK_TIPS = {
    "plastic": "Rinse and dry plastic items, then place them in the blue/dry waste bin. Do not burn plastic waste. Hand over to authorised waste collectors for recycling.",
    "plastic bottles": "Clean the plastic bottles, remove caps, and place them in the dry waste/blue bin. Ensure they are empty and dry for effective recycling.",
    "metal": "Metal waste like cans and tins should be rinsed, dried, and placed in the dry waste bin. They are highly recyclable — hand over to authorised scrap dealers.",
    "food waste": "Food waste is bio-degradable. Place it in the green/wet waste bin. Do not mix with dry or hazardous waste. Consider composting at home.",
    "food": "Food waste should be placed in the green/wet waste bin for composting. Separate from plastic packaging before disposal.",
    "paper": "Paper waste should be kept dry and placed in the blue/dry waste bin. Avoid mixing with wet waste. Old newspapers can be given to authorised waste collectors.",
    "glass": "Glass items should be carefully placed in the dry waste bin. Wrap broken glass in newspaper to prevent injury to waste handlers.",
    "cardboard": "Flatten cardboard boxes and place them in the dry waste bin. Keep dry for effective recycling. Remove any plastic tape or labels.",
    "e-waste": "E-waste should not be mixed with regular waste. Hand over to authorized e-waste recyclers as per the E-Waste Management Rules, 2016.",
    "hazardous waste": "Hazardous waste must be kept separate from regular waste and disposed through authorized hazardous waste facilities as per government rules.",
    "wet waste": "Wet waste includes kitchen waste, food scraps, and organic matter. Place in the green bin. Consider home composting for vegetable peels and tea leaves.",
    "dry waste": "Dry waste includes paper, plastic, glass, and metal. Segregate and place in the blue bin. Ensure items are clean and dry for recycling.",
    "waste": "Segregate your waste into three categories: wet waste (green bin), dry waste (blue bin), and reject/hazardous waste (red bin). Hand over segregated waste to authorised collectors.",
}


@router.post("/chat")
async def chat(
    message: str = Form(...),
    city: str = Form("Bangalore"),
    file: UploadFile | None = File(None),
):
    """Answer a waste disposal question using the RAG pipeline."""
    detected_materials = []
    t0 = time.time()

    # If image is attached, run YOLO detection
    if file and file.filename:
        try:
            from backend.services import yolo_client
            image_bytes = await file.read()
            yolo_result = await yolo_client.detect_materials(image_bytes)
            if isinstance(yolo_result, dict):
                detected_materials = yolo_result.get("labels", [])
            else:
                detected_materials = yolo_result or []
            detected_materials = [str(m) for m in detected_materials]
            logger.info("Chat image detection: %s", detected_materials)
        except Exception as e:
            logger.warning("Chat image detection failed: %s", e)

    # Determine materials and city
    if detected_materials:
        materials = detected_materials
    else:
        materials = _extract_materials(message)

    resolved_city = _extract_city(message, default=city)

    logger.info(
        "Chat query: message=%r, materials=%s, city=%s",
        message[:100], materials, resolved_city,
    )

    # Try RAG first
    answer = _chat_with_rag(message, materials, resolved_city)

    # Fallback with curated tips if RAG didn't produce a good answer
    if not answer:
        mat_key = materials[0] if materials else "waste"
        tip = _FALLBACK_TIPS.get(mat_key, _FALLBACK_TIPS["waste"])
        answer = f"In {resolved_city}: {tip}"

    elapsed_ms = int((time.time() - t0) * 1000)

    return {
        "answer": answer,
        "detected_materials": detected_materials,
        "materials_used": materials,
        "city": resolved_city,
        "response_ms": elapsed_ms,
    }
