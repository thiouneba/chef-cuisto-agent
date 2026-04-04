"""
Vision Module — Ingredient detection from images using GPT-4o vision.
Sends the image directly to the LLM with a structured extraction prompt.
"""

import logging
import json
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

logger = logging.getLogger(__name__)

VISION_PROMPT = """You are a culinary vision expert. 
Analyze this image carefully and identify ALL food ingredients visible.

Return a JSON object with this exact structure:
{
  "ingredients": ["ingredient1", "ingredient2", ...],
  "ingredients_text": "A natural comma-separated sentence listing all ingredients",
  "context": "Brief note about the image (e.g. fridge contents, market bag, kitchen counter...)",
  "confidence": "high/medium/low",
  "notes": "Any relevant observation (e.g. some items hard to identify, partially visible...)"
}

Be thorough — identify spices, condiments, vegetables, proteins, dairy, grains, everything visible.
Return ONLY the JSON, no extra text.
"""


def analyze_image_ingredients(
    image_base64: str,
    media_type: str = "image/jpeg",
    llm: ChatOpenAI = None,
) -> dict:
    """
    Analyze a base64-encoded image and extract all visible ingredients.
    
    Args:
        image_base64: Base64-encoded image string
        media_type: MIME type (image/jpeg, image/png, image/webp)
        llm: ChatOpenAI instance (reuses the agent's LLM)
    
    Returns:
        dict with 'ingredients' (list), 'ingredients_text' (str), 'context' (str)
    """
    if llm is None:
        from config import settings
        llm = ChatOpenAI(
            model="gpt-4o",  # Vision requires gpt-4o
            openai_api_key=settings.OPENAI_API_KEY,
        )

    message = HumanMessage(content=[
        {
            "type": "image_url",
            "image_url": {
                "url": f"data:{media_type};base64,{image_base64}",
                "detail": "high",
            },
        },
        {
            "type": "text",
            "text": VISION_PROMPT,
        },
    ])

    try:
        response = llm.invoke([message])
        raw = response.content.strip()

        # Strip markdown code fences if present
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]

        data = json.loads(raw)
        logger.info(f"Vision detected {len(data.get('ingredients', []))} ingredients")
        return data

    except json.JSONDecodeError as e:
        logger.error(f"Vision response not valid JSON: {e}")
        # Graceful fallback
        return {
            "ingredients": [],
            "ingredients_text": "Could not parse ingredients from image.",
            "context": "Image analysis failed",
            "confidence": "low",
            "notes": str(e),
        }
    except Exception as e:
        logger.error(f"Vision analysis error: {e}")
        raise
