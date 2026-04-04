"""
API Layer — FastAPI REST interface for Chef Cuisto Agent.

Endpoints:
    POST /recipe/from-text          → Generate recipe from ingredient list (text)
    POST /recipe/from-image         → Generate recipe from ingredient photo
    POST /session/{id}/constraints  → Set dietary constraints
    GET  /session/{id}              → Get session info & constraints
    DELETE /session/{id}/history    → Clear conversation history
    DELETE /session/{id}            → Full reset (history + constraints)
    GET  /constraints/available     → List all supported dietary constraints
    GET  /health                    → Health check
"""

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional
import base64
import logging

from agent.chef_agent import ChefAgent
from agent.memory import KNOWN_CONSTRAINTS
from config import settings

logging.basicConfig(
    level=settings.LOG_LEVEL,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

# ── App setup ────────────────────────────────────────
app = FastAPI(
    title="🍳 Chef Cuisto Agent",
    description="Autonomous AI Chef — give me your ingredients, I'll give you a recipe.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Session registry ──────────────────────────────────
_sessions: dict[str, ChefAgent] = {}

def get_or_create_session(session_id: str) -> ChefAgent:
    if session_id not in _sessions:
        _sessions[session_id] = ChefAgent(
            session_id=session_id,
            verbose=settings.DEBUG,
        )
    return _sessions[session_id]


# ── Schemas ───────────────────────────────────────────

class TextRecipeRequest(BaseModel):
    ingredients: str = Field(
        ...,
        min_length=3,
        example="2 chicken breasts, 1 zucchini, 3 tomatoes, garlic, olive oil, parmesan",
        description="Comma-separated or natural language list of available ingredients"
    )
    session_id: str = Field(default="default", example="user_42")

class RecipeResponse(BaseModel):
    recipe: str
    tools_used: list[dict]
    session_id: str
    detected_ingredients: Optional[list[str]] = None

class ConstraintsRequest(BaseModel):
    constraints: list[str] = Field(
        ...,
        example=["vegetarian", "gluten-free"],
        description="List of dietary constraints to apply"
    )

class PreferenceRequest(BaseModel):
    liked: list[str] = Field(default=[], example=["garlic", "lemon"])
    disliked: list[str] = Field(default=[], example=["cilantro", "mushrooms"])

class SessionInfoResponse(BaseModel):
    session_id: str
    constraints: list[str]
    liked: list[str]
    disliked: list[str]
    history_length: int


# ── Recipe Routes ─────────────────────────────────────

@app.post("/recipe/from-text", response_model=RecipeResponse, tags=["Recipe"])
async def recipe_from_text(request: TextRecipeRequest):
    """
    Generate a complete recipe from a text list of ingredients.
    
    Provide any ingredients you have — the Chef Agent will propose
    the best recipe respecting your dietary constraints.
    """
    agent = get_or_create_session(request.session_id)
    try:
        result = agent.cook_from_text(request.ingredients)
        return RecipeResponse(**result)
    except Exception as e:
        logger.error(f"Recipe from text error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/recipe/from-image", response_model=RecipeResponse, tags=["Recipe"])
async def recipe_from_image(
    session_id: str = "default",
    file: UploadFile = File(..., description="Photo of your ingredients (JPG, PNG, WEBP)"),
):
    """
    Upload a photo of your ingredients — the Agent will:
    1. Analyze the image with GPT-4o vision
    2. Extract all visible ingredients
    3. Generate a complete recipe respecting your constraints
    
    Accepts: JPG, PNG, WEBP
    """
    allowed_types = {
        "image/jpeg": "image/jpeg",
        "image/jpg": "image/jpeg",
        "image/png": "image/png",
        "image/webp": "image/webp",
    }

    content_type = file.content_type.lower()
    if content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported image type '{content_type}'. Use JPG, PNG or WEBP."
        )

    image_bytes = await file.read()
    if len(image_bytes) > 10 * 1024 * 1024:  # 10MB limit
        raise HTTPException(status_code=400, detail="Image too large. Max 10MB.")

    image_base64 = base64.b64encode(image_bytes).decode("utf-8")
    media_type = allowed_types[content_type]

    agent = get_or_create_session(session_id)
    try:
        result = agent.cook_from_image(image_base64=image_base64, media_type=media_type)
        return RecipeResponse(**result)
    except Exception as e:
        logger.error(f"Recipe from image error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ── Session / Constraints Routes ──────────────────────

@app.post("/session/{session_id}/constraints", tags=["Session"])
async def set_constraints(session_id: str, request: ConstraintsRequest):
    """
    Set dietary constraints for a session.
    These constraints will be respected for ALL future recipe requests.
    
    Supported: vegetarian, vegan, gluten-free, dairy-free, nut-free,
               halal, kosher, low-carb, keto, paleo, no-pork, no-seafood,
               no-eggs, low-sodium, diabetic-friendly
    """
    agent = get_or_create_session(session_id)
    agent.set_constraints(request.constraints)
    return {
        "status": "updated",
        "session_id": session_id,
        "constraints": agent.get_constraints(),
    }


@app.post("/session/{session_id}/preferences", tags=["Session"])
async def set_preferences(session_id: str, request: PreferenceRequest):
    """Set liked and disliked ingredients for a session."""
    agent = get_or_create_session(session_id)
    for ing in request.liked:
        agent.memory.like_ingredient(ing)
    for ing in request.disliked:
        agent.memory.dislike_ingredient(ing)
    return {
        "status": "updated",
        "preferences": agent.memory.get_preferences(),
    }


@app.get("/session/{session_id}", response_model=SessionInfoResponse, tags=["Session"])
async def get_session(session_id: str):
    """Get current session info: constraints, preferences, history length."""
    if session_id not in _sessions:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found.")
    agent = _sessions[session_id]
    return SessionInfoResponse(**agent.memory.get_summary())


@app.delete("/session/{session_id}/history", tags=["Session"])
async def clear_history(session_id: str):
    """Clear conversation history but keep dietary constraints."""
    if session_id not in _sessions:
        raise HTTPException(status_code=404, detail="Session not found.")
    _sessions[session_id].reset()
    return {"status": "history_cleared", "session_id": session_id}


@app.delete("/session/{session_id}", tags=["Session"])
async def full_reset(session_id: str):
    """Full reset: clear history AND constraints for a session."""
    if session_id not in _sessions:
        raise HTTPException(status_code=404, detail="Session not found.")
    _sessions[session_id].memory.full_reset()
    return {"status": "full_reset", "session_id": session_id}


# ── Info Routes ───────────────────────────────────────

@app.get("/constraints/available", tags=["Info"])
async def list_available_constraints():
    """List all supported dietary constraint keywords."""
    return {
        "constraints": sorted(list(KNOWN_CONSTRAINTS)),
        "usage": "Pass any of these in POST /session/{id}/constraints"
    }


@app.get("/health", tags=["System"])
async def health():
    return {
        "status": "ok",
        "agent": "Chef Cuisto",
        "model": settings.OPENAI_MODEL,
        "vision_model": "gpt-4o",
        "active_sessions": len(_sessions),
    }
