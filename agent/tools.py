"""
Culinary Tools — Specialized tools for the Chef Cuisto agent.

Tools:
    1. search_recipe_online     → Search recipes by ingredients on the web
    2. get_nutrition_info       → Nutritional data for a dish or ingredient
    3. get_cooking_technique    → Explain a cooking technique in detail
    4. convert_measurement      → Convert cooking measurements
    5. suggest_substitution     → Suggest ingredient substitutions
    6. check_constraint         → Verify if a recipe respects dietary constraints
"""

from langchain_core.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun
import logging

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# 1. RECIPE SEARCH
# ─────────────────────────────────────────────

_search = DuckDuckGoSearchRun()

@tool
def search_recipe_online(query: str) -> str:
    """
    Search the internet for recipe ideas and inspiration.
    Use this when you want to find classic recipes or creative combinations
    for a specific set of ingredients.
    Input: a search query like 'recipe with chicken zucchini tomatoes'
    """
    try:
        result = _search.run(query)
        return f"Recipe search results:\n{result}"
    except Exception as e:
        return f"Search error: {str(e)}"


# ─────────────────────────────────────────────
# 2. NUTRITION INFO
# ─────────────────────────────────────────────

@tool
def get_nutrition_info(food_item: str) -> str:
    """
    Get approximate nutritional information for a food item or dish.
    Use this to provide the user with nutritional highlights of the recipe.
    Input: a food name like 'grilled chicken breast' or 'pasta carbonara'
    """
    try:
        result = _search.run(f"nutritional value per 100g {food_item} calories protein carbs fat")
        return f"Nutrition info for '{food_item}':\n{result[:800]}"
    except Exception as e:
        return f"Could not retrieve nutrition info: {str(e)}"


# ─────────────────────────────────────────────
# 3. COOKING TECHNIQUE
# ─────────────────────────────────────────────

@tool
def get_cooking_technique(technique: str) -> str:
    """
    Get a detailed explanation of a specific cooking technique.
    Use this when a recipe step involves a technique the user might not know
    (e.g., 'blanching', 'deglazing', 'julienne', 'beurre blanc').
    Input: the name of the technique.
    """
    techniques = {
        "blanching": "Briefly boil vegetables (1-3 min) then immediately plunge into ice water. Preserves color, texture, and nutrients. Used before freezing or for salads.",
        "deglazing": "Add liquid (wine, stock, water) to a hot pan after searing to lift the caramelized bits (fond). Creates rich sauces. Stir constantly over high heat.",
        "julienne": "Cut vegetables into thin matchstick strips (3mm × 3mm × 5cm). Ensures even, quick cooking and elegant presentation.",
        "sauté": "Cook quickly in a small amount of fat over high heat, stirring frequently. Best for tender vegetables and small cuts of protein.",
        "braise": "Brown the ingredient first, then cook slowly in liquid (partially submerged) in a covered pot. Makes tough cuts tender and flavorful.",
        "fold": "Gently incorporate a light mixture (e.g., whipped cream) into a heavier one using a spatula with a down-across-up motion to preserve air.",
        "rest": "Let cooked meat sit covered for 5-15 min after cooking. Allows juices to redistribute — never skip this step.",
        "caramelize": "Cook onions or sugar slowly over medium-low heat until natural sugars brown (20-45 min for onions). Develops deep, sweet flavor.",
        "mise en place": "French for 'everything in its place'. Prepare and measure all ingredients BEFORE starting to cook. Essential for smooth execution.",
    }

    key = technique.lower().strip()
    if key in techniques:
        return f"🍳 {technique.title()}: {techniques[key]}"

    # Fallback to web search
    try:
        result = _search.run(f"cooking technique {technique} explained step by step")
        return f"Technique '{technique}':\n{result[:600]}"
    except Exception as e:
        return f"Could not find technique info: {str(e)}"


# ─────────────────────────────────────────────
# 4. MEASUREMENT CONVERTER
# ─────────────────────────────────────────────

@tool
def convert_measurement(conversion_request: str) -> str:
    """
    Convert cooking measurements between units.
    Use when a recipe uses unfamiliar units or the user needs conversions.
    Input examples:
        '2 cups to ml'
        '1 tablespoon to grams for flour'
        '350 fahrenheit to celsius'
        '200g butter to cups'
    """
    conversions = {
        # Volume
        "1 cup": "240 ml",
        "1 tablespoon": "15 ml",
        "1 teaspoon": "5 ml",
        "1 fluid ounce": "30 ml",
        "1 pint": "473 ml",
        # Temperature
        "180c": "356°F (standard oven — moderate)",
        "200c": "392°F (hot oven)",
        "220c": "428°F (very hot oven)",
        "350f": "177°C",
        "375f": "190°C",
        "400f": "204°C",
        # Common weights
        "1 cup flour": "~120g",
        "1 cup sugar": "~200g",
        "1 cup butter": "~227g",
        "1 cup rice": "~185g",
    }

    request_lower = conversion_request.lower()
    for key, value in conversions.items():
        if key in request_lower:
            return f"📏 {key} = {value}"

    # Fallback to web
    try:
        result = _search.run(f"cooking measurement conversion {conversion_request}")
        return result[:400]
    except Exception as e:
        return f"Conversion error: {str(e)}"


# ─────────────────────────────────────────────
# 5. INGREDIENT SUBSTITUTION
# ─────────────────────────────────────────────

@tool
def suggest_substitution(ingredient_and_reason: str) -> str:
    """
    Suggest substitutions for a missing ingredient or one that violates dietary constraints.
    Use when the user doesn't have an ingredient or needs a dietary-safe alternative.
    Input format: 'ingredient | reason'
    Examples:
        'butter | vegan'
        'eggs | allergy'
        'milk | lactose intolerant'
        'flour | gluten-free'
        'heavy cream | dairy-free'
    """
    substitutions = {
        # Vegan / dairy-free
        "butter|vegan": "Coconut oil (1:1), vegan butter, or olive oil for savory dishes",
        "butter|dairy-free": "Coconut oil (1:1) or vegan margarine",
        "milk|vegan": "Oat milk, almond milk, or soy milk (1:1)",
        "milk|dairy-free": "Oat milk or coconut milk (1:1)",
        "heavy cream|vegan": "Full-fat coconut cream (1:1) — whips similarly",
        "heavy cream|dairy-free": "Full-fat coconut cream",
        "cheese|vegan": "Nutritional yeast for flavor, cashew cream for texture",
        "eggs|vegan": "1 egg = 1 tbsp flaxseed + 3 tbsp water (let sit 5 min), or 1/4 cup applesauce",
        "eggs|allergy": "1 egg = 3 tbsp aquafaba (chickpea water) or 1/4 cup unsweetened applesauce",
        # Gluten-free
        "flour|gluten-free": "Rice flour, almond flour, or a 1:1 gluten-free blend",
        "breadcrumbs|gluten-free": "Crushed gluten-free crackers, almond meal, or ground oats (certified GF)",
        "soy sauce|gluten-free": "Tamari sauce or coconut aminos (1:1)",
        # Low-carb / keto
        "flour|keto": "Almond flour or coconut flour (use 1/4 the amount for coconut flour)",
        "sugar|keto": "Erythritol or stevia (adjust to taste)",
        "pasta|keto": "Zucchini noodles (zoodles) or shirataki noodles",
        "rice|keto": "Cauliflower rice",
        # Halal
        "wine|halal": "Grape juice + splash of vinegar, or pomegranate juice for depth",
        "pork|halal": "Beef, lamb, or chicken (same weight)",
        "gelatin|halal": "Agar-agar (same quantity)",
    }

    parts = ingredient_and_reason.lower().split("|")
    if len(parts) == 2:
        key = f"{parts[0].strip()}|{parts[1].strip()}"
        if key in substitutions:
            return f"✅ Substitute for {ingredient_and_reason}:\n→ {substitutions[key]}"

    # Fallback
    try:
        result = _search.run(f"substitute for {ingredient_and_reason} in cooking")
        return f"Substitution options for '{ingredient_and_reason}':\n{result[:500]}"
    except Exception as e:
        return f"Error finding substitution: {str(e)}"


# ─────────────────────────────────────────────
# 6. CONSTRAINT CHECKER
# ─────────────────────────────────────────────

@tool
def check_dietary_constraint(recipe_and_constraint: str) -> str:
    """
    Verify whether a recipe or ingredient is compatible with a dietary constraint.
    Use this to double-check compliance before proposing a recipe step.
    Input format: 'ingredient_or_dish | constraint'
    Examples:
        'parmesan cheese | vegan'
        'chicken broth | vegetarian'
        'soy sauce | gluten-free'
        'honey | vegan'
    """
    # Known violations
    violations = {
        "parmesan|vegan": "❌ Parmesan is NOT vegan (animal rennet + dairy)",
        "parmesan|vegetarian": "⚠️ Traditional parmesan uses animal rennet — use vegetarian hard cheese instead",
        "honey|vegan": "❌ Honey is NOT vegan. Use maple syrup or agave nectar instead.",
        "gelatin|vegan": "❌ Gelatin is animal-derived. Use agar-agar instead.",
        "gelatin|vegetarian": "❌ Gelatin is animal-derived. Use agar-agar instead.",
        "anchovies|vegetarian": "❌ Anchovies are fish — not vegetarian.",
        "worcestershire|vegetarian": "⚠️ Classic Worcestershire contains anchovies. Use Henderson's Relish instead.",
        "chicken broth|vegetarian": "❌ Chicken broth is not vegetarian. Use vegetable broth.",
        "chicken broth|vegan": "❌ Chicken broth is not vegan. Use vegetable broth.",
        "soy sauce|gluten-free": "❌ Standard soy sauce contains wheat. Use tamari or coconut aminos.",
        "regular pasta|gluten-free": "❌ Regular pasta contains gluten. Use rice pasta or chickpea pasta.",
        "beer|gluten-free": "❌ Regular beer contains gluten. Use gluten-free beer or omit.",
        "lard|halal": "❌ Lard is pork fat — not halal. Use vegetable oil or beef tallow.",
        "bacon|halal": "❌ Bacon is pork — not halal.",
        "wine|halal": "❌ Wine is alcohol — not halal. Use grape juice + vinegar as substitute.",
    }

    parts = recipe_and_constraint.lower().split("|")
    if len(parts) == 2:
        key = f"{parts[0].strip()}|{parts[1].strip()}"
        for violation_key, message in violations.items():
            if violation_key in key or key in violation_key:
                return message

    return f"✅ '{recipe_and_constraint}' appears compatible. No known violations detected."


# ─────────────────────────────────────────────
# TOOL REGISTRY
# ─────────────────────────────────────────────

def get_all_tools() -> list:
    """Return all culinary tools to register with the Chef Agent."""
    return [
        search_recipe_online,
        get_nutrition_info,
        get_cooking_technique,
        convert_measurement,
        suggest_substitution,
        check_dietary_constraint,
    ]
