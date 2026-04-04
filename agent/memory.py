"""
Preference Memory — Stores dietary constraints + conversation history per session.
Constraints persist across resets; history can be cleared independently.
"""

from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from config import settings
import logging

logger = logging.getLogger(__name__)

# All supported dietary constraints
KNOWN_CONSTRAINTS = {
    "vegetarian",
    "vegan",
    "gluten-free",
    "dairy-free",
    "nut-free",
    "halal",
    "kosher",
    "low-carb",
    "keto",
    "paleo",
    "no-pork",
    "no-seafood",
    "no-eggs",
    "low-sodium",
    "diabetic-friendly",
}


class PreferenceMemory:
    """
    Per-session memory that stores:
    - Dietary constraints (persistent — not cleared on reset)
    - Conversation history (sliding window)
    - Ingredient preferences (liked / disliked)
    """

    _store: dict[str, "PreferenceMemory"] = {}

    def __init__(self, session_id: str = "default"):
        self.session_id = session_id
        self._history = ChatMessageHistory()
        self._constraints: list[str] = []
        self._liked: list[str] = []
        self._disliked: list[str] = []

    # ── Constraints ──────────────────────────────

    def set_constraints(self, constraints: list[str]):
        """Set dietary constraints. Replaces existing ones."""
        normalized = [c.lower().strip() for c in constraints]
        self._constraints = normalized
        logger.info(f"[{self.session_id}] Constraints updated: {normalized}")

    def add_constraint(self, constraint: str):
        """Add a single constraint."""
        c = constraint.lower().strip()
        if c not in self._constraints:
            self._constraints.append(c)

    def remove_constraint(self, constraint: str):
        """Remove a single constraint."""
        c = constraint.lower().strip()
        self._constraints = [x for x in self._constraints if x != c]

    def get_constraints(self) -> list[str]:
        return self._constraints.copy()

    def get_constraints_summary(self) -> str:
        """Returns a formatted string to inject into the agent prompt."""
        if not self._constraints and not self._disliked:
            return "Dietary constraints: None. Cook freely!"

        parts = []
        if self._constraints:
            parts.append(f"⚠️ DIETARY CONSTRAINTS (strictly respect): {', '.join(self._constraints)}")
        if self._disliked:
            parts.append(f"🚫 Ingredients to avoid (user dislikes): {', '.join(self._disliked)}")
        if self._liked:
            parts.append(f"❤️ Preferred ingredients (use if possible): {', '.join(self._liked)}")

        return "\n".join(parts)

    # ── Preferences ──────────────────────────────

    def like_ingredient(self, ingredient: str):
        i = ingredient.lower().strip()
        if i not in self._liked:
            self._liked.append(i)

    def dislike_ingredient(self, ingredient: str):
        i = ingredient.lower().strip()
        if i not in self._disliked:
            self._disliked.append(i)

    def get_preferences(self) -> dict:
        return {"liked": self._liked, "disliked": self._disliked}

    # ── History ──────────────────────────────────

    def add_exchange(self, human: str, ai: str):
        self._history.add_message(HumanMessage(content=human))
        self._history.add_message(AIMessage(content=ai))

    def get_history(self) -> list[BaseMessage]:
        msgs = self._history.messages
        max_msgs = settings.MAX_MEMORY_MESSAGES * 2
        return msgs[-max_msgs:] if len(msgs) > max_msgs else msgs

    def clear_history(self):
        """Clear conversation history but keep constraints."""
        self._history.clear()
        logger.info(f"[{self.session_id}] History cleared (constraints preserved)")

    def full_reset(self):
        """Clear everything including constraints."""
        self._history.clear()
        self._constraints = []
        self._liked = []
        self._disliked = []
        logger.info(f"[{self.session_id}] Full reset done")

    def get_summary(self) -> dict:
        return {
            "session_id": self.session_id,
            "constraints": self._constraints,
            "liked": self._liked,
            "disliked": self._disliked,
            "history_length": len(self._history.messages),
        }
