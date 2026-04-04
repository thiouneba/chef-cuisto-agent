"""
Tests — Chef Cuisto Agent
Run: pytest tests/ -v
"""

import pytest
from unittest.mock import MagicMock, patch
import json
import base64


# ─────────────────────────────────────────────
# MEMORY / CONSTRAINTS TESTS
# ─────────────────────────────────────────────

class TestPreferenceMemory:

    def setup_method(self):
        from agent.memory import PreferenceMemory
        self.mem = PreferenceMemory(session_id="test_chef_01")
        self.mem.full_reset()

    def test_set_and_get_constraints(self):
        self.mem.set_constraints(["vegan", "gluten-free"])
        assert "vegan" in self.mem.get_constraints()
        assert "gluten-free" in self.mem.get_constraints()

    def test_constraints_are_lowercased(self):
        self.mem.set_constraints(["VEGAN", "Gluten-Free"])
        constraints = self.mem.get_constraints()
        assert "vegan" in constraints
        assert "gluten-free" in constraints

    def test_add_single_constraint(self):
        self.mem.add_constraint("halal")
        assert "halal" in self.mem.get_constraints()

    def test_remove_constraint(self):
        self.mem.set_constraints(["vegan", "halal"])
        self.mem.remove_constraint("halal")
        assert "halal" not in self.mem.get_constraints()
        assert "vegan" in self.mem.get_constraints()

    def test_constraints_summary_empty(self):
        summary = self.mem.get_constraints_summary()
        assert "None" in summary or "freely" in summary.lower()

    def test_constraints_summary_with_data(self):
        self.mem.set_constraints(["vegan"])
        self.mem.dislike_ingredient("mushrooms")
        summary = self.mem.get_constraints_summary()
        assert "vegan" in summary
        assert "mushrooms" in summary

    def test_liked_disliked_ingredients(self):
        self.mem.like_ingredient("Garlic")
        self.mem.dislike_ingredient("Cilantro")
        prefs = self.mem.get_preferences()
        assert "garlic" in prefs["liked"]
        assert "cilantro" in prefs["disliked"]

    def test_clear_history_keeps_constraints(self):
        self.mem.set_constraints(["keto"])
        self.mem.add_exchange("hello", "hi")
        self.mem.clear_history()
        assert self.mem.get_constraints() == ["keto"]
        assert len(self.mem.get_history()) == 0

    def test_full_reset_clears_everything(self):
        self.mem.set_constraints(["vegan"])
        self.mem.add_exchange("hello", "hi")
        self.mem.full_reset()
        assert self.mem.get_constraints() == []
        assert len(self.mem.get_history()) == 0

    def test_get_summary(self):
        self.mem.set_constraints(["vegetarian"])
        self.mem.add_exchange("q", "a")
        summary = self.mem.get_summary()
        assert summary["session_id"] == "test_chef_01"
        assert "vegetarian" in summary["constraints"]
        assert summary["history_length"] == 2


# ─────────────────────────────────────────────
# TOOLS TESTS
# ─────────────────────────────────────────────

class TestCulinaryTools:

    def test_convert_measurement_cup_to_ml(self):
        from agent.tools import convert_measurement
        result = convert_measurement.invoke({"conversion_request": "1 cup to ml"})
        assert "240" in result or "ml" in result.lower()

    def test_convert_measurement_temperature(self):
        from agent.tools import convert_measurement
        result = convert_measurement.invoke({"conversion_request": "180c to fahrenheit"})
        assert "356" in result or "°F" in result

    def test_suggest_substitution_butter_vegan(self):
        from agent.tools import suggest_substitution
        result = suggest_substitution.invoke({"ingredient_and_reason": "butter | vegan"})
        assert "coconut" in result.lower() or "vegan" in result.lower()

    def test_suggest_substitution_eggs_vegan(self):
        from agent.tools import suggest_substitution
        result = suggest_substitution.invoke({"ingredient_and_reason": "eggs | vegan"})
        assert "flax" in result.lower() or "aquafaba" in result.lower() or "applesauce" in result.lower()

    def test_suggest_substitution_flour_gluten_free(self):
        from agent.tools import suggest_substitution
        result = suggest_substitution.invoke({"ingredient_and_reason": "flour | gluten-free"})
        assert "rice" in result.lower() or "almond" in result.lower()

    def test_check_constraint_honey_vegan(self):
        from agent.tools import check_dietary_constraint
        result = check_dietary_constraint.invoke({"recipe_and_constraint": "honey | vegan"})
        assert "❌" in result or "not vegan" in result.lower()

    def test_check_constraint_soy_sauce_gluten_free(self):
        from agent.tools import check_dietary_constraint
        result = check_dietary_constraint.invoke({"recipe_and_constraint": "soy sauce | gluten-free"})
        assert "tamari" in result.lower() or "❌" in result

    def test_check_constraint_no_violation(self):
        from agent.tools import check_dietary_constraint
        result = check_dietary_constraint.invoke({"recipe_and_constraint": "olive oil | vegan"})
        assert "✅" in result or "compatible" in result.lower()

    def test_cooking_technique_known(self):
        from agent.tools import get_cooking_technique
        result = get_cooking_technique.invoke({"technique": "blanching"})
        assert "boil" in result.lower() or "ice" in result.lower()

    def test_cooking_technique_rest(self):
        from agent.tools import get_cooking_technique
        result = get_cooking_technique.invoke({"technique": "rest"})
        assert "juices" in result.lower() or "meat" in result.lower()


# ─────────────────────────────────────────────
# VISION TESTS (mocked)
# ─────────────────────────────────────────────

class TestVisionModule:

    @patch("agent.vision.ChatOpenAI")
    def test_analyze_image_returns_ingredients(self, mock_llm_cls):
        from agent.vision import analyze_image_ingredients

        mock_response = MagicMock()
        mock_response.content = json.dumps({
            "ingredients": ["tomatoes", "garlic", "olive oil", "basil"],
            "ingredients_text": "tomatoes, garlic, olive oil, and basil",
            "context": "kitchen counter",
            "confidence": "high",
            "notes": ""
        })
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = mock_response
        mock_llm_cls.return_value = mock_llm

        dummy_image = base64.b64encode(b"fake_image_bytes").decode("utf-8")
        result = analyze_image_ingredients(image_base64=dummy_image, llm=mock_llm)

        assert "tomatoes" in result["ingredients"]
        assert result["confidence"] == "high"

    @patch("agent.vision.ChatOpenAI")
    def test_analyze_image_handles_bad_json(self, mock_llm_cls):
        from agent.vision import analyze_image_ingredients

        mock_response = MagicMock()
        mock_response.content = "This is not JSON at all."
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = mock_response

        dummy_image = base64.b64encode(b"fake_image").decode("utf-8")
        result = analyze_image_ingredients(image_base64=dummy_image, llm=mock_llm)

        assert result["ingredients"] == []
        assert result["confidence"] == "low"


# ─────────────────────────────────────────────
# CHEF AGENT INTEGRATION (mocked)
# ─────────────────────────────────────────────

class TestChefAgentIntegration:

    @patch("agent.chef_agent.create_openai_tools_agent")
    @patch("agent.chef_agent.ChatOpenAI")
    @patch("agent.chef_agent.get_all_tools")
    def test_cook_from_text(self, mock_tools, mock_llm, mock_agent_fn):
        from agent.chef_agent import ChefAgent

        mock_tools.return_value = []
        with patch("agent.chef_agent.AgentExecutor") as mock_exec_cls:
            mock_exec = MagicMock()
            mock_exec.invoke.return_value = {
                "output": "Here is a tomato pasta recipe...",
                "intermediate_steps": [],
            }
            mock_exec_cls.return_value = mock_exec

            agent = ChefAgent(session_id="test_integration")
            result = agent.cook_from_text("tomatoes, pasta, garlic, olive oil")

            assert "recipe" in result
            assert "tomato" in result["recipe"].lower() or "pasta" in result["recipe"].lower()

    @patch("agent.chef_agent.create_openai_tools_agent")
    @patch("agent.chef_agent.ChatOpenAI")
    @patch("agent.chef_agent.get_all_tools")
    def test_constraints_are_set(self, mock_tools, mock_llm, mock_agent_fn):
        from agent.chef_agent import ChefAgent

        mock_tools.return_value = []
        with patch("agent.chef_agent.AgentExecutor"):
            agent = ChefAgent(session_id="test_constraints")
            agent.set_constraints(["vegan", "gluten-free"])
            constraints = agent.get_constraints()
            assert "vegan" in constraints
            assert "gluten-free" in constraints
