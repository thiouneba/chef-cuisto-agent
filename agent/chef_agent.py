"""
Chef Agent Core — Autonomous culinary AI agent.
Accepts image or text input, identifies ingredients, and generates complete recipes.
Author: Bassirou
Stack: LangChain + GPT-4o + Python
"""

from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

from agent.memory import PreferenceMemory
from agent.tools import get_all_tools
from agent.vision import analyze_image_ingredients
from config import settings
import logging

logger = logging.getLogger(__name__)


CHEF_SYSTEM_PROMPT = """You are Chef Jarvis, an expert culinary AI assistant with Michelin-star level knowledge.

Your mission:
1. Receive a list of available ingredients (from image analysis or user text)
2. Respect ALL dietary constraints provided
3. Propose ONE perfect recipe that maximizes the use of available ingredients
4. Provide a complete, detailed, step-by-step recipe until the final dish

Your recipe format MUST always include:
- 🍽️ **Dish name** and brief description
- ⏱️ **Timing**: prep time, cook time, total time
- 👥 **Servings**
- 📋 **Full ingredient list** with precise quantities
- 👨‍🍳 **Step-by-step instructions** (numbered, detailed, with temperatures and timings)
- 💡 **Chef's tips** (substitutions, variations, plating suggestions)
- 🥗 **Nutritional highlights** (approximate)

Rules:
- STRICTLY respect dietary constraints — this is non-negotiable
- If an ingredient is missing but common (salt, oil, water), you may include it and flag it
- Always suggest what to do with leftover ingredients
- Be encouraging and pedagogical — explain WHY each step matters
- If ingredients are very limited, be creative but realistic

The user's dietary constraints and preferences are provided in the conversation context.
"""


class ChefAgent:
    """
    Chef Cuisto Agent — processes ingredients (from image or text)
    and generates complete recipes respecting dietary constraints.
    """

    def __init__(self, session_id: str = "default", verbose: bool = False):
        self.session_id = session_id
        self.memory = PreferenceMemory(session_id=session_id)
        self.verbose = verbose
        self._build_agent()
        logger.info(f"Chef Agent initialized | session={session_id}")

    def _build_agent(self):
        self.llm = ChatOpenAI(
            model=settings.OPENAI_MODEL,
            temperature=settings.TEMPERATURE,
            openai_api_key=settings.OPENAI_API_KEY,
        )
        tools = get_all_tools()
        prompt = ChatPromptTemplate.from_messages([
            ("system", CHEF_SYSTEM_PROMPT),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])
        agent = create_openai_tools_agent(
            llm=self.llm,
            tools=tools,
            prompt=prompt,
        )
        self.executor = AgentExecutor(
            agent=agent,
            tools=tools,
            verbose=self.verbose,
            handle_parsing_errors=True,
            max_iterations=settings.MAX_AGENT_ITERATIONS,
            return_intermediate_steps=True,
        )

    def cook_from_text(self, ingredients_text: str) -> dict:
        """Generate a recipe from a textual list of ingredients."""
        constraints = self.memory.get_constraints_summary()
        prompt = f"""
Here are the available ingredients:
{ingredients_text}

{constraints}

Please propose the best possible recipe using these ingredients.
"""
        return self._run(prompt)

    def cook_from_image(self, image_base64: str, media_type: str = "image/jpeg") -> dict:
        """Analyze an image, extract ingredients, then generate a recipe."""
        logger.info(f"[{self.session_id}] Analyzing image for ingredients...")

        # Step 1: Vision — extract ingredients from image
        detected = analyze_image_ingredients(
            image_base64=image_base64,
            media_type=media_type,
            llm=self.llm,
        )
        logger.info(f"Detected ingredients: {detected['ingredients']}")

        # Step 2: Cook from detected ingredients
        constraints = self.memory.get_constraints_summary()
        prompt = f"""
I analyzed the photo and detected these ingredients:
{detected['ingredients_text']}

Additional context from image: {detected.get('context', 'none')}

{constraints}

Please propose the best possible recipe using these ingredients.
"""
        result = self._run(prompt)
        result["detected_ingredients"] = detected["ingredients"]
        return result

    def _run(self, prompt: str) -> dict:
        """Internal method to run the agent executor."""
        history = self.memory.get_history()
        try:
            result = self.executor.invoke({
                "input": prompt,
                "chat_history": history,
            })
            output = result["output"]
            steps = self._format_steps(result.get("intermediate_steps", []))
            self.memory.add_exchange(human=prompt, ai=output)
            return {
                "recipe": output,
                "tools_used": steps,
                "session_id": self.session_id,
            }
        except Exception as e:
            logger.error(f"Agent run error: {e}")
            raise

    def _format_steps(self, steps: list) -> list[dict]:
        formatted = []
        for action, observation in steps:
            formatted.append({
                "tool": action.tool,
                "input": str(action.tool_input)[:200],
                "output": str(observation)[:300],
            })
        return formatted

    def set_constraints(self, constraints: list[str]):
        """Set dietary constraints for this session."""
        self.memory.set_constraints(constraints)
        logger.info(f"Constraints set for session={self.session_id}: {constraints}")

    def get_constraints(self) -> list[str]:
        return self.memory.get_constraints()

    def reset(self):
        """Clear conversation history (keep constraints)."""
        self.memory.clear_history()
