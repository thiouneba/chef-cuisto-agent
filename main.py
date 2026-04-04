"""
Main entrypoint — Chef Cuisto Agent
Usage:
    python main.py           # CLI mode
    python main.py --serve   # FastAPI server
"""

import argparse
import sys
import uvicorn
from config import settings
import logging

logging.basicConfig(
    level=settings.LOG_LEVEL,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

BANNER = """
╔══════════════════════════════════════════════════════════╗
║       🍳  C H E F  C U I S T O  A G E N T  v1.0        ║
║   Tell me your ingredients — I'll cook up a recipe!      ║
╚══════════════════════════════════════════════════════════╝

  Type your ingredients and press Enter.
  Commands:
    /constraints [list]  → set dietary constraints (e.g. /constraints vegan, gluten-free)
    /session             → show current session info
    /reset               → clear conversation history
    /quit                → exit
"""


def run_cli():
    from agent.chef_agent import ChefAgent
    print(BANNER)
    agent = ChefAgent(session_id="cli", verbose=True)

    while True:
        try:
            user_input = input("\n🧑 Your ingredients: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n\nBon appétit! 👨‍🍳")
            sys.exit(0)

        if not user_input:
            continue

        if user_input == "/quit":
            print("Bon appétit! 👨‍🍳")
            break

        if user_input == "/reset":
            agent.reset()
            print("🗑️  Conversation history cleared.")
            continue

        if user_input == "/session":
            info = agent.memory.get_summary()
            print(f"\n📋 Session info: {info}")
            continue

        if user_input.startswith("/constraints"):
            parts = user_input.replace("/constraints", "").strip()
            constraints = [c.strip() for c in parts.split(",") if c.strip()]
            agent.set_constraints(constraints)
            print(f"✅ Constraints set: {constraints}")
            continue

        print("\n⏳ Chef Cuisto is thinking...")
        result = agent.cook_from_text(user_input)
        print(f"\n👨‍🍳 Chef Cuisto:\n\n{result['recipe']}")

        if result["tools_used"] and settings.DEBUG:
            print("\n[Debug] Tools used:")
            for step in result["tools_used"]:
                print(f"  → [{step['tool']}] {step['input'][:60]}")


def run_server():
    from api.routes import app
    print(f"\n🚀 Chef Cuisto API running on http://localhost:8000")
    print(f"📖 Docs: http://localhost:8000/docs\n")
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=settings.DEBUG)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chef Cuisto Agent")
    parser.add_argument("--serve", action="store_true", help="Start FastAPI server")
    args = parser.parse_args()

    if args.serve:
        run_server()
    else:
        run_cli()
