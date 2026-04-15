# 🍳 Chef Cuisto Agent

> An autonomous multimodal AI cooking agent built with **LangChain and GPT-4o**, capable of processing both textual and visual ingredient inputs. The system leverages an agentic workflow to extract structured ingredient data, enforce dietary constraints, and generate coherent, step-by-step recipes through a multi-stage reasoning pipeline.

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/LangChain-0.3-1C3C3C?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/GPT--4o-Vision-412991?style=for-the-badge&logo=openai&logoColor=white"/>
  <img src="https://img.shields.io/badge/FastAPI-0.115-009688?style=for-the-badge&logo=fastapi"/>
</p>

---

## ✨ What it does

1. **You send** a photo of your fridge/ingredients OR a text list
2. **GPT-4o Vision** analyzes the image and identifies all ingredients
3. **The Chef Agent** reasons about what can be cooked
4. **It generates** a complete recipe: title, timing, servings, ingredient list, step-by-step instructions, tips, and nutrition highlights
5. **It respects** your dietary constraints — strictly (vegan, halal, gluten-free, keto, and more)

---

## 🏗️ Architecture

```
chef-cuisto-agent/
│
├── main.py                  # CLI or API server entrypoint
├── config.py                # Pydantic settings
│
├── agent/
│   ├── chef_agent.py        # ChefAgent — orchestrates everything
│   ├── vision.py            # GPT-4o image analysis → ingredient list
│   ├── memory.py            # Dietary constraints + preference memory
│   └── tools.py             # 6 culinary tools
│
├── api/
│   └── routes.py            # FastAPI endpoints
│
└── tests/
    └── test_chef.py         # Unit + integration tests
```

---

## 🛠️ Culinary Tools

| Tool | Purpose |
|------|---------|
| `search_recipe_online` | Web search for recipe inspiration |
| `get_nutrition_info` | Nutritional info for any dish |
| `get_cooking_technique` | Detailed technique explanations (blanching, braising...) |
| `convert_measurement` | Cups → ml, °F → °C, etc. |
| `suggest_substitution` | Constraint-safe ingredient swaps |
| `check_dietary_constraint` | Validate recipe compliance |

---

## 🚀 Quick Start

```bash
git clone https://github.com/bassirou/chef-cuisto-agent.git
cd chef-cuisto-agent

python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env  # Add your OPENAI_API_KEY
```

### CLI mode

```bash
python main.py

🧑 Your ingredients: chicken breast, zucchini, tomatoes, garlic, olive oil, lemon

👨‍🍳 Chef Cuisto:
🍽️ Provençal Chicken with Zucchini...
⏱️ Prep: 10 min | Cook: 25 min | Total: 35 min
...
```

### API mode

```bash
python main.py --serve
# → http://localhost:8000/docs
```

---

## 🔌 API Usage

### Set dietary constraints first
```bash
curl -X POST http://localhost:8000/session/user_42/constraints \
  -H "Content-Type: application/json" \
  -d '{"constraints": ["vegan", "gluten-free"]}'
```

### Recipe from text
```bash
curl -X POST http://localhost:8000/recipe/from-text \
  -H "Content-Type: application/json" \
  -d '{
    "ingredients": "tofu, broccoli, soy sauce, garlic, sesame oil, rice",
    "session_id": "user_42"
  }'
```

### Recipe from image
```bash
curl -X POST "http://localhost:8000/recipe/from-image?session_id=user_42" \
  -F "file=@fridge_photo.jpg"
```

---

## 🥗 Supported Dietary Constraints

`vegetarian` · `vegan` · `gluten-free` · `dairy-free` · `nut-free` · `halal` · `kosher` · `low-carb` · `keto` · `paleo` · `no-pork` · `no-seafood` · `no-eggs` · `low-sodium` · `diabetic-friendly`

Constraints persist across all requests in a session. Only a full reset clears them.

---

## 🧪 Tests

```bash
pytest tests/ -v
```

---

## 💡 Design Decisions

**Why `temperature=0.3` instead of 0?**
Cooking benefits from slight creativity. A fully deterministic model (temp=0) tends to always suggest the same obvious dishes. A small temperature nudge produces more interesting, varied recipes without losing coherence.

**Why GPT-4o for vision?**
GPT-4o has the best food recognition accuracy. The vision prompt requests structured JSON output, making parsing reliable and the downstream agent clean.

---

## 👤 Author

**Bassirou** — AI Engineer  
Specializing in LangChain, RAG pipelines, and production AI systems.

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0077B5?style=flat&logo=linkedin)](https://www.linkedin.com/in/bassirou-thioune-01b9131b6/)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-181717?style=flat&logo=github)](https://github.com/thiouneba)

---

## 📄 License

MIT — feel free to use, fork, and build on this.
