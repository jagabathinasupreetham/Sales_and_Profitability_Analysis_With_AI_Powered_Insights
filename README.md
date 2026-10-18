# Sales AI Chatbot — Fresh Project

A standalone Streamlit chatbot for the Superstore sales dataset. It does not import or reuse the previous chatbot source code or its old `Resolution_Status` schema.

## Architecture
User question → Validator 1 → Validator 2 → Sarvam JSON plan → JSON validation → deterministic Pandas analytics → exact result → Sarvam natural-language explanation.

The LLM proposes a constrained JSON analytics plan. It does **not** write or execute arbitrary SQL or Python. The plan parser validates the action, metrics, grouping field, sorting and filters before the local engine runs. For comparison plans, if the model omits `metric` but supplies a valid `metrics` list, the parser derives the primary metric from explicit ranking language in the question or falls back to the first declared metric. If neither field is supplied, the app returns a clear validation error.

## Dataset
The bundled CSV is expected to have 51,290 rows and 23 columns. The app checks the schema and displays the loaded path and dimensions. It stops with a clear error if the wrong dataset is loaded.

## Setup on macOS
1. Extract the ZIP.
2. Open Terminal and `cd` into `sales_ai_chatbot`.
3. Create an environment and install dependencies:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```
4. Copy `.env.example` to `.env` and add your Sarvam API key.
5. Start the app:
   ```bash
   streamlit run app.py
   ```

You can use **Demo mode** without an API key to test deterministic analytics. Sarvam calls require a valid API key and a model/API endpoint supported by your account.

## Tests
```bash
python -m pytest
```

The AI-plan regression tests cover a missing primary metric in multi-metric plans, compatibility with older single-metric plans, clear rejection when all metric fields are missing, and the sales/profit-by-category calculation.

## Deterministic question routing

The app now tries `analytics/question_router.py` before demo or Sarvam planning.
This module handles common, clearly phrased analytics questions locally with
Pandas, including totals, averages, grouped summaries, top/bottom rankings,
monthly/yearly trends, highest/lowest periods, and comparisons between two
explicit years with absolute and percentage change. Requests that need multiple
metrics together or do not match a supported pattern continue to the existing
planner. This is intentionally broad coverage, not a claim that every possible
natural-language question can be resolved without clarification.

Security: keep `.env` local and never include it in a ZIP shared for review.
Use `.env.example` as a template instead.
