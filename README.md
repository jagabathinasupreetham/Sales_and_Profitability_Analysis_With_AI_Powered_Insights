# Sales & Profitability Analytics with AI-Powered Insights

A portfolio project combining Python/Pandas, SQL, Tableau, and a Streamlit chatbot to explore sales and profitability data.

## Project Overview

This project uses the Global Superstore dataset to explore sales and profit patterns, answer business questions, and present findings through visualizations and a chatbot. The dataset used during development contains **51,290 rows and 23 columns**.

## Features

- Clean and prepare data with Python and Pandas.
- Perform exploratory data analysis (EDA).
- Use SQL for business-focused summaries and analysis.
- Present metrics and trends in a Tableau dashboard.
- Explore supported dataset questions through a Streamlit chatbot:
  - **AI Mode:** Uses an LLM API to help interpret natural-language questions and phrase responses conversationally.
  - **Demo Mode:** Uses predefined logic for supported questions without requiring an LLM API key.
- Automated tests for selected analytics and chatbot behaviours.

## Technology Stack

Python · Pandas · SQL · Tableau · Streamlit · LLM API · pytest

## Workflow

1. Load the Global Superstore dataset.
2. Clean and prepare the data with Python and Pandas.
3. Explore sales and profitability patterns.
4. Perform business analysis using SQL.
5. Build Tableau visualizations.
6. Use the Streamlit chatbot to explore supported questions.
7. Validate selected outputs through automated tests and manual checks.

## Chatbot Architecture

At a high level, the chatbot checks whether a question is relevant to the dataset and whether the application can handle it. Supported questions are converted into a structured request and passed to analytics logic that calculates results from the data. AI Mode can use an LLM to help interpret questions and phrase responses; Demo Mode uses predefined logic and returns direct results.

The LLM is not intended to replace data calculations. Verify important business figures against the underlying data or independent SQL/Pandas calculations before using them for decisions.

## Getting Started

### Requirements

- Python 3
- Git
- An LLM API key for AI Mode, if required by the app configuration

### 1. Clone the repository

Replace `<your-repository-url>` with your GitHub repository URL:

```bash
git clone <your-repository-url>
cd sales_ai_chatbot
```

If your repository folder has a different name, use that folder name in the `cd` command.

### 2. Create and activate a virtual environment (macOS)

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
python3 -m pip install -r requirements.txt
```

### 4. Configure AI Mode

Create a local `.env` file and set the API key using the environment-variable name expected by your project configuration.

**Never commit `.env` files or expose API keys in source code, screenshots, or GitHub.** Demo Mode can be used without an LLM API key if supported by the current app configuration.

### 5. Run the app

```bash
python3 -m streamlit run app.py
```

Open the local URL printed in the terminal.

## Running Tests

With the virtual environment activated:

```bash
python3 -m pytest -v
```

Or call the environment's interpreter directly:

```bash
./.venv/bin/python3 -m pytest -v
```

The test suite previously completed with **44 passing tests** in the development environment. Rerun tests on the version you publish, since results can change after code or dependency updates.

## Example Business Questions

- Which product category generates the highest total sales?
- What are the total sales across all orders?
- Compare total sales and total profit by category.
- Compare total sales and total profit by region.

Supported questions depend on the current implementation.

## Screenshots

Add screenshots to a folder such as `screenshots/`, then update these paths:

- Tableau dashboard: `tableau_dashboard.png`
- Chatbot interface: `chatbot_interface.png`

Embed them in this README using:

```markdown
![Tableau dashboard](screenshots/tableau-dashboard.png)

![Chatbot interface](screenshots/chatbot.png)
```

## Dataset

This project uses the **Global Superstore** dataset. Before publishing, add the original source/download link and check the dataset's license and redistribution terms. If redistribution is not permitted, document how to obtain the dataset rather than committing the raw data.

## Limitations

- Manual testing of representative questions found some inconsistent responses in both AI Mode and Demo Mode.
- No overall chatbot accuracy percentage is claimed.
- AI-generated interpretations or explanations may be incomplete or inconsistent.
- The chatbot supports questions covered by its current analytics logic and dataset.
- Verify important business results against independent calculations.

## Possible Future Improvements

- Improve consistency for questions requesting multiple metrics.
- Expand validation for supported and unsupported questions.
- Add automated tests for edge cases and multi-metric questions.
- Improve error messages for unsupported questions.
- Add screenshots and a short project demonstration.

## Author

**Supreetham Jagabathina**
