from pathlib import Path
import re

import pandas as pd
import streamlit as st

from config import DEFAULT_MODEL
from chatbot.validators import (
    validator_1_dataset_related,
    validator_2_answerable,
)
from chatbot.sarvam_client import chat_completion, SarvamError
from chatbot.json_parser import parse_plan, PlanError
from chatbot.demo_planner import demo_plan
from analytics.pandas_engine import execute_plan
from analytics.question_router import try_deterministic_answer
from analytics.sql_engine import dataframe_to_sqlite


ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "cleaned_sales_data.csv"
EXPECTED_SHAPE = (51290, 23)

REQUIRED_COLUMNS = {
    "row_id",
    "order_id",
    "order_date",
    "ship_date",
    "customer_id",
    "customer_name",
    "category",
    "sub-category",
    "product_name",
    "sales",
    "profit",
    "quantity",
    "discount",
    "shipping_cost",
}


st.set_page_config(
    page_title="Sales AI Analytics Chatbot",
    page_icon="📊",
    layout="wide",
)


@st.cache_data(show_spinner=False)
def load_dataset(path_string: str, modified_time: float):
    """Load and validate the sales dataset."""

    df = pd.read_csv(path_string, encoding="utf-8")

    missing = REQUIRED_COLUMNS - set(df.columns)

    if missing:
        raise ValueError(
            "Dataset schema mismatch. Missing columns: "
            + ", ".join(sorted(missing))
        )

    for col in ["order_date", "ship_date"]:
        df[col] = pd.to_datetime(df[col], errors="coerce")

    for col in [
        "sales",
        "profit",
        "quantity",
        "discount",
        "shipping_cost",
        "row_id",
    ]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    if df.shape != EXPECTED_SHAPE:
        raise ValueError(
            f"Wrong dataset loaded: found {df.shape[0]:,} rows × "
            f"{df.shape[1]} columns; expected "
            f"{EXPECTED_SHAPE[0]:,} × {EXPECTED_SHAPE[1]}. "
            "This app intentionally refuses to continue with "
            "a different dataset."
        )

    return df


@st.cache_resource(show_spinner=False)
def get_sqlite_connection(path_string: str, modified_time: float):
    """Create the SQLite connection used by the application."""

    df = load_dataset(path_string, modified_time)
    return dataframe_to_sqlite(df)


def build_plan_prompt(question: str, columns: list[str]) -> str:
    """Build a concise prompt for the Sarvam analytics planner."""

    return f"""
Convert the user's question into exactly ONE valid JSON analytics plan.
Return JSON only. No markdown, explanations, or additional text.

Allowed actions:
total, average, count, group_sum, group_average, top_n,
monthly_trend, profit_loss_products, compare_groups

Allowed metrics:
sales, profit, quantity, discount, shipping_cost, orders, customers

Allowed group_by fields:
category, sub-category, segment, region, country, market, ship_mode,
order_priority, product_name, customer_name, state

Use this JSON structure:
{{
  "action": "total",
  "metric": "sales",
  "group_by": null,
  "limit": 10,
  "direction": "desc",
  "filters": {{}}
}}

Rules:
1. Include all six JSON keys shown above.
2. Use only allowed actions, metrics, and grouping fields.
3. For total sales, use action "total" and metric "sales".
4. For average sales, use action "average" and metric "sales".
5. For distinct order counts, use metric "orders".
6. For distinct customer counts, use metric "customers".
7. For monthly sales trends, use action "monthly_trend",
   metric "sales", and group_by null.
8. For a specified year, use a filter such as
   {{"order_date": "2014"}}.
9. For highest or top results, use direction "desc".
10. For lowest results, use direction "asc".
11. Use an empty filters object when no filter is needed.
12. Do not invent values or add extra JSON keys.

Dataset columns: {columns}
Question: {question}
""".strip()


def build_minimal_plan_prompt(question: str) -> str:
    """Build a smaller fallback prompt if the first plan is truncated."""

    return f"""
Return exactly one valid JSON object and nothing else.

Keys: action, metric, group_by, limit, direction, filters.

Allowed actions: total, average, count, group_sum, group_average,
top_n, monthly_trend, profit_loss_products, compare_groups.

Allowed metrics: sales, profit, quantity, discount, shipping_cost,
orders, customers.

Allowed group_by fields: category, sub-category, segment, region,
country, market, ship_mode, order_priority, product_name,
customer_name, state.

Rules:
- Total sales: action total, metric sales.
- Highest or lowest month: action monthly_trend, metric sales,
  group_by null.
- Monthly trend: action monthly_trend, group_by null.
- For a year, use filters {{"order_date": "2014"}} with the requested year.
- Highest means direction desc; lowest means direction asc.
- Use limit 1 for a single highest or lowest result.
- Include all six keys.
- Use null for group_by when no grouping is required.
- Use {{}} for filters when there is no filter.

Example:
{{"action":"monthly_trend","metric":"sales","group_by":null,
"limit":1,"direction":"desc","filters":{{"order_date":"2014"}}}}

Question: {question}
""".strip()


def generate_analytics_plan(
    question: str,
    columns: list[str],
    model: str,
):
    """Generate and validate an analytics plan."""

    prompts = [
        build_plan_prompt(question, columns),
        build_minimal_plan_prompt(question),
    ]

    last_error = None

    for attempt, prompt in enumerate(prompts):
        try:
            raw_plan = chat_completion(
                [
                    {
                        "role": "system",
                        "content": (
                            "You generate strict JSON analytics plans. "
                            "Return one complete JSON object only. "
                            "Never include explanations or markdown."
                        ),
                    },
                    {
                        "role": "user",
                        "content": prompt,
                    },
                ],
                model=model,
                temperature=0,
            )

        except SarvamError as exc:
            last_error = exc

            if (
                attempt == 0
                and "finish reason: 'length'" in str(exc)
            ):
                continue

            raise

        try:
            plan = parse_plan(
                raw_plan,
                question=question,
            )
            return plan, raw_plan

        except PlanError as exc:
            last_error = exc

            if attempt == 0:
                continue

            raise

    raise SarvamError(
        "Sarvam could not generate a complete, valid analytics plan "
        "after retrying with a shorter prompt. "
        f"Last error: {last_error}"
    )


def result_as_text(result: dict) -> str:
    """Convert the computed analytics result into readable text."""

    if "value" in result:
        value = result["value"]

        if isinstance(value, (int, float)):
            formatted = f"{value:,.2f}"
        else:
            formatted = str(value)

        return (
            f"**{result['summary']}: {formatted}**\n\n"
            f"Rows used: {result['rows_used']:,}"
        )

    table = pd.DataFrame(result.get("table", []))

    if table.empty:
        return (
            f"{result['summary']}\n\n"
            "No matching records were found."
        )

    answer = (
        f"**{result['summary']}**\n\n"
        + table.to_markdown(index=False)
    )

    answer += f"\n\nRows used: {result['rows_used']:,}"

    profit_col = (
        "total_profit"
        if "total_profit" in table.columns
        else None
    )

    group_by = result.get("group_by")

    if profit_col and group_by and group_by in table.columns:
        numeric_profit = pd.to_numeric(
            table[profit_col],
            errors="coerce",
        )

        if numeric_profit.notna().any():
            highest = table.loc[numeric_profit.idxmax()]

            answer += (
                f"\n\n**Highest-profit "
                f"{group_by.replace('_', ' ')}:** "
                f"{highest[group_by]} "
                f"(total profit: ${numeric_profit.max():,.2f})"
            )

    return answer


def direct_time_comparison_answer(question: str, df: pd.DataFrame):
    """
    Directly calculate highest/lowest monthly or yearly sales.

    This handles these question types without calling Sarvam:
    - Which year had the highest total sales?
    - Which year had the lowest total sales?
    - Which month had the highest sales in 2014?
    - Which month had the lowest sales in 2014?

    Returns an answer string for supported questions, or None if the
    question should be handled by the normal analytics pipeline.
    """

    q = question.lower()

    asks_sales = bool(re.search(r"\bsales?\b", q))

    asks_highest = bool(
        re.search(
            r"\b(highest|highest-selling|most|maximum|max|top)\b",
            q,
        )
    )

    asks_lowest = bool(
        re.search(
            r"\b(lowest|least|minimum|min)\b",
            q,
        )
    )

    # Only handle questions that ask for one extreme.
    if not asks_sales or asks_highest == asks_lowest:
        return None

    asks_month = bool(re.search(r"\bmonths?\b", q))
    asks_year = bool(re.search(r"\byears?\b", q))

    # A month question takes priority over the year mentioned as a filter.
    if asks_month:
        period_type = "month"
    elif asks_year:
        period_type = "year"
    else:
        return None

    work = df[["order_date", "sales"]].copy()

    work["order_date"] = pd.to_datetime(
        work["order_date"],
        errors="coerce",
    )

    work["sales"] = pd.to_numeric(
        work["sales"],
        errors="coerce",
    )

    work = work.dropna(subset=["order_date", "sales"])

    if work.empty:
        return None

    # For monthly questions, optionally restrict calculations to the
    # specific year mentioned by the user.
    if period_type == "month":
        year_match = re.search(r"\b(20\d{2})\b", q)

        if year_match:
            requested_year = int(year_match.group(1))
            work = work[
                work["order_date"].dt.year == requested_year
            ]

            if work.empty:
                return (
                    f"No sales records were found for {requested_year}."
                )

        work["period"] = (
            work["order_date"].dt.to_period("M").astype(str)
        )

        summary_label = "monthly sales"
        period_label = "month"

    else:
        work["period"] = work["order_date"].dt.year.astype(str)

        summary_label = "annual sales"
        period_label = "year"

    totals = (
        work.groupby("period", as_index=False)["sales"]
        .sum()
        .sort_values("period")
        .reset_index(drop=True)
    )

    if totals.empty:
        return None

    if asks_highest:
        winner_index = totals["sales"].idxmax()
        extreme_label = "highest"
    else:
        winner_index = totals["sales"].idxmin()
        extreme_label = "lowest"

    winner = totals.loc[winner_index]

    period_value = str(winner["period"])
    sales_value = float(winner["sales"])

    if period_type == "month":
        parsed_period = pd.to_datetime(
            period_value,
            errors="coerce",
        )

        if pd.notna(parsed_period):
            readable_period = parsed_period.strftime("%B %Y")
        else:
            readable_period = period_value
    else:
        readable_period = period_value

    # Preserve the existing behavior: whole numbers have no decimal
    # places; values containing cents display two decimal places.
    if sales_value.is_integer():
        formatted_sales = f"{sales_value:,.0f}"
    else:
        formatted_sales = f"{sales_value:,.2f}"

    # Format the comparison table for easier reading.
    display_table = totals.copy()

    display_table["sales"] = display_table["sales"].map(
        lambda value: (
            f"{value:,.0f}"
            if float(value).is_integer()
            else f"{value:,.2f}"
        )
    )

    display_table = display_table.rename(
        columns={
            "period": period_label,
            "sales": "total_sales",
        }
    )

    table_markdown = display_table.to_markdown(index=False)

    if period_type == "month":
        answer = (
            f"**{readable_period} had the {extreme_label} sales, "
            f"at {formatted_sales}.**"
        )
    else:
        answer = (
            f"**{readable_period} had the {extreme_label} total sales "
            f"of {formatted_sales}.**"
        )

    return (
        answer
        + "\n\n### Exact computed results\n\n"
        + f"**{summary_label.title()} by {period_label}**\n\n"
        + table_markdown
        + f"\n\nRows used: {len(work):,}"
    )


def natural_answer(
    question: str,
    result: dict,
    model: str,
    demo: bool,
) -> str:
    """Explain the result while preserving deterministic calculations."""

    exact = result_as_text(result)

    # Retain support for monthly questions when the normal analytics
    # engine has already returned a monthly trend.
    q = question.lower()

    asks_month = bool(re.search(r"\bmonths?\b", q))
    asks_sales = bool(re.search(r"\bsales?\b", q))

    asks_highest = bool(
        re.search(
            r"\b(highest|highest-selling|most|maximum|max|top)\b",
            q,
        )
    )

    asks_lowest = bool(
        re.search(
            r"\b(lowest|least|minimum|min)\b",
            q,
        )
    )

    if asks_month and asks_sales and asks_highest != asks_lowest:
        table = pd.DataFrame(result.get("table", []))

        if (
            not table.empty
            and "month" in table.columns
            and "sales" in table.columns
        ):
            table["sales"] = pd.to_numeric(
                table["sales"],
                errors="coerce",
            )

            table = table.dropna(subset=["sales"])

            if not table.empty:
                if asks_highest:
                    winner = table.loc[table["sales"].idxmax()]
                    direction = "highest"
                else:
                    winner = table.loc[table["sales"].idxmin()]
                    direction = "lowest"

                month_value = str(winner["month"])
                month_date = pd.to_datetime(
                    month_value,
                    errors="coerce",
                )

                readable_month = (
                    month_date.strftime("%B %Y")
                    if pd.notna(month_date)
                    else month_value
                )

                sales_value = float(winner["sales"])

                if sales_value.is_integer():
                    formatted_sales = f"{sales_value:,.0f}"
                else:
                    formatted_sales = f"{sales_value:,.2f}"

                return (
                    f"**{readable_month} had the {direction} sales, "
                    f"at {formatted_sales}.**\n\n"
                    "---\n\n"
                    "### Exact computed results\n\n"
                    + exact
                )

    if demo:
        return (
            exact
            + "\n\n*Demo mode: deterministic local analytics; "
            "Sarvam explanation is not enabled.*"
        )

    messages = [
        {
            "role": "system",
            "content": (
                "You are a careful sales analytics assistant. "
                "Explain only the supplied computed result. "
                "Never invent or change numbers. "
                "Be concise and use at most two sentences. "
                "Do not reproduce the full table."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Question: {question}\n"
                f"Computed result summary: {result.get('summary', '')}\n"
                f"Rows analysed: {result.get('rows_used', 0)}\n"
                "Briefly explain the result in clear English."
            ),
        },
    ]

    try:
        explanation = chat_completion(
            messages,
            model=model,
            temperature=0.2,
        )

        if (
            not isinstance(explanation, str)
            or not explanation.strip()
        ):
            explanation = (
                "The analytics calculation completed successfully."
            )

        return (
            explanation.strip()
            + "\n\n---\n\n"
            + "### Exact computed results\n\n"
            + exact
        )

    except SarvamError as exc:
        return (
            "The analytics calculation completed successfully, "
            "but the AI explanation was unavailable.\n\n"
            + exact
            + f"\n\n*Sarvam explanation unavailable: {str(exc)}*"
        )


st.title("📊 Sales & Profitability Analytics Chatbot")

st.caption(
    "Fresh standalone app • Superstore sales dataset • "
    "Pandas + SQLite • Sarvam AI"
)

st.caption(
    "Build: deterministic-question-router-2026-10-18-v1"
)


with st.sidebar:
    st.header("⚙️ Settings")

    model = st.selectbox(
        "Choose model",
        [DEFAULT_MODEL, "sarvam-105b"],
        index=0,
    )

    demo = st.toggle(
        "Demo mode (no API call)",
        value=True,
    )

    st.markdown("---")

    st.write("**Architecture**")

    st.caption(
        "Question → Validators → Direct calculations or JSON plan → "
        "Deterministic analytics → Answer"
    )

    st.markdown("---")

    st.write("Add your API key to `.env` to enable Sarvam calls.")


try:
    modified = DATA_PATH.stat().st_mtime
    df = load_dataset(str(DATA_PATH), modified)

except FileNotFoundError:
    st.error(f"Dataset file not found at: {DATA_PATH}")
    st.stop()

except Exception as exc:
    st.error(str(exc))
    st.stop()


with st.expander("📂 Connected Dataset", expanded=True):
    col1, col2, col3 = st.columns(3)

    col1.metric("Rows", f"{len(df):,}")
    col2.metric("Columns", f"{len(df.columns):,}")
    col3.metric(
        "Distinct orders",
        f"{df['order_id'].nunique():,}",
    )

    st.caption(f"Loaded file: {DATA_PATH}")

    st.dataframe(df.head(8), width="stretch")


try:
    connection = get_sqlite_connection(
        str(DATA_PATH),
        modified,
    )

except Exception as exc:
    st.warning(
        f"SQLite analytics connection could not be initialized: {exc}"
    )


if "messages" not in st.session_state:
    st.session_state.messages = []


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


question = st.chat_input(
    "Ask a question about sales, profit, products, regions…"
)


if question:
    st.session_state.messages.append(
        {
            "role": "user",
            "content": question,
        }
    )

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner(
            "Validating question and analysing sales data…"
        ):
            ok1, message1 = validator_1_dataset_related(question)

            if not ok1:
                answer = message1

            else:
                ok2, message2 = validator_2_answerable(question)

                if not ok2:
                    answer = message2

                else:
                    raw_plan = None

                    try:
                        # Route common analytics questions to deterministic
                        # Pandas calculations first. This covers totals,
                        # averages, rankings, grouped summaries, time trends,
                        # and comparisons between explicit years without an
                        # LLM planning call.
                        answer = try_deterministic_answer(question, df)

                        # Keep the earlier monthly/annual extreme handler as
                        # a compatibility fallback for its established cases.
                        if answer is None:
                            answer = direct_time_comparison_answer(
                                question,
                                df,
                            )

                        if answer is None:
                            # Ambiguous or less-common questions retain the
                            # existing Demo mode / Sarvam planning workflow.
                            if demo:
                                plan = demo_plan(question)
                            else:
                                plan, raw_plan = generate_analytics_plan(
                                    question,
                                    list(df.columns),
                                    model,
                                )

                            result = execute_plan(df, plan)

                            answer = natural_answer(
                                question,
                                result,
                                model,
                                demo,
                            )

                            with st.expander(
                                "View validated analytics plan"
                            ):
                                st.json(plan)

                            with st.expander(
                                "View exact computed result"
                            ):
                                st.json(result)

                    except (
                        PlanError,
                        SarvamError,
                        ValueError,
                        KeyError,
                    ) as exc:
                        answer = (
                            "Analysis could not be completed safely: "
                            f"{str(exc)}"
                        )

                        if not demo and raw_plan is not None:
                            answer += (
                                "\n\n**Debug — AI plan returned "
                                "for this question:**\n```text\n"
                                + str(raw_plan)[:2000]
                                + "\n```"
                            )

                    except Exception as exc:
                        answer = (
                            "Unexpected analysis error: "
                            f"{type(exc).__name__}: {str(exc)}"
                        )

            st.markdown(answer)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
        }
    )