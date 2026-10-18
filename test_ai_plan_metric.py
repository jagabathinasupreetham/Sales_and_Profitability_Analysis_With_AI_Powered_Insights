"""Regression tests for AI-generated plans that omit the primary metric."""
import json
from pathlib import Path

import pandas as pd
import pytest

from chatbot.json_parser import PlanError, parse_plan
from analytics.pandas_engine import execute_plan


def test_missing_primary_metric_is_inferred_from_question_for_comparison():
    raw = json.dumps({
        "action": "group_sum",
        "metrics": ["sales", "profit"],
        "group_by": "category",
        "limit": 10,
        "direction": "desc",
        "filters": {},
    })
    plan = parse_plan(raw, question="Compare sales and profit by category and show highest profit")
    assert plan["metric"] == "profit"
    assert plan["metrics"] == ["sales", "profit"]


def test_legacy_single_metric_plan_still_works():
    plan = parse_plan(json.dumps({
        "action": "group_sum", "metric": "sales", "group_by": "category"
    }))
    assert plan["metric"] == "sales"
    assert plan["metrics"] == ["sales"]


def test_missing_metric_and_metrics_is_rejected_clearly():
    with pytest.raises(PlanError, match="missing a valid 'metric'"):
        parse_plan(json.dumps({"action": "total"}))


def test_sales_and_profit_by_category_calculates_against_bundled_dataset():
    project_root = Path(__file__).resolve().parents[1]
    df = pd.read_csv(project_root / "data" / "cleaned_sales_data.csv")
    raw = json.dumps({
        "action": "group_sum",
        "metrics": ["sales", "profit"],
        "group_by": "category",
        "limit": 10,
        "direction": "desc",
        "filters": {},
    })
    plan = parse_plan(raw, question="Compare sales and profit by category")
    result = execute_plan(df, plan)
    assert result["rows_used"] == 51290
    assert len(result["table"]) == 3
    assert result["table"][0]["category"] == "Technology"
    assert result["table"][0]["total_profit"] == pytest.approx(663778.73318)
    assert result["table"][0]["total_sales"] == pytest.approx(4744557.49808)


def test_monthly_trend_normalizes_order_date_group_field():
    raw = json.dumps({
        "action": "monthly_trend", "metric": "sales", "group_by": "order_date",
        "limit": 12, "direction": "asc", "filters": {"order_date": "2014"},
    })
    plan = parse_plan(raw, question="Show the monthly sales trend for 2014")
    assert plan["action"] == "monthly_trend"
    assert plan["group_by"] is None
    assert plan["filters"]["order_date"] == "2014"


def test_monthly_sales_trend_filters_2014_to_twelve_months():
    project_root = Path(__file__).resolve().parents[1]
    df = pd.read_csv(project_root / "data" / "cleaned_sales_data.csv")
    plan = parse_plan(json.dumps({
        "action": "group_sum", "metric": "sales", "group_by": "month",
        "limit": 12, "direction": "asc", "filters": {},
    }), question="Show the monthly sales trend for 2014")
    result = execute_plan(df, plan)
    assert result["rows_used"] == 17531
    assert len(result["table"]) == 12
    assert result["table"][0]["month"] == "2014-01"
    assert result["table"][-1]["month"] == "2014-12"
    assert sum(row["sales"] for row in result["table"]) > 0


def test_recovers_monthly_trend_when_ai_returns_group_sum_by_order_date():
    """Regression: actual recurring error from AI mode."""
    raw = json.dumps({
        "action": "group_sum", "metric": "sales", "group_by": "order_date",
        "limit": 10, "direction": "desc", "filters": {},
    })
    plan = parse_plan(raw, question="Show the monthly sales trend for 2014")
    assert plan["action"] == "monthly_trend"
    assert plan["metric"] == "sales"
    assert plan["group_by"] is None
    assert plan["filters"]["order_date"] == "2014"


def test_recovers_monthly_profit_trend_when_ai_returns_wrong_action():
    raw = json.dumps({
        "action": "group_sum", "metric": "profit", "group_by": "order_date",
        "limit": 10, "direction": "desc", "filters": {},
    })
    plan = parse_plan(raw, question="Show the monthly profit trend for 2014")
    assert plan["action"] == "monthly_trend"
    assert plan["metric"] == "profit"
    assert plan["filters"]["order_date"] == "2014"


def test_does_not_rewrite_ordinary_date_grouping_without_monthly_intent():
    raw = json.dumps({
        "action": "group_sum", "metric": "sales", "group_by": "order_date",
        "limit": 10, "direction": "desc", "filters": {},
    })
    with pytest.raises(PlanError, match="Unsupported group field: 'order_date'"):
        parse_plan(raw, question="Show sales grouped by order date")


def test_recovered_monthly_trend_executes_against_dataset():
    project_root = Path(__file__).resolve().parents[1]
    df = pd.read_csv(project_root / "data" / "cleaned_sales_data.csv")
    raw = json.dumps({
        "action": "group_sum", "metric": "sales", "group_by": "order_date",
        "limit": 10, "direction": "desc", "filters": {},
    })
    plan = parse_plan(raw, question="Show the monthly sales trend for 2014")
    result = execute_plan(df, plan)
    assert result["rows_used"] == 17531
    assert len(result["table"]) == 12
    assert result["table"][0]["month"] == "2014-01"
    assert result["table"][-1]["month"] == "2014-12"


def test_recovers_monthly_trend_when_ai_returns_group_sum_by_month():
    raw = json.dumps({
        "action": "group_sum", "metric": "sales", "group_by": "month",
        "limit": 10, "direction": "desc", "filters": {},
    })
    plan = parse_plan(raw, question="Show the monthly sales trend for 2014")
    assert plan["action"] == "monthly_trend"
    assert plan["metric"] == "sales"
    assert plan["group_by"] is None
    assert plan["filters"]["order_date"] == "2014"


def test_recovers_case_insensitive_month_group_alias():
    raw = json.dumps({
        "action": "group_sum", "metric": "profit", "group_by": "Month",
        "limit": 10, "direction": "desc", "filters": {},
    })
    plan = parse_plan(raw, question="Show profit month-wise trend in 2014")
    assert plan["action"] == "monthly_trend"
    assert plan["metric"] == "profit"
    assert plan["group_by"] is None
    assert plan["filters"]["order_date"] == "2014"


def test_explicit_monthly_trend_action_normalizes_month_alias_without_question():
    raw = json.dumps({
        "action": "monthly_trend", "metric": "sales", "group_by": "month",
        "limit": 12, "direction": "asc", "filters": {},
    })
    plan = parse_plan(raw)
    assert plan["action"] == "monthly_trend"
    assert plan["group_by"] is None


def test_recovers_date_group_sum_with_year_even_if_prompt_omits_monthly_wording():
    raw = json.dumps({
        "action": "group_sum", "metric": "sales", "group_by": "order_date",
        "limit": 10, "direction": "desc", "filters": {},
    })
    plan = parse_plan(raw, question="Show sales by order date in 2014")
    assert plan["action"] == "monthly_trend"
    assert plan["group_by"] is None
    assert plan["filters"]["order_date"] == "2014"


def test_recovers_date_group_sum_when_year_filter_is_already_present():
    raw = json.dumps({
        "action": "group_sum", "metric": "sales", "group_by": "order_date",
        "limit": 10, "direction": "desc", "filters": {"order_date": "2014"},
    })
    plan = parse_plan(raw, question="Show sales for 2014")
    assert plan["action"] == "monthly_trend"
    assert plan["group_by"] is None
    assert plan["filters"]["order_date"] == "2014"


def test_monthly_trend_date_alias_with_case_and_whitespace_is_recovered():
    raw = json.dumps({
        "action": "group_sum", "metric": "sales", "group_by": " Order_Date ",
        "limit": 10, "direction": "desc", "filters": {},
    })
    plan = parse_plan(raw, question="Show the monthly sales trend for 2014")
    assert plan["action"] == "monthly_trend"
    assert plan["group_by"] is None
    assert plan["filters"]["order_date"] == "2014"


def test_monthly_trend_recovers_order_date_group_and_year_filter():
    raw = json.dumps({
        "action": "group_sum", "metric": "sales", "group_by": "order_date",
        "limit": 12, "direction": "asc", "filters": {},
    })
    plan = parse_plan(raw, question="Show the monthly sales trend for 2014")
    assert plan["action"] == "monthly_trend"
    assert plan["metric"] == "sales"
    assert plan["group_by"] is None
    assert plan["filters"]["order_date"] == "2014"


def test_monthly_trend_recovers_month_group_even_with_wrong_action():
    raw = json.dumps({
        "action": "group_sum", "metric": "sales", "group_by": "month",
        "limit": 12, "direction": "asc", "filters": {},
    })
    plan = parse_plan(raw, question="Show the monthly sales trend for 2014")
    assert plan["action"] == "monthly_trend"
    assert plan["group_by"] is None
    assert plan["filters"]["order_date"] == "2014"


def test_monthly_trend_recovers_even_if_model_returns_wrong_categorical_group():
    raw = json.dumps({
        "action": "group_sum", "metric": "sales", "group_by": "category",
        "limit": 12, "direction": "asc", "filters": {},
    })
    plan = parse_plan(raw, question="Show the monthly sales trend for 2014")
    assert plan["action"] == "monthly_trend"
    assert plan["group_by"] is None
    assert plan["filters"]["order_date"] == "2014"


def test_monthly_sales_trend_for_2014_returns_only_twelve_2014_months():
    project_root = Path(__file__).resolve().parents[1]
    df = pd.read_csv(project_root / "data" / "cleaned_sales_data.csv")
    df["order_date"] = pd.to_datetime(df["order_date"], errors="coerce")
    raw = json.dumps({
        "action": "group_sum", "metric": "sales", "group_by": "order_date",
        "limit": 12, "direction": "asc", "filters": {},
    })
    plan = parse_plan(raw, question="Show the monthly sales trend for 2014")
    result = execute_plan(df, plan)
    months = [row["month"] for row in result["table"]]
    assert len(months) == 12
    assert months[0] == "2014-01"
    assert months[-1] == "2014-12"
    assert all(month.startswith("2014-") for month in months)
    assert result["rows_used"] == int((df["order_date"].dt.year == 2014).sum())
    assert result["table"][0]["sales"] == pytest.approx(241268.56, abs=0.01)


def test_monthly_trend_year_filter_works_with_datetime_column():
    df = pd.DataFrame({
        "order_date": pd.to_datetime(["2013-01-10", "2014-01-10", "2014-02-10"]),
        "sales": [10.0, 20.0, 30.0],
    })
    result = execute_plan(df, {
        "action": "monthly_trend", "metric": "sales", "metrics": ["sales"],
        "group_by": None, "filters": {"order_date": "2014"},
        "limit": 12, "direction": "asc",
    })
    assert result["table"] == [
        {"month": "2014-01", "sales": 20.0},
        {"month": "2014-02", "sales": 30.0},
    ]
    assert result["rows_used"] == 2


def test_demo_plan_extracts_year_from_monthly_trend_question():
    from chatbot.demo_planner import demo_plan

    plan = demo_plan("Show the monthly sales trend for 2014")
    assert plan["action"] == "monthly_trend"
    assert plan["metric"] == "sales"
    assert plan["filters"] == {"order_date": "2014"}


def test_demo_plan_supports_monthly_profit_trend():
    from chatbot.demo_planner import demo_plan

    plan = demo_plan("Show the monthly profit trend for 2014")
    assert plan["action"] == "monthly_trend"
    assert plan["metric"] == "profit"
    assert plan["filters"] == {"order_date": "2014"}


def test_demo_monthly_sales_trend_executes_only_2014_rows():
    from chatbot.demo_planner import demo_plan

    project_root = Path(__file__).resolve().parents[1]
    df = pd.read_csv(project_root / "data" / "cleaned_sales_data.csv")
    df["order_date"] = pd.to_datetime(df["order_date"], errors="coerce")
    plan = demo_plan("Show the monthly sales trend for 2014")
    result = execute_plan(df, plan)
    assert len(result["table"]) == 12
    assert all(row["month"].startswith("2014-") for row in result["table"])
    assert result["rows_used"] == int((df["order_date"].dt.year == 2014).sum())
    assert result["table"][0]["sales"] == pytest.approx(241268.56, abs=0.01)


def test_ai_monthly_trend_plan_recovers_date_alias_before_group_validation():
    for group_by in ("order_date", "month", "Order_Date", " Month ", "category"):
        raw = json.dumps({
            "action": "group_sum", "metric": "sales", "group_by": group_by,
            "limit": 12, "direction": "asc", "filters": {},
        })
        plan = parse_plan(raw, question="Show the monthly sales trend for 2014")
        assert plan["action"] == "monthly_trend"
        assert plan["group_by"] is None
        assert plan["filters"]["order_date"] == "2014"
