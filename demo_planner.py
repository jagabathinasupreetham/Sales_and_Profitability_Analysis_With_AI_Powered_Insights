"""Deterministic demo-mode analytics plan generation."""

import re


def _extract_top_limit(question: str) -> int:
    """
    Extract the requested number of top items.

    Examples:
        "Show the top 3 products by sales." -> 3
        "Show only the top product by sales." -> 1
        "Show the top customer by profit." -> 1

    Defaults to 10 when no specific limit is requested.
    The supported maximum is 20.
    """
    q = question.casefold()

    # Handle explicit numeric requests such as "top 3 products".
    match = re.search(r"\b(?:top|first)\s+(\d+)\b", q)

    if match:
        requested_limit = int(match.group(1))
        return max(1, min(requested_limit, 20))

    # Handle singular requests such as "top product" or "top customer".
    singular_top_pattern = (
        r"\b(?:only\s+)?"
        r"(?:the\s+)?"
        r"top\s+"
        r"(?:product|customer|region|category|item)\b"
    )

    if re.search(singular_top_pattern, q):
        return 1

    return 10


def demo_plan(question: str) -> dict:
    """Create a deterministic analytics plan from a question."""
    q = question.casefold()

    category_terms = [
        "by category",
        "each category",
        "category-wise",
        "category wise",
    ]

    # Compare sales and profit by category.
    if (
        "sales" in q
        and "profit" in q
        and any(term in q for term in category_terms + ["compare"])
    ):
        return {
            "action": "group_sum",
            "metric": "profit",
            "metrics": ["sales", "profit"],
            "group_by": "category",
            "limit": 10,
            "direction": "desc",
            "filters": {},
        }

    # Monthly trends.
    if any(
        term in q
        for term in [
            "monthly",
            "month by month",
            "trend over time",
            "sales trend",
            "trend by month",
            "per month",
            "each month",
            "time series",
        ]
    ):
        year_match = re.search(r"\b(19\d{2}|20\d{2})\b", q)

        if (
            "shipping cost" in q
            or "shipping_cost" in q
            or "shipping" in q
        ):
            trend_metric = "shipping_cost"
        elif "profit" in q or "profitability" in q:
            trend_metric = "profit"
        elif "quantity" in q or "units sold" in q:
            trend_metric = "quantity"
        elif "discount" in q:
            trend_metric = "discount"
        else:
            trend_metric = "sales"

        filters = {}

        if year_match:
            filters["order_date"] = year_match.group(1)

        return {
            "action": "monthly_trend",
            "metric": trend_metric,
            "group_by": None,
            "limit": 12,
            "direction": "asc",
            "filters": filters,
        }

    # Products with negative or lowest profit.
    if any(
        term in q
        for term in [
            "negative profit",
            "lowest profit",
            "loss-making",
            "least profitable",
        ]
    ):
        return {
            "action": "profit_loss_products",
            "metric": "profit",
            "group_by": "product_name",
            "limit": 10,
            "direction": "asc",
            "filters": {},
        }

    # Average order value.
    if "average order value" in q or "avg order value" in q:
        return {
            "action": "average",
            "metric": "sales",
            "group_by": None,
            "limit": 10,
            "direction": "desc",
            "filters": {},
        }

    # Order count.
    if any(
        term in q
        for term in [
            "how many orders",
            "number of orders",
            "order count",
            "total orders",
        ]
    ):
        return {
            "action": "total",
            "metric": "orders",
            "group_by": None,
            "limit": 10,
            "direction": "desc",
            "filters": {},
        }

    # Customer count.
    if any(
        term in q
        for term in [
            "customer count",
            "number of customers",
            "unique customers",
        ]
    ):
        return {
            "action": "total",
            "metric": "customers",
            "group_by": None,
            "limit": 10,
            "direction": "desc",
            "filters": {},
        }

    # Profit by category.
    if "profit" in q and any(term in q for term in category_terms):
        return {
            "action": "group_sum",
            "metric": "profit",
            "group_by": "category",
            "limit": 10,
            "direction": "desc",
            "filters": {},
        }

    # Sales by category.
    if "sales" in q and any(term in q for term in category_terms):
        return {
            "action": "group_sum",
            "metric": "sales",
            "group_by": "category",
            "limit": 10,
            "direction": "desc",
            "filters": {},
        }

    # Top-N questions.
    if any(
        term in q
        for term in [
            "top",
            "highest",
            "most",
            "best-selling",
        ]
    ):
        if "profit" in q:
            metric = "profit"
        else:
            metric = "sales"

        if "customer" in q:
            group = "customer_name"
        elif "product" in q:
            group = "product_name"
        elif "region" in q:
            group = "region"
        else:
            group = "category"

        limit = _extract_top_limit(question)

        return {
            "action": "top_n",
            "metric": metric,
            "group_by": group,
            "limit": limit,
            "direction": "desc",
            "filters": {},
        }

    # Average and mean questions.
    if "average" in q or "mean" in q:
        if "profit" in q:
            metric = "profit"
        elif "discount" in q:
            metric = "discount"
        else:
            metric = "sales"

        return {
            "action": "average",
            "metric": metric,
            "group_by": None,
            "limit": 10,
            "direction": "desc",
            "filters": {},
        }

    # General count questions.
    if any(
        term in q
        for term in [
            "count",
            "how many",
            "number of rows",
        ]
    ):
        return {
            "action": "count",
            "metric": "sales",
            "group_by": None,
            "limit": 10,
            "direction": "desc",
            "filters": {},
        }

    # Default total calculation.
    if "profit" in q:
        metric = "profit"
    elif "quantity" in q:
        metric = "quantity"
    else:
        metric = "sales"

    return {
        "action": "total",
        "metric": metric,
        "group_by": None,
        "limit": 10,
        "direction": "desc",
        "filters": {},
    }