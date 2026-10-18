"""Deterministic analytics engine. The LLM never executes code or arbitrary SQL."""

import pandas as pd

METRIC_MAP = {
    "sales": "sales",
    "profit": "profit",
    "quantity": "quantity",
    "discount": "discount",
    "shipping_cost": "shipping_cost",
    "orders": "order_id",
    "customers": "customer_id",
}


def execute_plan(df: pd.DataFrame, plan: dict) -> dict:
    action = plan["action"]
    metric = plan["metric"]
    metrics = plan.get("metrics") or [metric]
    # Keep compatibility with plans created before multi-metric support.
    metrics = list(dict.fromkeys(metrics))
    if metric not in metrics:
        metrics.insert(0, metric)
    group_by = plan.get("group_by")
    limit = plan.get("limit", 10)
    direction = plan.get("direction", "desc")
    work = df.copy()

    filters = plan.get("filters") or {}
    if isinstance(filters, dict):
        for key, value in filters.items():
            if value in (None, ""):
                continue
            # A year requested against order_date is a calendar-year filter, not
            # an exact string match against full ISO dates.
            if key in {"order_date", "ship_date"} and isinstance(value, (str, int)) and str(value).strip().isdigit() and len(str(value).strip()) == 4:
                if key in work.columns:
                    dates = pd.to_datetime(work[key], errors="coerce")
                    work = work[dates.dt.year == int(str(value).strip())]
                continue
            if key not in work.columns:
                continue
            if isinstance(value, (str, int, float, bool)):
                work = work[work[key].astype(str).str.casefold() == str(value).casefold()]

    if metric not in METRIC_MAP:
        raise ValueError(f"Unsupported metric: {metric!r}")
    field = METRIC_MAP[metric]

    if action == "total":
        if metric in {"orders", "customers"}:
            value = int(work[field].nunique())
        else:
            value = float(work[field].sum())
        return {"summary": f"Total {metric.replace('_', ' ')}", "value": value, "rows_used": len(work)}

    if action == "average":
        if metric in {"orders", "customers"}:
            raise ValueError("Average is not supported for distinct orders/customers.")
        if metric == "sales":
            order_sales = work.groupby("order_id")["sales"].sum()
            value = float(order_sales.mean()) if len(order_sales) else 0.0
            summary = "Average order value"
        else:
            value = float(work[field].mean()) if len(work) else 0.0
            summary = f"Average {metric.replace('_', ' ')}"
        return {"summary": summary, "value": value, "rows_used": len(work)}

    if action == "count":
        return {"summary": "Number of rows", "value": int(len(work)), "rows_used": len(work)}

    supported_group_actions = {
        "group_sum", "group_average", "top_n", "monthly_trend",
        "profit_loss_products", "compare_groups",
    }
    if action not in supported_group_actions:
        raise ValueError(f"Action {action!r} is not implemented.")

    if action == "monthly_trend":
        if "order_date" not in work.columns:
            raise ValueError("The dataset has no order_date field.")
        work["month"] = pd.to_datetime(work["order_date"], errors="coerce").dt.to_period("M").astype(str)
        grouped = work.dropna(subset=["month"]).groupby("month", as_index=False)[field].sum()
        grouped = grouped.sort_values("month")
        return {"summary": f"Monthly {metric.replace('_', ' ')} trend", "table": grouped.to_dict("records"), "rows_used": len(work)}

    if action == "profit_loss_products":
        grouped = work.groupby("product_name", as_index=False)["profit"].sum().sort_values("profit")
        return {"summary": "Products with the lowest total profit", "table": grouped.head(limit).to_dict("records"), "rows_used": len(work)}

    if not group_by:
        group_by = "category"
    if group_by not in work.columns:
        raise ValueError(f"Group field {group_by!r} is not in the dataset.")

    # Multi-metric aggregation: numeric metrics such as sales and profit together.
    if len(metrics) > 1:
        if action not in {"group_sum", "top_n", "compare_groups"}:
            raise ValueError("Multiple metrics are supported only for grouped sum, top-N, and group comparison.")
        if any(m not in METRIC_MAP for m in metrics):
            raise ValueError("One or more requested metrics are unsupported.")
        if any(m in {"orders", "customers"} for m in metrics):
            raise ValueError("Multi-metric aggregation currently supports numeric sum metrics only; orders/customers are distinct counts.")
        grouped = work.groupby(group_by, as_index=False)[metrics].sum()
        grouped = grouped.rename(columns={m: f"total_{m}" for m in metrics})
        sort_column = f"total_{metric}"
        grouped = grouped.sort_values(sort_column, ascending=(direction == "asc"))
        if action == "top_n":
            grouped = grouped.head(limit)
        return {
            "summary": f"Comparison of {', '.join(metrics)} by {group_by}",
            "table": grouped.to_dict("records"),
            "rows_used": len(work),
            "group_by": group_by,
            "primary_metric": metric,
        }

    if metric in {"orders", "customers"}:
        grouped = work.groupby(group_by)[field].nunique().reset_index(name="value")
    elif action == "group_average":
        grouped = work.groupby(group_by)[field].mean().reset_index(name="value")
    else:
        grouped = work.groupby(group_by)[field].sum().reset_index(name="value")

    if action == "top_n":
        grouped = grouped.sort_values("value", ascending=(direction == "asc")).head(limit)
    elif action == "compare_groups":
        grouped = grouped.sort_values("value", ascending=False)
    else:
        grouped = grouped.sort_values("value", ascending=(direction == "asc"))
    return {
        "summary": f"{metric.replace('_', ' ')} by {group_by}",
        "table": grouped.to_dict("records"),
        "rows_used": len(work),
        "group_by": group_by,
        "primary_metric": metric,
    }
