"""Strict JSON extraction and validation for the LLM-generated analytics plan."""

from __future__ import annotations
import json
import re

ALLOWED_ACTIONS = {
    "total", "average", "count", "group_sum", "group_average", "top_n",
    "monthly_trend", "profit_loss_products", "compare_groups",
}

ALLOWED_METRICS = {
    "sales", "profit", "quantity", "discount", "shipping_cost", "orders", "customers",
}

ALLOWED_GROUPS = {
    "category", "sub-category", "segment", "region", "country", "market",
    "ship_mode", "order_priority", "product_name", "customer_name", "state",
}


class PlanError(ValueError):
    """Raised when a model-generated analytics plan is invalid."""


def _extract_json_object(raw: str) -> dict:
    """Extract one complete JSON object without accepting arbitrary prose."""
    if not isinstance(raw, str) or not raw.strip():
        raise PlanError("The model returned an empty response.")

    text = raw.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].strip().lower() in {"```", "```json"}:
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()

    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass

    start = None
    depth = 0
    in_string = False
    escaped = False
    for index, char in enumerate(text):
        if start is None:
            if char == "{":
                start = index
                depth = 1
            continue

        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                candidate = text[start:index + 1]
                try:
                    obj = json.loads(candidate)
                except json.JSONDecodeError as exc:
                    raise PlanError(
                        f"The model returned malformed JSON: {exc.msg}."
                    ) from exc
                if not isinstance(obj, dict):
                    raise PlanError("The analytics plan must be a JSON object.")
                return obj

    if start is None:
        preview = " ".join(text.split())[:180]
        raise PlanError(
            "The model response contained no JSON object. "
            f"Response preview: {preview!r}"
        )
    raise PlanError("The model returned incomplete JSON.")


def _infer_primary_metric(question: str | None, metrics: list[str]) -> str:
    """Choose a safe primary metric when the model omits `metric` but supplies `metrics`.

    Explicit ranking/profitability language takes priority. Otherwise, preserve the
    model's declared metric order instead of inventing a new metric.
    """
    q = (question or "").casefold().replace("-", " ")
    if "profit" in metrics and any(term in q for term in (
        "highest profit", "most profit", "most profitable", "profitability",
        "lowest profit", "least profitable", "profit margin", "profit first",
    )):
        return "profit"
    if "sales" in metrics and any(term in q for term in (
        "highest sales", "most sales", "highest revenue", "most revenue",
        "sales first", "revenue first",
    )):
        return "sales"
    return metrics[0]



def _requested_monthly_metric(question: str | None) -> str:
    """Infer the metric for a clearly worded monthly-trend request."""
    q = (question or "").casefold().replace("-", " ")
    # Prefer a metric explicitly named by the user; sales is the conservative
    # default for the common phrase "monthly trend" in this sales dataset.
    for metric in ("shipping_cost", "profit", "quantity", "discount", "sales"):
        term = metric.replace("_", " ")
        if term in q:
            return metric
    if "revenue" in q:
        return "sales"
    return "sales"


def _normalize_monthly_trend_intent(
    question: str | None, action: str, metric: str | None,
    group_by: str | None, raw_metrics, filters,
):
    """Normalize a clearly requested monthly trend before strict plan validation.

    Do not rely on the model choosing a particular ``group_by`` value. In real
    responses it may return ``order_date``, ``month``, ``category``, or another
    value, and it may incorrectly label the action as ``group_sum``. When the
    user's wording clearly requests a monthly trend, the deterministic executor
    must derive month buckets from order_date itself.
    """
    q = (question or "").casefold().replace("-", " ").replace("_", " ")
    has_month_word = bool(re.search(r"\bmonths?\b|\bmonthly\b", q))
    has_trend_word = any(term in q for term in (
        "trend", "over time", "time series", "month wise", "monthwise",
        "month by month", "per month", "each month",
    ))
    has_year = bool(re.search(r"\b(19\d{2}|20\d{2})\b", q))
    # A date/month grouping plus a named metric and year is also strong intent,
    # e.g. "sales by order date in 2014".
    normalized_group = (
        group_by.strip().casefold().replace("-", " ").replace("_", " ")
        if isinstance(group_by, str) else group_by
    )
    date_grouped = normalized_group in {
        "order date", "orderdate", "month", "monthly", "date", "order month",
        "month of order date", "order date month",
    }
    named_metric = any(term in q for term in (
        "sales", "revenue", "profit", "quantity", "discount", "shipping cost", "shipping",
    ))
    clear_monthly_intent = has_month_word or has_trend_word
    should_recover = clear_monthly_intent or (has_year and date_grouped and named_metric)
    if not should_recover:
        return action, metric, group_by, raw_metrics, filters

    action = "monthly_trend"
    metric = _requested_monthly_metric(question)
    # A monthly trend is one series; discard malformed comparison metrics.
    raw_metrics = None
    group_by = None
    filters = dict(filters) if isinstance(filters, dict) else {}
    year_match = re.search(r"\b(19\d{2}|20\d{2})\b", question or "")
    if year_match:
        # The explicit year in the user's question takes precedence over a
        # malformed model date filter so results cannot silently span years.
        filters["order_date"] = year_match.group(1)
    return action, metric, group_by, raw_metrics, filters

def parse_plan(raw: str, question: str | None = None) -> dict:
    """Parse and strictly validate a plan, supporting old and multi-metric plans.

    Older plans use `metric`; comparison plans may also provide `metrics`. If a
    comparison plan includes a valid non-empty `metrics` list but omits `metric`,
    infer the primary metric from the question when possible, otherwise use the
    first declared metric. If neither field is present, reject the plan clearly.
    """
    obj = _extract_json_object(raw)
    action = obj.get("action")
    metric = obj.get("metric")
    group_by = obj.get("group_by")
    limit = obj.get("limit", 10)
    direction = obj.get("direction", "desc")
    filters = obj.get("filters", {})
    raw_metrics = obj.get("metrics")

    # Recover the real user intent before strict validation. Sarvam sometimes
    # labels a monthly-trend request as group_sum grouped by order_date/month.
    action, metric, group_by, raw_metrics, filters = _normalize_monthly_trend_intent(
        question, action, metric, group_by, raw_metrics, filters
    )

    # Last-chance guard immediately before group validation. Keep this separate
    # from the helper above intentionally: the recurring production error shows
    # some plans can arrive with date aliases that bypassed earlier normalization
    # (for example due to an unexpected spelling/casing or plan shape). A clear
    # monthly-trend question plus a date/month grouping must never be rejected as
    # an unsupported categorical field; the executor derives calendar months from
    # order_date itself.
    q_normalized = (question or "").casefold().replace("-", " ").replace("_", " ")
    group_normalized = (
        group_by.strip().casefold().replace("-", " ").replace("_", " ")
        if isinstance(group_by, str) else group_by
    )
    asks_monthly_trend = (
        bool(re.search(r"\bmonthly\b|\bmonth\s+by\s+month\b|\bmonthwise\b|\bmonth\s+wise\b", q_normalized))
        or ("trend" in q_normalized and bool(re.search(r"\bmonth(s)?\b", q_normalized)))
        or "time series" in q_normalized
    )
    if asks_monthly_trend and group_normalized in {
        "order date", "orderdate", "month", "monthly", "date", "order month",
        "month of order date", "order date month",
    }:
        action = "monthly_trend"
        metric = _requested_monthly_metric(question)
        raw_metrics = None
        group_by = None
        filters = dict(filters) if isinstance(filters, dict) else {}
        year_match = re.search(r"\b(19\d{2}|20\d{2})\b", question or "")
        if year_match and not any(key in filters for key in ("order_date", "ship_date")):
            filters["order_date"] = year_match.group(1)

    if action not in ALLOWED_ACTIONS:
        raise PlanError(f"Unsupported analytics action: {action!r}")

    if raw_metrics is None:
        if metric not in ALLOWED_METRICS:
            raise PlanError(
                "The plan is missing a valid 'metric'. Return 'metric' for a single-metric query, "
                "or provide both 'metric' and a non-empty 'metrics' list for a comparison."
            )
        metrics = [metric]
    else:
        if not isinstance(raw_metrics, list) or not raw_metrics:
            raise PlanError("Plan 'metrics' must be a non-empty list of allowed metrics.")
        if any(not isinstance(item, str) or item not in ALLOWED_METRICS for item in raw_metrics):
            raise PlanError("The plan contains an unsupported metric in 'metrics'.")
        metrics = list(dict.fromkeys(raw_metrics))
        if metric not in ALLOWED_METRICS:
            metric = _infer_primary_metric(question, metrics)
        elif metric not in metrics:
            metrics.insert(0, metric)

    if len(metrics) > 1 and action not in {"group_sum", "top_n", "compare_groups"}:
        raise PlanError(
            "Multiple metrics are supported only for grouped sum, top-N, and group comparison actions."
        )
    # Date-trend plans often use order_date as their grouping field because the
    # source data is date-grained. The executor creates month buckets itself, so
    # normalize this common model output instead of rejecting an otherwise valid
    # monthly_trend plan. Raw dates are not ordinary categorical group fields.
    if action == "monthly_trend" and group_by in {"order_date", "month", "monthly"}:
        group_by = None
    if group_by is not None and group_by not in ALLOWED_GROUPS:
        raise PlanError(f"Unsupported group field: {group_by!r}")
    if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 20:
        raise PlanError("Plan limit must be an integer between 1 and 20.")
    if direction not in {"asc", "desc"}:
        raise PlanError("Plan direction must be 'asc' or 'desc'.")
    if not isinstance(filters, dict):
        raise PlanError("Plan filters must be a JSON object.")

    return {
        "action": action,
        "metric": metric,
        "metrics": metrics,
        "group_by": group_by,
        "limit": limit,
        "direction": direction,
        "filters": filters,
    }
