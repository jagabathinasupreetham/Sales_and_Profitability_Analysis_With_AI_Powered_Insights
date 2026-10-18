"""Deterministic natural-language router for common Superstore analytics.

This module deliberately supports a broad, explicit set of intents without
executing generated code or SQL. It returns None when a question is ambiguous
or outside the implemented patterns so the existing demo/Sarvam planner can
handle it.
"""
from __future__ import annotations

import re
from typing import Optional

import pandas as pd


METRICS = {
    "sales": ("sales", "Sales"),
    "profit": ("profit", "Profit"),
    "quantity": ("quantity", "Quantity"),
    "discount": ("discount", "Discount"),
    "shipping_cost": ("shipping_cost", "Shipping cost"),
}


GROUP_ALIASES = {
    "category": ("category", "categories"),
    "sub-category": ("sub-category", "sub category", "subcategory"),
    "segment": ("segment", "customer segment"),
    "region": ("region", "regions"),
    "country": ("country", "countries"),
    "market": ("market", "markets"),
    "ship_mode": ("ship mode", "shipping mode", "ship_mode"),
    "order_priority": ("order priority", "priority", "order_priority"),
    "product_name": ("product", "products", "product name", "product_name"),
    "customer_name": ("customer", "customers", "customer name", "customer_name"),
    "state": ("state", "states"),
    "city": ("city", "cities"),
}


def _metric(question: str) -> Optional[str]:
    q = question.casefold().replace("_", " ")

    if re.search(r"\bshipping\s+costs?\b|\bshipping\b", q):
        return "shipping_cost"

    if re.search(r"\bprofit(?:ability)?\b|\bprofit margin\b", q):
        return "profit"

    if re.search(r"\bquantity\b|\bunits sold\b|\bnumber of units\b", q):
        return "quantity"

    if re.search(r"\bdiscounts?\b", q):
        return "discount"

    if re.search(r"\bsales?\b|\brevenue\b|\bturnover\b", q):
        return "sales"

    return None


def _group_field(question: str, df: pd.DataFrame) -> Optional[str]:
    q = question.casefold().replace("_", " ")

    # "Product category" refers to the category column, not product_name.
    # This explicit phrase prevents the shorter "product" alias from
    # capturing category-ranking questions.
    if (
        "category" in df.columns
        and re.search(r"\bproduct\s+categor(?:y|ies)\b", q)
    ):
        return "category"

    # More specific names must be tested before shorter overlapping aliases.
    for field, aliases in sorted(
        GROUP_ALIASES.items(),
        key=lambda item: -max(map(len, item[1])),
    ):
        if field not in df.columns:
            continue

        for alias in aliases:
            alias_q = alias.casefold().replace("_", " ")
            pattern = r"(?<!\w)" + re.escape(alias_q) + r"(?!\w)"

            if re.search(pattern, q):
                return field

    return None


def _requested_rank(question: str) -> Optional[int]:
    """Extract an explicitly requested ordinal rank, such as second-highest.

    Returns:
        The requested one-based rank, or None if no ordinal rank was found.
    """
    q = (question or "").casefold()

    ordinal_words = {
        "first": 1,
        "1st": 1,
        "second": 2,
        "2nd": 2,
        "third": 3,
        "3rd": 3,
        "fourth": 4,
        "4th": 4,
        "fifth": 5,
        "5th": 5,
        "sixth": 6,
        "6th": 6,
        "seventh": 7,
        "7th": 7,
        "eighth": 8,
        "8th": 8,
        "ninth": 9,
        "9th": 9,
        "tenth": 10,
        "10th": 10,
    }

    match = re.search(
        r"\b(first|1st|second|2nd|third|3rd|fourth|4th|"
        r"fifth|5th|sixth|6th|seventh|7th|eighth|8th|"
        r"ninth|9th|tenth|10th)"
        r"(?:\s*-\s*|\s+)?"
        r"(?:highest|lowest|largest|smallest|best|worst)\b",
        q,
    )

    if not match:
        return None

    return ordinal_words[match.group(1)]


def _ranking_limit(question: str) -> int:
    """Extract a ranking limit, defaulting to one for singular requests."""
    q = (question or "").casefold()

    # Explicit numeric requests take precedence.
    limit_match = re.search(
        r"\b(?:top|bottom|first|last)\s+(\d{1,3})\b",
        q,
    )

    if limit_match:
        requested_limit = int(limit_match.group(1))
        return max(1, min(requested_limit, 100))

    # Words indicating a single requested result.
    singular_pattern = (
        r"\b(?:only\s+)?"
        r"(?:the\s+)?"
        r"(?:top|bottom|highest|lowest|best|worst)\s+"
        r"(?:product|products|customer|customers|region|regions|"
        r"product\s+category|product\s+categories|category|categories|"
        r"item|items|record|records|result|results)"
        r"\b"
    )

    if re.search(singular_pattern, q):
        return 1

    # Natural phrasing often places the metric between the ranking word and
    # entity, e.g. "highest total sales for product category".
    if (
        re.search(r"\b(highest|lowest|best|worst)\b", q)
        and re.search(
            r"\b(product\s+categories?|categories?|products?|customers?|regions?|items?)\b",
            q,
        )
    ):
        return 1

    # A singular entity in a ranking question is also a one-result request.
    if re.search(r"\b(?:only\s+)?(?:the\s+)?(?:top|bottom)\b", q):
        if re.search(
            r"\b(?:product|customer|region|category|item)\b",
            q,
        ):
            return 1

    return 10


def _filtered_frame(question: str, df: pd.DataFrame) -> pd.DataFrame:
    """Apply only unambiguous year and exact categorical-value filters."""
    work = df.copy()
    years = re.findall(r"\b(?:19|20)\d{2}\b", question)

    # Two years are kept for period comparisons; other intents use the one
    # explicit year that is most likely the requested filter.
    if len(years) == 1 and "order_date" in work.columns:
        dates = pd.to_datetime(work["order_date"], errors="coerce")
        work = work.loc[dates.dt.year == int(years[0])]

    q = question.casefold()

    excluded = {
        "order_date",
        "ship_date",
        "sales",
        "profit",
        "quantity",
        "discount",
        "shipping_cost",
    }

    categorical = [
        c
        for c in (
            "category",
            "sub-category",
            "segment",
            "region",
            "country",
            "market",
            "ship_mode",
            "order_priority",
            "state",
            "city",
            "product_name",
            "customer_name",
        )
        if c in work.columns and c not in excluded
    ]

    # Match exact dataset values mentioned in the question, longest first.
    for col in categorical:
        values = work[col].dropna().astype(str).unique().tolist()

        for value in sorted(values, key=len, reverse=True):
            if len(value.strip()) < 2:
                continue

            pattern = r"(?<!\w)" + re.escape(value.casefold()) + r"(?!\w)"

            if re.search(pattern, q):
                work = work[
                    work[col].astype(str).str.casefold() == value.casefold()
                ]
                break

    return work


def _format_num(value: float, currency: bool = False) -> str:
    number = float(value)
    formatted = f"{number:,.2f}"
    return f"${formatted}" if currency else formatted


def _table_answer(
    title: str,
    table: pd.DataFrame,
    rows_used: int,
    intro: str = "",
) -> str:
    if table.empty:
        return f"{intro or title}\n\nNo matching records were found."

    heading = intro if intro else f"**{title}**"

    return (
        f"{heading}\n\n"
        "### Exact computed results\n\n"
        + table.to_markdown(index=False)
        + f"\n\nRows used: {rows_used:,}"
    )


def _ship_mode_overlap_note(work: pd.DataFrame) -> str:
    """Explain when orders occur under multiple shipping modes."""
    required_columns = {"order_id", "ship_mode"}

    if not required_columns.issubset(work.columns):
        return ""

    valid_orders = work.dropna(subset=["order_id", "ship_mode"])

    if valid_orders.empty:
        return ""

    modes_per_order = (
        valid_orders.groupby("order_id")["ship_mode"].nunique()
    )

    overlapping_orders = int((modes_per_order > 1).sum())

    if overlapping_orders == 0:
        return ""

    return (
        "\n\n**Note:** "
        f"{overlapping_orders:,} orders appear under multiple shipping modes "
        "in this dataset. Therefore, the shipping-mode counts may add up "
        "to more than the overall number of distinct orders."
    )


def try_deterministic_answer(
    question: str,
    df: pd.DataFrame,
) -> Optional[str]:
    """Answer common analytics questions locally; return None if uncertain."""
    q = (question or "").strip().casefold()

    if not q:
        return None

    metric = _metric(q)

    # Preserve the existing multi-metric planner for requests that explicitly
    # ask to compare multiple measures in one answer.
    asks_sales_and_profit = (
        bool(re.search(r"\bsales?\b|\brevenue\b", q))
        and bool(re.search(r"\bprofit(?:ability)?\b", q))
    )

    if asks_sales_and_profit:
        group_field = _group_field(q, df)
        grouped_intent = bool(
            re.search(r"\b(by|per|each|every|wise|grouped|group|across)\b", q)
        )

        # Handle multi-metric grouped comparisons deterministically so the
        # planner cannot silently substitute a different grouping dimension.
        if group_field and grouped_intent:
            required = {group_field, "sales", "profit"}
            if not required.issubset(df.columns):
                return None

            work = _filtered_frame(q, df)
            if work.empty:
                return "No matching records were found for those filters."

            result = (
                work.groupby(group_field, dropna=False)
                .agg(total_sales=("sales", "sum"), total_profit=("profit", "sum"))
                .reset_index()
                .sort_values("total_sales", ascending=False, kind="stable")
            )
            result["total_sales"] = result["total_sales"].map(
                lambda x: _format_num(x, currency=True)
            )
            result["total_profit"] = result["total_profit"].map(
                lambda x: _format_num(x, currency=True)
            )
            return _table_answer(
                f"Sales and profit by {group_field.replace('_', ' ')}",
                result,
                len(work),
            )

        # Do not attempt an ungrouped multi-metric answer through a single-metric route.
        if group_field:
            return None

    if metric is None:
        # Count distinct orders. If the user requests a breakdown, group by
        # the requested field instead of returning one overall count.
        asks_order_count = bool(
            re.search(
                r"\b(how many|number of|count|total)\b.*\borders?\b"
                r"|\borders?\s+count\b",
                q,
            )
        )

        if asks_order_count:
            if "order_id" not in df.columns:
                return None

            group_field = _group_field(q, df)

            # Only group when the question explicitly asks for a breakdown.
            grouped_intent = bool(
                re.search(
                    r"\b(by|per|each|every|wise|grouped|group|across)\b",
                    q,
                )
            )

            if group_field and grouped_intent:
                work = _filtered_frame(q, df)

                if group_field not in work.columns:
                    return None

                if work.empty:
                    return "No matching records were found for those filters."

                result = (
                    work.groupby(group_field, dropna=False)["order_id"]
                    .nunique()
                    .reset_index(name="distinct_orders")
                    .sort_values(
                        "distinct_orders",
                        ascending=False,
                        kind="stable",
                    )
                )

                answer = _table_answer(
                    f"Distinct orders by {group_field.replace('_', ' ')}",
                    result,
                    len(work),
                )

                # Add context only to shipping-mode breakdowns. The counts
                # themselves remain unchanged.
                if group_field == "ship_mode":
                    answer += _ship_mode_overlap_note(work)

                return answer

            # Preserve the existing behavior for questions asking for one
            # overall distinct-order count.
            work = _filtered_frame(q, df)

            return (
                f"**Distinct orders: {work['order_id'].nunique():,}**"
                f"\n\nRows used: {len(work):,}"
            )

        if re.search(
            r"\b(how many|number of|count|total|unique)\b.*\bcustomers?\b",
            q,
        ):
            if "customer_id" not in df.columns:
                return None

            work = _filtered_frame(q, df)
            group_field = _group_field(q, df)

            grouped_intent = bool(
                re.search(
                    r"\b(by|per|each|every|wise|grouped|group|across)\b",
                    q,
                )
            )

            if group_field and grouped_intent:
                result = (
                    work.groupby(group_field, dropna=False)["customer_id"]
                    .nunique()
                    .reset_index(name="distinct_customers")
                    .sort_values(
                        "distinct_customers",
                        ascending=False,
                        kind="stable",
                    )
                )

                return _table_answer(
                    f"Distinct customers by {group_field.replace('_', ' ')}",
                    result,
                    len(work),
                )

            return (
                f"**Distinct customers: {work['customer_id'].nunique():,}**"
                f"\n\nRows used: {len(work):,}"
            )

        if re.search(
            r"\b(how many|count|number of)\b.*\b(rows|records|transactions)\b",
            q,
        ):
            matching = _filtered_frame(q, df)
            return (
                f"**Matching rows: {len(matching):,}**"
                f"\n\nRows used: {len(matching):,}"
            )

        return None

    field, label = METRICS[metric]

    if field not in df.columns:
        return None

    # Comparisons between two explicit calendar years, including absolute
    # and percentage changes. Calculate directly from the dataset, never the LLM.
    years = re.findall(r"\b(?:19|20)\d{2}\b", q)

    asks_compare = bool(
        re.search(
            r"\b(compare|comparison|versus|vs\.?|difference|change|"
            r"increase|decrease|growth)\b",
            q,
        )
    )

    if (
        asks_compare
        and len(set(years)) == 2
        and "order_date" in df.columns
    ):
        year_a, year_b = sorted({int(y) for y in years})
        dates = pd.to_datetime(df["order_date"], errors="coerce")
        values = pd.to_numeric(df[field], errors="coerce")
        totals = {}

        comparison_mask = dates.dt.year.isin([year_a, year_b])

        for year in (year_a, year_b):
            mask = dates.dt.year == year
            totals[year] = float(values.loc[mask].sum())

        delta = totals[year_b] - totals[year_a]
        pct = (
            delta / totals[year_a] * 100.0
            if totals[year_a]
            else None
        )

        table = pd.DataFrame([
            {"year": year_a, f"total_{metric}": totals[year_a]},
            {"year": year_b, f"total_{metric}": totals[year_b]},
        ])

        table[f"total_{metric}"] = table[f"total_{metric}"].map(
            lambda x: _format_num(
                x,
                metric in {"sales", "profit", "shipping_cost"},
            )
        )

        change_text = (
            f"Change from {year_a} to {year_b}: "
            f"**{_format_num(delta, metric in {'sales', 'profit', 'shipping_cost'})}**."
        )

        if pct is not None:
            change_text += f" Percentage change: **{pct:+.2f}%**."

        return _table_answer(
            f"{label} comparison",
            table,
            int(comparison_mask.sum()),
            f"{change_text}\n\n{label} totals for {year_a} and {year_b}",
        )

    work = _filtered_frame(q, df).copy()
    work[field] = pd.to_numeric(work[field], errors="coerce")
    work = work.dropna(subset=[field])

    if work.empty:
        return "No matching records were found for those filters."

    # Highest/lowest month or year (year in a month question acts as a filter).
    high = bool(re.search(r"\b(highest|maximum|max|most|top|best)\b", q))
    low = bool(re.search(r"\b(lowest|minimum|min|least|worst|bottom)\b", q))

    month_intent = bool(re.search(r"\b(month|monthly|months)\b", q))
    year_intent = bool(re.search(r"\b(year|yearly|annual|annually)\b", q))

    if (
        high != low
        and (month_intent or year_intent)
        and "order_date" in work.columns
    ):
        dates = pd.to_datetime(work["order_date"], errors="coerce")
        work = work.assign(_date=dates).dropna(subset=["_date"])

        if month_intent:
            work["period"] = work["_date"].dt.to_period("M").astype(str)
            period_name = "month"
        else:
            work["period"] = work["_date"].dt.year.astype(str)
            period_name = "year"

        totals = (
            work.groupby("period", as_index=False)[field]
            .sum()
            .sort_values("period")
        )

        winner = totals.loc[
            totals[field].idxmax() if high else totals[field].idxmin()
        ]

        period_value = str(winner["period"])

        if month_intent:
            parsed = pd.to_datetime(period_value, errors="coerce")
            period_value = (
                parsed.strftime("%B %Y")
                if pd.notna(parsed)
                else period_value
            )

        direction = "highest" if high else "lowest"
        currency = metric in {"sales", "profit", "shipping_cost"}

        display = totals.rename(
            columns={
                "period": period_name,
                field: f"total_{metric}",
            }
        ).copy()

        display[f"total_{metric}"] = display[f"total_{metric}"].map(
            lambda x: _format_num(x, currency)
        )

        return _table_answer(
            f"{label} by {period_name}",
            display,
            len(work),
            f"**{period_value} had the {direction} {label.lower()}: "
            f"{_format_num(winner[field], currency)}.**",
        )

    # Monthly/yearly trend tables.
    trend = bool(
        re.search(
            r"\b(trend|over time|time series|month by month|year by year|"
            r"per month|per year|each month|each year|monthly|yearly|annual)\b",
            q,
        )
    )

    if trend and "order_date" in work.columns:
        dates = pd.to_datetime(work["order_date"], errors="coerce")
        work = work.assign(_date=dates).dropna(subset=["_date"])

        if (
            month_intent
            or re.search(
                r"\bmonthly\b|\bper month\b|\beach month\b|\bmonth by month\b",
                q,
            )
        ):
            work["period"] = work["_date"].dt.to_period("M").astype(str)
        else:
            work["period"] = work["_date"].dt.year.astype(str)

        totals = (
            work.groupby("period", as_index=False)[field]
            .sum()
            .sort_values("period")
        )

        display = totals.rename(
            columns={"period": "period", field: f"total_{metric}"}
        )

        display[f"total_{metric}"] = display[f"total_{metric}"].map(
            lambda x: _format_num(
                x,
                metric in {"sales", "profit", "shipping_cost"},
            )
        )

        return _table_answer(f"{label} trend", display, len(work))

    # Grouped aggregations and rankings.
    group_field = _group_field(q, df)

    grouped_intent = bool(
        re.search(r"\b(by|per|each|every|wise|grouped|group|across)\b", q)
        or high
        or low
    )

    if group_field and grouped_intent:
        if group_field not in work.columns:
            return None

        grouped = work.groupby(group_field, dropna=False)[field]

        if re.search(r"\baverage|mean\b", q):
            result = grouped.mean().reset_index(name=f"average_{metric}")
            value_col = f"average_{metric}"
            operation = "average"
        else:
            result = grouped.sum().reset_index(name=f"total_{metric}")
            value_col = f"total_{metric}"
            operation = "total"

        result = result.sort_values(
            value_col,
            ascending=low,
            kind="stable",
        )

        if high or low:
            requested_rank = _requested_rank(q)

            if requested_rank is not None:
                # Ordinal requests select exactly one row at the requested
                # position. Rank 2 means the second row after sorting.
                result = result.iloc[
                    requested_rank - 1:requested_rank
                ]
            else:
                limit = _ranking_limit(q)
                result = result.head(limit)

        currency = metric in {"sales", "profit", "shipping_cost"}

        result[value_col] = result[value_col].map(
            lambda x: _format_num(x, currency)
        )

        title = (
            f"{operation.title()} {label.lower()} by "
            f"{group_field.replace('_', ' ')}"
        )

        if high != low:
            if _requested_rank(q) is not None and not result.empty:
                title = (
                    f"Ranked {label.lower()} by "
                    f"{group_field.replace('_', ' ')}"
                )
            else:
                title = (
                    f"{'Top' if high else 'Bottom'} {len(result)} "
                    f"by {label.lower()} and {group_field.replace('_', ' ')}"
                )

        return _table_answer(title, result, len(work))

    # Scalar totals and averages. Distinct order/customer counts were handled above.
    if re.search(r"\b(total|sum|overall|how much|amount of|combined)\b", q):
        value = float(work[field].sum())
        currency = metric in {"sales", "profit", "shipping_cost"}

        return (
            f"**Total {label.lower()}: {_format_num(value, currency)}**"
            f"\n\nRows used: {len(work):,}"
        )

    if re.search(r"\b(average|mean|avg)\b", q):
        # For sales, explicitly use average order value when the user says so;
        # otherwise report mean row-level sales to avoid silently changing meaning.
        if (
            metric == "sales"
            and re.search(r"\b(order value|per order)\b", q)
            and "order_id" in work.columns
        ):
            value = float(work.groupby("order_id")[field].sum().mean())
            name = "Average order value"
        else:
            value = float(work[field].mean())
            name = f"Average {label.lower()} per record"

        return (
            f"**{name}: "
            f"{_format_num(value, metric in {'sales', 'profit', 'shipping_cost'})}**"
            f"\n\nRows used: {len(work):,}"
        )

    return None