from pathlib import Path

import pandas as pd
import pytest

from analytics.question_router import try_deterministic_answer


@pytest.fixture(scope="module")
def sales_df():
    root = Path(__file__).resolve().parents[1]
    df = pd.read_csv(root / "data" / "cleaned_sales_data.csv")
    df["order_date"] = pd.to_datetime(df["order_date"], errors="coerce")
    return df


def test_compare_years_returns_exact_change_and_percentage(sales_df):
    answer = try_deterministic_answer(
        "Compare total sales between 2013 and 2014.", sales_df
    )
    assert answer is not None
    assert "894,119.42" in answer
    assert "+26.25%" in answer
    assert "2013" in answer and "2014" in answer


def test_highest_year_sales(sales_df):
    answer = try_deterministic_answer(
        "Which year had the highest total sales?", sales_df
    )
    assert answer is not None
    assert "2014 had the highest sales" in answer
    assert "4,299,865.87" in answer


def test_lowest_year_sales(sales_df):
    answer = try_deterministic_answer(
        "Which year had the lowest total sales?", sales_df
    )
    assert answer is not None
    assert "2011 had the lowest sales" in answer
    assert "2,259,450.90" in answer


def test_highest_month_sales_for_specific_year(sales_df):
    answer = try_deterministic_answer(
        "Which month had the highest sales in 2013?", sales_df
    )
    assert answer is not None
    assert "December 2013" in answer
    assert "405,454.38" in answer


def test_total_sales_for_year(sales_df):
    answer = try_deterministic_answer("What were total sales in 2014?", sales_df)
    assert answer is not None
    assert "Total sales" in answer
    assert "4,299,865.87" in answer


def test_sales_by_category(sales_df):
    answer = try_deterministic_answer("Show total sales by category", sales_df)
    assert answer is not None
    assert "Technology" in answer
    assert "Office Supplies" in answer
    assert "Furniture" in answer


def test_top_three_products_by_profit(sales_df):
    answer = try_deterministic_answer(
        "Show the top 3 products by profit", sales_df
    )
    assert answer is not None
    assert "Top 3" in answer
    assert "total_profit" in answer


def test_top_product_by_sales(sales_df):
    answer = try_deterministic_answer(
        "Show the top product by sales", sales_df
    )

    assert answer is not None
    assert "Apple Smart Phone, Full Size" in answer
    assert "86,935.78" in answer
    assert "Cisco Smart Phone, Full Size" not in answer


def test_second_highest_product_by_sales(sales_df):
    answer = try_deterministic_answer(
        "Show the second-highest product by sales", sales_df
    )

    assert answer is not None
    assert "Cisco Smart Phone, Full Size" in answer
    assert "76,441.53" in answer
    assert "Apple Smart Phone, Full Size" not in answer


def test_third_highest_product_by_sales(sales_df):
    answer = try_deterministic_answer(
        "Show the third-highest product by sales", sales_df
    )

    assert answer is not None
    assert "Motorola Smart Phone, Full Size" in answer
    assert "73,156.30" in answer
    assert "Apple Smart Phone, Full Size" not in answer
    assert "Cisco Smart Phone, Full Size" not in answer


def test_top_three_products_by_sales(sales_df):
    answer = try_deterministic_answer(
        "Show the top 3 products by sales", sales_df
    )

    assert answer is not None
    assert "Apple Smart Phone, Full Size" in answer
    assert "Cisco Smart Phone, Full Size" in answer
    assert "Motorola Smart Phone, Full Size" in answer


def test_second_lowest_product_by_sales(sales_df):
    answer = try_deterministic_answer(
        "Show the second-lowest product by sales", sales_df
    )

    assert answer is not None

    expected = (
        sales_df.groupby("product_name")["sales"]
        .sum()
        .sort_values(ascending=True)
        .iloc[1]
    )
    expected_product = (
        sales_df.groupby("product_name")["sales"]
        .sum()
        .sort_values(ascending=True)
        .index[1]
    )

    assert expected_product in answer
    assert f"{expected:,.2f}" in answer


def test_monthly_profit_trend_for_year(sales_df):
    answer = try_deterministic_answer(
        "Show the monthly profit trend for 2014", sales_df
    )
    assert answer is not None
    assert "2014-01" in answer and "2014-12" in answer
    assert "2013-" not in answer


def test_multi_metric_question_returns_sales_and_profit_by_category(sales_df):
    answer = try_deterministic_answer(
        "Compare sales and profit by category", sales_df
    )
    assert answer is not None
    assert "Sales and profit by category" in answer
    assert "Technology" in answer
    assert "total_sales" in answer
    assert "total_profit" in answer


def test_product_category_ranking_groups_by_category_not_product(sales_df):
    answer = try_deterministic_answer(
        "Which product category generates the highest total sales?", sales_df
    )
    assert answer is not None
    assert "category" in answer.casefold()
    assert "Technology" in answer
    assert "Apple Smart Phone, Full Size" not in answer


def test_multi_metric_question_by_region_uses_region_grouping(sales_df):
    answer = try_deterministic_answer(
        "Compare total sales and total profit for each region", sales_df
    )
    assert answer is not None
    assert "Sales and profit by region" in answer
    assert "Central" in answer
    assert "North" in answer
    assert "total_sales" in answer
    assert "total_profit" in answer


def test_ambiguous_question_falls_back(sales_df):
    answer = try_deterministic_answer("Tell me something interesting", sales_df)
    assert answer is None