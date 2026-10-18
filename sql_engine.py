"""SQLite-backed dataset preview and basic SQL analytics helpers."""
import sqlite3
import pandas as pd

def dataframe_to_sqlite(df: pd.DataFrame):
    connection = sqlite3.connect(":memory:", check_same_thread=False)
    df.to_sql("sales", connection, index=False, if_exists="replace")
    return connection

def run_readonly_query(connection, query: str):
    """Only allow a single SELECT statement; intended for developer diagnostics."""
    stripped = query.strip()
    if not stripped.lower().startswith("select") or ";" in stripped.rstrip(";"):
        raise ValueError("Only one SELECT statement is allowed.")
    return pd.read_sql_query(stripped, connection)
