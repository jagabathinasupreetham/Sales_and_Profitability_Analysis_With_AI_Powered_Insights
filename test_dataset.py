from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "cleaned_sales_data.csv"

def test_dataset_shape():
    df = pd.read_csv(DATA)
    assert df.shape == (51290, 23)

def test_required_columns():
    df = pd.read_csv(DATA, nrows=1)
    required = {"order_id", "order_date", "category", "product_name", "sales", "profit", "quantity"}
    assert required.issubset(set(df.columns))
