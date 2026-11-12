import pandas as pd
from data_utils import *

RAW = "experiment10_raw_dataset_-_experiment10_raw_dataset.csv"

def _clean():
    return clean_data(pd.read_csv(RAW))

def test_validation():
    v = validate_data(pd.read_csv(RAW))
    assert v["rows"] == 185 and v["cols"] == 18 and v["duplicate_count"] == 4
    assert not v["missing_required"]

def test_clean_has_no_missing_core_values():
    df, _ = _clean()
    assert df[NUMERIC_COLS + ["Order ID", "Order Date", "Category", "Region"]].isna().sum().sum() == 0
    assert not df.duplicated().any()

def test_standardisation():
    df, _ = _clean()
    assert set(df["Segment"]) <= {"Consumer", "Corporate", "Home Office", "Unknown"}
    assert set(df["Category"]) <= {"Technology", "Furniture", "Office Supplies"}
    assert set(df["Region"]) <= {"East", "West", "South", "Central"}
    assert set(df["Ship Mode"]) <= {"Standard Class", "Second Class", "First Class", "Same Day", "Unknown"}

def test_value_ranges():
    df, _ = _clean()
    assert df["Sales"].max() < 100000 and df["Sales"].min() > 0
    assert df["Quantity"].between(1, 1000).all() and df["Discount"].between(0, 1).all()

def test_dates():
    df, _ = _clean()
    assert df["Order Date"].dtype.kind == "M"
    assert pd.Timestamp("2024-12-26") in set(df["Order Date"])

def test_kpis():
    df, _ = _clean()
    k = calculate_kpis(df)
    assert abs(k["profit_margin"] - k["total_profit"] / k["total_sales"] * 100) < 1e-9
    assert k["total_orders"] == df["Order ID"].nunique()

if __name__ == "__main__":  # run without pytest:  python test_data_utils.py
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print("PASS", name)
