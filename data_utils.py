"""Reusable data loading, validation, cleaning and statistics functions."""
import io
import json

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = [
    "Order ID", "Order Date", "Ship Date", "Ship Mode", "Customer ID", "Segment",
    "Region", "State", "City", "Category", "Sub-Category", "Product Name",
    "Sales", "Quantity", "Discount", "Profit",
]
NUMERIC_COLS = ["Sales", "Quantity", "Discount", "Profit"]
DATE_COLS = ["Order Date", "Ship Date"]

SHIP_MODE_MAP = {
    "standard class": "Standard Class", "second class": "Second Class",
    "first class": "First Class", "same day": "Same Day", "sameday": "Same Day",
}
SEGMENT_MAP = {
    "consumer": "Consumer", "corporate": "Corporate", "home office": "Home Office",
}
REGION_MAP = {"east": "East", "west": "West", "south": "South", "central": "Central"}


# ----------------------------------------------------------------------------
# 1. LOADING
# ----------------------------------------------------------------------------
def load_file(uploaded_file) -> pd.DataFrame:
    """Load CSV / XLSX / JSON from a file-like object that has a .name attribute."""
    name = uploaded_file.name.lower()
    if name.endswith(".csv"):
        return pd.read_csv(uploaded_file)
    if name.endswith((".xlsx", ".xls")):
        return pd.read_excel(uploaded_file)
    if name.endswith(".json"):
        raw = uploaded_file.read()
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        try:
            return pd.read_json(io.StringIO(raw))
        except ValueError:
            return pd.json_normalize(json.loads(raw))
    raise ValueError("Unsupported file type. Please upload .csv, .xlsx or .json")


# ----------------------------------------------------------------------------
# 2. VALIDATION
# ----------------------------------------------------------------------------
def validate_data(df: pd.DataFrame) -> dict:
    """Return dimensions, column info, missing values and duplicate info."""
    missing = df.isna().sum()
    info = pd.DataFrame({
        "Column": df.columns,
        "Data Type": df.dtypes.astype(str).values,
        "Missing Values": missing.values,
        "Missing %": (missing.values / max(len(df), 1) * 100).round(2),
        "Unique Values": df.nunique().values,
    })
    return {
        "rows": df.shape[0],
        "cols": df.shape[1],
        "column_info": info,
        "total_missing": int(missing.sum()),
        "duplicate_count": int(df.duplicated().sum()),
        "duplicate_rows": df[df.duplicated(keep=False)],
        "missing_required": [c for c in REQUIRED_COLUMNS if c not in df.columns],
    }


# ----------------------------------------------------------------------------
# 3. CLEANING HELPERS
# ----------------------------------------------------------------------------
def parse_dates(series: pd.Series) -> pd.Series:
    """Parse mixed date formats (2025-09-07, 26/12/2024, 21-Mar-2025, 03-12-2025)."""
    s = series.astype("string").str.strip()
    out = pd.Series(pd.NaT, index=s.index, dtype="datetime64[ns]")
    for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%d-%m-%Y", "%d-%b-%Y", "%m/%d/%Y", "%Y/%m/%d"):
        mask = out.isna() & s.notna()
        if not mask.any():
            break
        out[mask] = pd.to_datetime(s[mask], format=fmt, errors="coerce")
    return out


def standardize_text(series: pd.Series, mapping: dict) -> pd.Series:
    """Map inconsistent text spellings to canonical labels."""
    key = series.astype("string").str.strip().str.lower()
    return key.map(mapping).where(key.isin(mapping), series.astype("string").str.strip())


def to_numeric_safe(series: pd.Series) -> pd.Series:
    """Convert to numeric; invalid text ('unknown', '?', 'five') and inf become NaN."""
    return pd.to_numeric(series, errors="coerce").astype("float64").replace([np.inf, -np.inf], np.nan)


def _mode_map(df, key, value):
    """Most common `value` for every `key` (used to repair Category / Region)."""
    tmp = df.dropna(subset=[key, value])
    return tmp.groupby(key)[value].agg(lambda x: x.mode().iloc[0]).to_dict()


def clean_data(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Full preprocessing pipeline. Returns (clean_df, log_of_actions)."""
    log = []
    df = df.copy()
    df.columns = df.columns.str.strip()
    n0 = len(df)

    # --- duplicates --------------------------------------------------------
    df = df.drop_duplicates().reset_index(drop=True)
    log.append(f"Removed {n0 - len(df)} exact duplicate records.")

    # --- whitespace on text columns ---------------------------------------
    for c in df.select_dtypes(include=["object", "string"]).columns:
        df[c] = df[c].astype("string").str.strip()

    # --- dates ---------------------------------------------------------------
    for c in DATE_COLS:
        if c in df:
            before = df[c].isna().sum()
            df[c] = parse_dates(df[c])
            log.append(f"Parsed '{c}' from mixed formats; {df[c].isna().sum() - before} "
                       f"invalid values (e.g. 'not-a-date') set to missing.")

    # --- numeric columns -------------------------------------------------------
    for c in NUMERIC_COLS:
        if c in df:
            before = df[c].isna().sum()
            df[c] = to_numeric_safe(df[c])
            log.append(f"Converted '{c}' to numeric; {df[c].isna().sum() - before} "
                       f"non-numeric entries (text / inf) set to missing.")

    # --- impossible / outlier values -> NaN --------------------------------------
    rules = {
        "Sales": df["Sales"] <= 0 if "Sales" in df else None,
        "Quantity": (df["Quantity"] <= 0) | (df["Quantity"] > 1000) if "Quantity" in df else None,
        "Discount": (df["Discount"] < 0) | (df["Discount"] > 1) if "Discount" in df else None,
    }
    for c, bad in rules.items():
        if bad is not None:
            log.append(f"'{c}': {int(bad.sum())} impossible values (negative / zero / >limit) set to missing.")
            df.loc[bad, c] = np.nan
    if "Sales" in df:  # extreme outliers (e.g. 9,999,999.99 placeholder)
        q1, q3 = df["Sales"].quantile([0.25, 0.75])
        upper = q3 + 10 * (q3 - q1)
        bad = df["Sales"] > upper
        log.append(f"'Sales': {int(bad.sum())} extreme outliers (> {upper:,.0f}) set to missing.")
        df.loc[bad, "Sales"] = np.nan

    # --- inconsistent categorical data ---------------------------------------------
    if "Ship Mode" in df:
        df["Ship Mode"] = standardize_text(df["Ship Mode"], SHIP_MODE_MAP)
        df.loc[~df["Ship Mode"].isin(SHIP_MODE_MAP.values()), "Ship Mode"] = pd.NA
    if "Segment" in df:
        df["Segment"] = standardize_text(df["Segment"], SEGMENT_MAP)
        df.loc[~df["Segment"].isin(SEGMENT_MAP.values()), "Segment"] = "Unknown"
    if "Region" in df:
        df["Region"] = standardize_text(df["Region"], REGION_MAP)
    if "Category" in df and "Sub-Category" in df:
        cat_map = _mode_map(df.assign(Category=df["Category"].str.title()), "Sub-Category", "Category")
        valid = {"Technology", "Furniture", "Office Supplies"}
        # Derive Category from Sub-Category (fixes 'technology', 'Gadgets', 'Office supply', blanks)
        df["Category"] = df["Sub-Category"].map(
            {k: v for k, v in cat_map.items() if v in valid}).fillna(df["Category"].str.title())
        df.loc[~df["Category"].isin(valid), "Category"] = pd.NA
    if "State" in df and "Region" in df:  # repair region from state
        df["Region"] = df["State"].map(_mode_map(df, "State", "Region")).fillna(df["Region"])
    log.append("Standardised Ship Mode, Segment, Region and Category spellings/casing "
               "(e.g. 'SameDay'->'Same Day', 'consumer'->'Consumer', 'Gadgets' repaired from Sub-Category).")

    # --- drop unusable rows -----------------------------------------------------------
    n1 = len(df)
    df = df.dropna(subset=["Order ID", "Order Date"])
    if "State" in df:
        df = df[df["State"].str.lower() != "test state"]
    if "Product Name" in df:
        df = df[df["Product Name"].fillna("") != "Mystery Product"]
    df = df.reset_index(drop=True)
    log.append(f"Dropped {n1 - len(df)} unusable rows (missing Order ID / Order Date, test or placeholder records).")

    # --- missing values -> imputation -----------------------------------------------------
    keys = "Sub-Category" if "Sub-Category" in df else None
    for c in ["Sales", "Quantity", "Discount"]:
        if c in df and df[c].isna().any():
            n = int(df[c].isna().sum())
            med = df.groupby(keys)[c].transform("median") if keys else df[c].median()
            df[c] = df[c].fillna(med).fillna(df[c].median())
            log.append(f"Imputed {n} missing '{c}' values with the Sub-Category median.")
    if "Quantity" in df:
        df["Quantity"] = df["Quantity"].round().astype(int)
    if "Profit" in df and df["Profit"].isna().any():
        n = int(df["Profit"].isna().sum())
        margin = (df["Profit"] / df["Sales"])
        med_margin = margin.groupby(df[keys]).transform("median") if keys else margin.median()
        df["Profit"] = df["Profit"].fillna(df["Sales"] * med_margin.fillna(margin.median()))
        log.append(f"Imputed {n} missing 'Profit' values using Sub-Category median profit margin x Sales.")
    for c in ["Ship Mode", "City", "Customer ID", "Category", "Product Name"]:
        if c in df and df[c].isna().any():
            n = int(df[c].isna().sum())
            df[c] = df[c].fillna("Unknown")
            log.append(f"Filled {n} missing '{c}' values with 'Unknown'.")
    if {"Ship Date", "Order Date"} <= set(df.columns):
        bad = df["Ship Date"] < df["Order Date"]
        df.loc[bad, "Ship Date"] = pd.NaT  # shipping before ordering is impossible
        df["Shipping Days"] = (df["Ship Date"] - df["Order Date"]).dt.days
        log.append(f"Added 'Shipping Days'; {int(bad.sum())} Ship Dates earlier than Order Date set to missing.")

    # --- derived columns -----------------------------------------------------------------------
    df["Year"] = df["Order Date"].dt.year
    df["Month"] = df["Order Date"].dt.to_period("M").dt.to_timestamp()

    # --- duplicates created by cleaning ------------------------------------------------------------
    n2 = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    if n2 != len(df):
        log.append(f"Removed {n2 - len(df)} additional duplicates revealed after cleaning.")
    log.append(f"Final dataset: {df.shape[0]} rows x {df.shape[1]} columns.")
    return df, log


# ----------------------------------------------------------------------------
# 4. STATISTICS / KPIs
# ----------------------------------------------------------------------------
def summary_statistics(df: pd.DataFrame, cols=NUMERIC_COLS) -> pd.DataFrame:
    """Mean, median, mode, std, variance, min, max, quartiles, skew for numeric columns."""
    cols = [c for c in cols if c in df.columns]
    out = pd.DataFrame({
        "Mean": df[cols].mean(), "Median": df[cols].median(),
        "Mode": df[cols].mode().iloc[0], "Std Dev": df[cols].std(),
        "Variance": df[cols].var(), "Min": df[cols].min(),
        "Q1 (25%)": df[cols].quantile(0.25), "Q3 (75%)": df[cols].quantile(0.75),
        "Max": df[cols].max(), "Skewness": df[cols].skew(),
    })
    return out.round(2)


def categorical_summary(df: pd.DataFrame, col: str, top: int = 10) -> pd.DataFrame:
    """Frequency table for a categorical column."""
    vc = df[col].value_counts().head(top)
    return pd.DataFrame({col: vc.index, "Count": vc.values, "Share %": (vc.values / len(df) * 100).round(2)})


def calculate_kpis(df: pd.DataFrame) -> dict:
    """Five KPIs: Total Sales, Total Profit, Total Orders, Total Quantity, Profit Margin %."""
    total_sales = float(df["Sales"].sum())
    total_profit = float(df["Profit"].sum())
    return {
        "total_sales": total_sales,
        "total_profit": total_profit,
        "total_orders": int(df["Order ID"].nunique()),
        "total_quantity": int(df["Quantity"].sum()),
        "profit_margin": (total_profit / total_sales * 100) if total_sales else 0.0,
    }
