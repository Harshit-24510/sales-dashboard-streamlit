"""Sales Analytics Dashboard - Practical 10 (Streamlit web application)."""
import os

import pandas as pd
import plotly.express as px
import streamlit as st

from data_utils import (REQUIRED_COLUMNS, calculate_kpis, categorical_summary,
                        clean_data, load_file, summary_statistics, validate_data)

SAMPLE_FILE = "experiment10_raw_dataset_-_experiment10_raw_dataset.csv"

st.set_page_config(page_title="Sales Analytics Dashboard", page_icon="📊", layout="wide")


# ----------------------------------------------------------------------------
# Cached helpers
# ----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def cached_clean(df: pd.DataFrame):
    return clean_data(df)


class _SampleFile:
    """Wrap the bundled sample CSV so it behaves like an uploaded file."""
    name = SAMPLE_FILE

    def __init__(self, path):
        self._path = path

    def read(self, *a):
        with open(self._path, "rb") as f:
            return f.read()


def get_raw_data():
    """Return (DataFrame, source_name) from the uploader or the bundled sample; handles errors."""
    uploaded = st.sidebar.file_uploader("Upload CSV / Excel / JSON", type=["csv", "xlsx", "xls", "json"])
    use_sample = False
    if uploaded is None and os.path.exists(SAMPLE_FILE):
        use_sample = st.sidebar.checkbox("Use bundled sample dataset", value=True)
    try:
        if uploaded is not None:
            return load_file(uploaded), uploaded.name
        if use_sample:
            return pd.read_csv(SAMPLE_FILE), SAMPLE_FILE
    except Exception as e:  # corrupt file, wrong format, empty file ...
        st.error(f"Could not read the file: {e}")
    return None, None


# ----------------------------------------------------------------------------
# UI sections
# ----------------------------------------------------------------------------
def render_validation(raw: pd.DataFrame):
    v = validate_data(raw)
    st.subheader("Dataset dimensions")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", v["rows"])
    c2.metric("Columns", v["cols"])
    c3.metric("Missing values", v["total_missing"])
    c4.metric("Duplicate records", v["duplicate_count"])
    st.subheader("Column names, data types and missing values")
    st.dataframe(v["column_info"], use_container_width=True, hide_index=True)
    if v["duplicate_count"]:
        with st.expander(f"View duplicate records ({v['duplicate_count']} duplicates)"):
            st.dataframe(v["duplicate_rows"], use_container_width=True)
    st.subheader("Raw data preview")
    st.dataframe(raw.head(20), use_container_width=True)
    return v


def render_preprocessing(raw, clean, log):
    st.subheader("Cleaning steps applied")
    for step in log:
        st.markdown(f"- {step}")
    c1, c2 = st.columns(2)
    c1.metric("Rows before → after", f"{len(raw)} → {len(clean)}")
    c2.metric("Missing values before → after",
              f"{int(raw.isna().sum().sum())} → {int(clean[REQUIRED_COLUMNS].isna().sum().sum())}")
    st.subheader("Statistical summary (numeric columns)")
    st.dataframe(summary_statistics(clean), use_container_width=True)
    col_a, col_b = st.columns(2)
    for col, holder in (("Category", col_a), ("Region", col_b)):
        holder.markdown(f"**{col} frequency**")
        holder.dataframe(categorical_summary(clean, col), hide_index=True, use_container_width=True)
    st.download_button("⬇️ Download cleaned data (CSV)", clean.to_csv(index=False).encode("utf-8"),
                       "cleaned_sales_data.csv", "text/csv")


def apply_filters(df: pd.DataFrame) -> pd.DataFrame:
    """Sidebar filters -> filtered DataFrame."""
    st.sidebar.header("🔎 Filters")
    dmin, dmax = df["Order Date"].min().date(), df["Order Date"].max().date()
    date_range = st.sidebar.date_input("Order date range", (dmin, dmax), min_value=dmin, max_value=dmax)
    regions = st.sidebar.multiselect("Region", sorted(df["Region"].dropna().unique()),
                                     default=sorted(df["Region"].dropna().unique()))
    cats = st.sidebar.multiselect("Category", sorted(df["Category"].unique()),
                                  default=sorted(df["Category"].unique()))
    segs = st.sidebar.multiselect("Segment", sorted(df["Segment"].unique()),
                                  default=sorted(df["Segment"].unique()))
    modes = st.sidebar.multiselect("Ship Mode", sorted(df["Ship Mode"].unique()),
                                   default=sorted(df["Ship Mode"].unique()))
    mask = df["Region"].isin(regions) & df["Category"].isin(cats) & df["Segment"].isin(segs) \
        & df["Ship Mode"].isin(modes)
    if isinstance(date_range, tuple) and len(date_range) == 2:
        mask &= df["Order Date"].between(pd.Timestamp(date_range[0]), pd.Timestamp(date_range[1]))
    return df[mask]


def render_kpis(df):
    k = calculate_kpis(df)
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("💰 Total Sales", f"${k['total_sales']:,.0f}")
    c2.metric("📈 Total Profit", f"${k['total_profit']:,.0f}")
    c3.metric("🧾 Total Orders", f"{k['total_orders']:,}")
    c4.metric("📦 Quantity Sold", f"{k['total_quantity']:,}")
    c5.metric("% Profit Margin", f"{k['profit_margin']:.2f}%")


def render_charts(df):
    r1c1, r1c2 = st.columns(2)
    monthly = df.groupby("Month", as_index=False)[["Sales", "Profit"]].sum()
    fig = px.line(monthly, x="Month", y=["Sales", "Profit"], markers=True, title="Monthly Sales & Profit Trend")
    fig.update_layout(legend_title_text="", yaxis_title="Amount ($)")
    r1c1.plotly_chart(fig, use_container_width=True)

    cat = df.groupby("Category", as_index=False)[["Sales", "Profit"]].sum()
    fig = px.bar(cat, x="Category", y=["Sales", "Profit"], barmode="group", title="Sales & Profit by Category")
    fig.update_layout(legend_title_text="", yaxis_title="Amount ($)")
    r1c2.plotly_chart(fig, use_container_width=True)

    r2c1, r2c2 = st.columns(2)
    reg = df.groupby("Region", as_index=False)["Sales"].sum()
    r2c1.plotly_chart(px.pie(reg, names="Region", values="Sales", hole=0.4, title="Sales Share by Region"),
                      use_container_width=True)
    top = df.groupby("Product Name", as_index=False)["Profit"].sum().nlargest(10, "Profit")
    r2c2.plotly_chart(px.bar(top.sort_values("Profit"), x="Profit", y="Product Name", orientation="h",
                             title="Top 10 Products by Profit"), use_container_width=True)

    r3c1, r3c2 = st.columns(2)
    r3c1.plotly_chart(px.scatter(df, x="Discount", y="Profit", color="Category", size="Sales",
                                 hover_data=["Product Name", "Region"], title="Discount vs Profit"),
                      use_container_width=True)
    heat = df.pivot_table(index="Region", columns="Category", values="Sales", aggfunc="sum", fill_value=0)
    r3c2.plotly_chart(px.imshow(heat, text_auto=".0f", aspect="auto", color_continuous_scale="Blues",
                                title="Sales Heatmap: Region × Category"), use_container_width=True)

    state = df.groupby("State", as_index=False)["Sales"].sum().sort_values("Sales", ascending=False)
    st.plotly_chart(px.bar(state, x="State", y="Sales", color="Sales", title="Sales by State"),
                    use_container_width=True)


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main():
    st.title("📊 Sales Analytics Dashboard")
    st.caption("Upload → Validate → Clean → Filter → Visualize")
    st.sidebar.header("📁 Data")
    raw, source = get_raw_data()
    if raw is None:
        st.info("👈 Upload a CSV, Excel or JSON file from the sidebar to begin.")
        st.stop()
    if raw.empty:
        st.error("The uploaded file contains no data.")
        st.stop()
    st.sidebar.success(f"Loaded: {source}")

    missing_cols = [c for c in REQUIRED_COLUMNS if c not in raw.columns]
    if missing_cols:
        st.error(f"Missing required columns: {', '.join(missing_cols)}")
        st.dataframe(raw.head(), use_container_width=True)
        st.stop()

    try:
        clean, log = cached_clean(raw)
    except Exception as e:
        st.error(f"Data cleaning failed: {e}")
        st.stop()

    tab1, tab2, tab3, tab4 = st.tabs(["1️⃣ Upload & Validation", "2️⃣ Preprocessing & Statistics",
                                      "3️⃣ Dashboard", "4️⃣ Data Explorer"])
    with tab1:
        render_validation(raw)
    with tab2:
        render_preprocessing(raw, clean, log)
    filtered = apply_filters(clean)
    with tab3:
        if filtered.empty:
            st.warning("No records match the selected filters. Please widen your selection.")
        else:
            st.caption(f"Showing {len(filtered)} of {len(clean)} records")
            render_kpis(filtered)
            st.divider()
            render_charts(filtered)
    with tab4:
        st.dataframe(filtered, use_container_width=True)
        st.download_button("⬇️ Download filtered data", filtered.to_csv(index=False).encode("utf-8"),
                           "filtered_data.csv", "text/csv")


if __name__ == "__main__":
    main()
