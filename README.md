# Practical 10 – Sales Analytics Dashboard (Streamlit)

Upload CSV / Excel / JSON sales data → validate → clean → filter → explore KPIs & interactive charts.

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Files
- `app.py` – Streamlit UI (tabs, filters, KPIs, charts)
- `data_utils.py` – reusable functions: loading, validation, cleaning, statistics, KPIs
- `test_data_utils.py` – tests (`python test_data_utils.py`)
- `experiment10_raw_dataset_-_experiment10_raw_dataset.csv` – sample dataset

## Deployment
Push to GitHub → https://share.streamlit.io → New app → select repo, branch `main`, main file `app.py`.
