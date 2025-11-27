"""
Streamlit app: Indian FII / DII inflow-outflow & sector heatmap
Clean, minimal dashboard with:
- FII / DII net flows (demo or NSE-like fetch placeholder)
- Sector-wise daily heatmap
- Monthly sector trend line chart
- CSV download of filtered data
"""

import datetime as dt

import numpy as np
import pandas as pd
import plotly.express as px
import requests
import streamlit as st

# ------------------------
# Config / Palette
# ------------------------
APP_TITLE = "Indian FII / DII Trend Analysis"

PALETTE = {
    "bg": "#020617",
    "panel": "#020617",
    "card": "#020617",
    "border": "#1f2937",
    "text": "#e5e7eb",
    "accent_buy": "#16a34a",   # green inflow
    "accent_sell": "#ef4444",  # red outflow
    "neutral": "#94a3b8",
}

st.set_page_config(page_title=APP_TITLE, layout="wide")

# ------------------------
# Simple dark CSS
# ------------------------
CUSTOM_CSS = f"""
<style>
/* App background */
.main {{
    background: radial-gradient(circle at top left, #0f172a 0, #020617 45%, #020617 100%);
    color: {PALETTE['text']};
}}
[data-testid="stSidebar"] {{
    background: linear-gradient(180deg, #020617 0%, #020617 50%, #020617 100%);
    border-right: 1px solid {PALETTE['border']};
}}
/* Titles */
h1, h2, h3, h4, h5 {{
    color: {PALETTE['text']} !important;
}}
/* Metric card */
.metric-card {{
    background: rgba(15,23,42,0.95);
    border-radius: 18px;
    padding: 14px 18px;
    border: 1px solid {PALETTE['border']};
    box-shadow: 0 18px 35px rgba(15, 23, 42, 0.8);
}}
.metric-label {{
    font-size: 0.8rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #9ca3af;
}}
.metric-value {{
    font-size: 1.5rem;
    font-weight: 600;
    margin-top: 4px;
}}
.metric-positive {{ color: #4ade80; }}
.metric-negative {{ color: #f97373; }}
.section-card {{
    background: rgba(15,23,42,0.9);
    border-radius: 22px;
    padding: 16px 18px;
    border: 1px solid {PALETTE['border']};
    box-shadow: 0 10px 30px rgba(15, 23, 42, 0.9);
}}
.section-label {{
    text-transform: uppercase;
    letter-spacing: 0.15em;
    font-size: 0.7rem;
    color: #6b7280;
    margin-bottom: 4px;
}}
.footer-text {{
    font-size: 0.78rem;
    color: #6b7280;
    text-align: center;
    margin-top: 10px;
}}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ------------------------
# Demo data (fallback)
# ------------------------
@st.cache_data(show_spinner=False)
def generate_demo_data(start_date: dt.date, end_date: dt.date) -> pd.DataFrame:
    """Generate synthetic FII/DII sector flow data between start_date and end_date."""
    dates = pd.date_range(start_date, end_date, freq="B")  # business days
    sectors = [
        "Financials",
        "Information Technology",
        "Energy",
        "Consumer Discretionary",
        "Pharmaceuticals",
        "Automobiles",
        "FMCG",
        "Metals",
        "Utilities",
        "Real Estate",
    ]

    rng = np.random.default_rng(seed=42)
    rows = []

    for d in dates:
        for s in sectors:
            base = rng.normal(loc=0.0, scale=1.0)
            multiplier = 1000 + rng.integers(-400, 400)
            total_value = base * multiplier

            fii = total_value * rng.uniform(0.6, 1.2)
            dii = -total_value * rng.uniform(0.4, 1.1)  # often opposite sign

            rows.append(
                {
                    "date": d.date(),
                    "sector": s,
                    "fii_net": float(np.round(fii, 2)),
                    "dii_net": float(np.round(dii, 2)),
                    "total_net": float(np.round(fii + dii, 2)),
                }
            )

    return pd.DataFrame(rows)


# ------------------------
# Data fetcher (placeholder for NSE)
# ------------------------
@st.cache_data(show_spinner=False)
def fetch_nse_fii_dii(start_date: dt.date, end_date: dt.date) -> pd.DataFrame:
    """
    Placeholder NSE fetch.

    Tries to hit a fiidii-like endpoint; if it fails, falls back to demo data.
    This endpoint/structure may change anytime, so treat it as a best-effort stub.
    """
    url = f"https://www.nseindia.com/api/fiidii?from={start_date.isoformat()}&to={end_date.isoformat()}"
    headers = {"User-Agent": "Mozilla/5.0"}

    try:
        resp = requests.get(url, headers=headers, timeout=10)
        if resp.status_code == 200:
            try:
                data = resp.json()
                # This is all speculative; real structure may differ.
                if isinstance(data, dict) and "data" in data:
                    df = pd.json_normalize(data["data"])
                else:
                    df = pd.json_normalize(data)

                # Try to map to expected columns; if missing, fall back to demo.
                # You will need to adapt this part once you inspect real JSON.
                possible_date_cols = [c for c in df.columns if "date" in c.lower()]
                if possible_date_cols:
                    date_col = possible_date_cols[0]
                    df["date"] = pd.to_datetime(df[date_col]).dt.date
                else:
                    raise ValueError("No date column in NSE response")

                if "sector" not in df.columns:
                    raise ValueError("No sector column in NSE response")

                # Very rough, you will need to map real column names:
                if "fii_net" not in df.columns:
                    # try some common variant names
                    for c in df.columns:
                        if "fii" in c.lower() and "net" in c.lower():
                            df = df.rename(columns={c: "fii_net"})
                            break

                if "dii_net" not in df.columns:
                    for c in df.columns:
                        if ("dii" in c.lower() or "domestic" in c.lower()) and "net" in c.lower():
                            df = df.rename(columns={c: "dii_net"})
                            break

                if "fii_net" not in df.columns:
                    raise ValueError("No fii_net column after mapping")

                if "dii_net" not in df.columns:
                    df["dii_net"] = 0.0

                df["total_net"] = df["fii_net"] + df["dii_net"]
                return df[["date", "sector", "fii_net", "dii_net", "total_net"]]

            except Exception:
                # If parsing fails, go to demo
                pass
    except Exception:
        pass

    st.info("Live NSE fetch failed or unsupported — using demo data instead.")
    return generate_demo_data(start_date, end_date)


# ------------------------
# Processing helpers
# ------------------------
def prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize columns, ensure date & month, and basic types."""
    df = df.copy()

    required = {"date", "sector", "fii_net", "dii_net", "total_net"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in data: {missing}")

    df["date"] = pd.to_datetime(df["date"]).dt.date
    df["sector"] = df["sector"].astype(str)
    df["fii_net"] = pd.to_numeric(df["fii_net"], errors="coerce").fillna(0.0)
    df["dii_net"] = pd.to_numeric(df["dii_net"], errors="coerce").fillna(0.0)
    df["total_net"] = pd.to_numeric(df["total_net"], errors="coerce").fillna(0.0)

    # month as Timestamp for plotting
    df["month"] = pd.to_datetime(df["date"]).to_period("M").dt.to_timestamp()
    return df


def sector_daily_pivot(df: pd.DataFrame, metric_col: str) -> pd.DataFrame:
    """Pivot sector vs date table using chosen metric (total_net / fii_net / dii_net)."""
    return df.pivot_table(
        index="sector",
        columns="date",
        values=metric_col,
        aggfunc="sum",
        fill_value=0,
    )


def monthly_aggregates(df: pd.DataFrame, metric_col: str) -> pd.DataFrame:
    """Monthly aggregates by sector for chosen metric."""
    m = df.groupby(["month", "sector"])[metric_col].sum().reset_index()
    m["month_str"] = m["month"].dt.strftime("%Y-%m")
    return m


# ------------------------
# Main UI
# ------------------------
def main():
    st.title(APP_TITLE)
    st.markdown(
        "Minimalist dashboard showing **FII/DII inflow–outflow**, "
        "**sector-wise heatmap**, and **monthly trends**. "
        "For live usage, adapt the NSE fetch mapping to your data source."
    )

    # Sidebar
    with st.sidebar:
        st.header("Filters")

        today = dt.date.today()
        default_start = today - dt.timedelta(days=180)

        start_date = st.date_input("Start Date", default_start)
        end_date = st.date_input("End Date", today)

        if start_date > end_date:
            st.error("Start date must be before end date.")
            st.stop()

        st.markdown("---")
        st.caption("Data source is best-effort NSE → falls back to demo data.")

    # Fetch & prepare
    raw_df = fetch_nse_fii_dii(start_date, end_date)
    try:
        df = prepare_data(raw_df)
    except Exception as e:
        st.error(f"Data format error: {e}")
        st.stop()

    all_sectors = sorted(df["sector"].unique())

    # Controls row
    col1, col2, col3 = st.columns([1.3, 1.3, 1])

    with col1:
        sectors_sel = st.multiselect(
            "Select sectors",
            all_sectors,
            default=all_sectors[: min(5, len(all_sectors))],
        )

    with col2:
        metric_name_map = {
            "Net (FII + DII)": "total_net",
            "FII net only": "fii_net",
            "DII net only": "dii_net",
        }
        metric_label = st.selectbox(
            "Metric", list(metric_name_map.keys()), index=0
        )
        metric_col = metric_name_map[metric_label]

    with col3:
        agg_period = st.selectbox("Aggregate view", ["Daily", "Monthly"], index=0)

    if sectors_sel:
        filtered = df[df["sector"].isin(sectors_sel)]
    else:
        filtered = df.copy()

    if filtered.empty:
        st.warning("No data for the selected filters.")
        st.stop()

    # ------------------------
    # KPI section
    # ------------------------
    k1, k2, k3 = st.columns(3)

    total_val = filtered[metric_col].sum()
    inflow = filtered[filtered[metric_col] > 0][metric_col].sum()
    outflow = filtered[filtered[metric_col] < 0][metric_col].sum()

    with k1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown('<div class="metric-label">Total Inflow</div>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="metric-value metric-positive">₹ {inflow:,.0f}</div>',
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

    with k2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown('<div class="metric-label">Total Outflow</div>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="metric-value metric-negative">₹ {outflow:,.0f}</div>',
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

    with k3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown('<div class="metric-label">Net Total</div>', unsafe_allow_html=True)
        css_class = "metric-positive" if total_val >= 0 else "metric-negative"
        st.markdown(
            f'<div class="metric-value {css_class}">₹ {total_val:,.0f}</div>',
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("")

    # ------------------------
    # Heatmap
    # ------------------------
    st.subheader("Daily Sector Heatmap")

    pivot = sector_daily_pivot(filtered, metric_col)

    if pivot.empty:
        st.info("No data available for selected filters.")
    else:
        heat_df = (
            pivot.reset_index()
            .melt(id_vars=["sector"], var_name="date", value_name="net")
        )
        heat_df["date_str"] = heat_df["date"].astype(str)

        fig_heat = px.density_heatmap(
            heat_df,
            x="date_str",
            y="sector",
            z="net",
            color_continuous_scale=[[0, "red"], [0.5, "white"], [1, "green"]],
            title=f"Daily Heatmap — {metric_label}",
            height=500,
        )
        fig_heat.update_layout(
            template="plotly_dark",
            xaxis_title="Date",
            yaxis_title="Sector",
            margin=dict(l=10, r=10, t=40, b=10),
        )
        st.plotly_chart(fig_heat, use_container_width=True)

    # ------------------------
    # Monthly line chart
    # ------------------------
    st.subheader("Monthly Net Flow by Sector")

    monthly = monthly_aggregates(filtered, metric_col)
    if monthly.empty:
        st.info("No monthly data for selected filters.")
    else:
        fig_month = px.line(
            monthly,
            x="month_str",
            y=metric_col,
            color="sector",
            markers=True,
            title=f"Monthly Sector Trends — {metric_label}",
        )
        fig_month.update_layout(
            template="plotly_dark",
            xaxis_title="Month",
            yaxis_title="Net Value",
            margin=dict(l=10, r=10, t=40, b=10),
        )
        st.plotly_chart(fig_month, use_container_width=True)

    # ------------------------
    # Data table + download
    # ------------------------
    st.subheader("Filtered Data Table")
    st.dataframe(filtered)

    csv_bytes = filtered.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download CSV",
        csv_bytes,
        "fii_dii_filtered_data.csv",
        "text/csv",
    )

    st.markdown(
        '<div class="footer-text">'
        "Built with Streamlit — for production, plug in a stable NSE / data-vendor API "
        "and map columns properly."
        "</div>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
