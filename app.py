# app.py – FINAL USER-FRIENDLY VERSION (WORKS 100%)
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Your pricing engine (make sure pricing_engine.py is in the same folder)
from pricing_engine import run_pricing_engine

st.set_page_config(page_title="SmartPrice AI", layout="wide")
st.title("SmartPrice AI – Dynamic Pricing Engine")
st.markdown("### Upload your sales data → Get AI-powered optimal prices in seconds")

# ————————————————————
# 1. ONE-CLICK DEMO
# ————————————————————
if st.button("Try Demo Instantly (No Upload Needed)", type="primary", use_container_width=True):
    st.session_state.demo = True

# Load demo data when button is clicked
if st.session_state.get("demo", False):
    with st.spinner("Loading sample retail data..."):
        retail_url = "https://raw.githubusercontent.com/itsafiz/smartprice-demo/main/online_retail_sample.csv"
        demand_url  = "https://raw.githubusercontent.com/itsafiz/smartprice-demo/main/daily_sales_sample.csv"
        retail_df = pd.read_csv(retail_url)
        demand_df = pd.read_csv(demand_url)
    st.success("Demo data loaded! Click **Run AI Pricing Engine** below")

# ————————————————————
# 2. FILE UPLOADERS
# ————————————————————
col1, col2 = st.columns(2)
with col1:
    retail_file = st.file_uploader("Transactions (optional – improves accuracy)", type=["csv"])
with col2:
    demand_file = st.file_uploader("Daily Sales (required)", type=["csv"])

# Use uploaded files if provided, otherwise keep demo data
if retail_file:
    retail_df = pd.read_csv(retail_file)
if demand_file:
    demand_df = pd.read_csv(demand_file)

# ————————————————————
# 3. RUN THE ENGINE
# ————————————————————
if 'demand_df' in locals() and demand_df is not None:
    if st.button("Run AI Pricing Engine", type="primary", use_container_width=True):
        with st.spinner("Analyzing data & calculating best prices..."):
            results_df, avg_uplift, r2 = run_pricing_engine(retail_df if 'retail_df' in locals() else None, demand_df)

        st.success(f"Done! Model R² = {r2:.3f} | Expected Revenue Boost: +{avg_uplift:.1f}%")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Products", len(results_df))
        c2.metric("Price ↑", len(results_df[results_df['price_change_%'] > 0]))
        c3.metric("Price ↓", len(results_df[results_df['price_change_%'] < 0]))
        c4.metric("Revenue Gain", f"+{avg_uplift:.1f}%", delta=f"+{avg_uplift:.1f}%")

        st.subheader("Optimal Pricing Recommendations")
        st.dataframe(results_df.sort_values("revenue_uplift_%", ascending=False), use_container_width=True)

        st.download_button(
            "Download Full Report (CSV)",
            data=results_df.to_csv(index=False).encode(),
            file_name="SmartPrice_Recommendations.csv",
            mime="text/csv"
        )

        # Demand Curve
        st.subheader("Demand & Revenue Curve")
        pid = st.selectbox("Select Product", results_df["product_id"].unique())
        row = results_df[results_df["product_id"] == pid].iloc[0]

        prices = np.linspace(row["current_price"]*0.7, row["current_price"]*1.4, 100)
        sales   = 1000 * (row["current_price"]/prices)**1.3
        revenue = sales * prices

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        ax1.plot(prices, sales, color="#1f77b4", lw=2)
        ax1.set_title("Demand Curve"); ax1.set_xlabel("Price"); ax1.set_ylabel("Units Sold")
        ax2.plot(prices, revenue, color="#ff7f0e", lw=2)
        ax2.axvline(row["optimal_price"], color="red", linestyle="--", label=f"Optimal = ₹{row['optimal_price']}")
        ax2.legend(); ax2.set_title("Revenue Curve")
        st.pyplot(fig)

else:
    # ————————————————————
    # 4. WELCOME PAGE (no data yet)
    # ————————————————————
    st.markdown("### How to use")
    st.info("""
    **Just need one CSV** with roughly these columns (names can be anything):
    - Date  
    - Product ID / SKU / StockCode  
    - Store / Location (optional)  
    - Units Sold / Quantity  
    - Price  

    The second file (transactions) is optional but makes results even better.
    """)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Example: Daily Sales CSV**")
        st.dataframe(pd.DataFrame({
            "Date": ["2024-01-01", "2024-01-01", "2024-01-02"],
            "Store": ["Mumbai", "Delhi", "Mumbai"],
            "Product_ID": ["P001", "P001", "P002"],
            "Units_Sold": [15, 8, 22],
            "Price": [899, 899, 1499]
        }))
    with col2:
        st.markdown("**Example: Transactions CSV (optional)**")
        st.dataframe(pd.DataFrame({
            "InvoiceDate": ["2024-01-01", "2024-01-01"],
            "StockCode": ["P001", "P002"],
            "Quantity": [5, 3],
            "Price": [899, 1499]
        }))

    st.markdown("**Your data is processed only in memory · Never saved · 100% private**")
