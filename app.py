# app.py – 100% WORKING – December 2025
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pricing_engine import run_pricing_engine

st.set_page_config(page_title="SmartPrice AI", layout="wide")
st.title("SmartPrice AI – Dynamic Pricing Engine")
st.markdown("### Upload your sales data → Get AI-powered optimal prices instantly")

# ——————————————————————
# ONE-CLICK DEMO (auto runs)
# ——————————————————————
if st.button("Try Demo Instantly (No Upload Needed)", type="primary", use_container_width=True):
    st.session_state.demo = True

if st.session_state.get("demo"):
    # Small realistic demo data
    demand_df = pd.DataFrame({
        "date": pd.date_range("2024-01-01", periods=60).tolist() * 4,
        "store_id": ["North", "South", "East", "West"] * 60,
        "product_id": [f"P{str(i).zfill(4)}" for i in range(1, 17)] * 15,
        "units_sold": np.random.randint(15, 150, 240),
        "price": np.round(np.random.uniform(199, 3999, 240), 2)
    })
    retail_df = None
    st.success("Demo loaded – running automatically!")
    auto_run = True
else:
    auto_run = False

# ——————————————————————
# FILE UPLOADERS
# ——————————————————————
col1, col2 = st.columns(2)
with col1:
    retail_file = st.file_uploader("Transactions CSV (optional)", type=["csv"])
with col2:
    demand_file = st.file_uploader("Daily Sales / Inventory CSV", type=["csv"])

if retail_file:
    retail_df = pd.read_csv(retail_file)
if demand_file:
    demand_df = pd.read_csv(demand_file)

# ——————————————————————
# RUN BUTTON
# ——————————————————————
run_now = auto_run
if not auto_run and demand_file:
    if st.button("Run AI Pricing Engine", type="primary", use_container_width=True):
        run_now = True

# ——————————————————————
# EXECUTE THE ENGINE
# ——————————————————————
if run_now and 'demand_df' in locals() and not demand_df.empty:
    with st.spinner("Training model & optimizing prices..."):
        results_df, avg_uplift, r2 = run_pricing_engine(
            retail_df if 'retail_df' in locals() else None,
            demand_df
        )

    st.success(f"Done! Model R²: {r2:.3f} | Expected Revenue Uplift: +{avg_uplift:.1f}%")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Products", len(results_df))
    c2.metric("Price Increases", len(results_df[results_df['price_change_%'] > 0]))
    c3.metric("Price Decreases", len(results_df[results_df['price_change_%'] < 0]))
    c4.metric("Revenue Gain", f"+{avg_uplift:.1f}%", delta=f"+{avg_uplift:.1f}%")

    st.subheader("Optimal Pricing Recommendations")
    st.dataframe(results_df.sort_values("revenue_uplift_%", ascending=False),
                 use_container_width=True, hide_index=True)

    st.download_button("Download Report (CSV)",
                       data=results_df.to_csv(index=False).encode(),
                       file_name="SmartPrice_Recommendations.csv",
                       mime="text/csv")

    st.subheader("Demand & Revenue Curve")
    pid = st.selectbox("Select Product", results_df["product_id"].unique())
    row = results_df[results_df["product_id"] == pid].iloc[0]

    prices = np.linspace(row["current_price"]*0.7, row["current_price"]*1.4, 100)
    sales = 500 * (row["current_price"]/prices)**1.2
    revenue = sales * prices

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.plot(prices, sales, color="#6366f1", lw=3)
    ax1.set_title("Demand Curve"); ax1.set_xlabel("Price"); ax1.set_ylabel("Units Sold")
    ax2.plot(prices, revenue, color="#10b981", lw=3)
    ax2.axvline(row["optimal_price"], color="red", linestyle="--", lw=3,
                label=f"Optimal = ₹{row['optimal_price']}")
    ax2.legend(); ax2.set_title("Revenue Curve")
    st.pyplot(fig)

    st.caption("Your data was processed only in memory – never stored or shared.")

# ——————————————————————
# WELCOME PAGE
# ——————————————————————
else:
    st.info("Click the red button above for an instant demo!\n\n"
            "Or upload your own CSV containing Date, Product ID, Store ID, Units Sold, Price.")
    st.markdown("**100% private • No login • Works on phone**")
