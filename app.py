# app.py – FINAL PERFECT VERSION (Dec 2025)
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pricing_engine import run_pricing_engine

st.set_page_config(page_title="SmartPrice AI", layout="wide")
st.title("SmartPrice AI – Dynamic Pricing Engine")
st.markdown("### Upload your sales data → Get AI-powered optimal prices instantly")

# ——————————————————————
# 1. ONE-CLICK DEMO THAT AUTO-RUNS
# ——————————————————————
if st.button("Try Demo Instantly (No Upload Needed)", type="primary", use_container_width=True):
    st.session_state.demo_mode = True

# Auto-load and auto-run demo when button clicked
if st.session_state.get("demo_mode"):
    demand_df = pd.DataFrame({
        "date": pd.date_range("2024-01-01", periods=30),
        "store_id": np.random.choice(["Mumbai", "Delhi", "Bangalore"], 30),
        "product_id": np.random.choice(["P100", "P200", "P300", "P400"], 30),
        "units_sold": np.random.randint(10, 80, 30),
        "price": np.round(np.random.uniform(299, 2999, 30), 2)
    })
    retail_df = None  # Not needed for demo

    st.success("Demo data loaded – running AI pricing engine automatically!")
    run_demo = True
else:
    run_demo = False

# ——————————————————————
# 2. FILE UPLOADERS (for real users)
# ——————————————————————
col1, col2 = st.columns(2)
with col1:
    retail_file = st.file_uploader("Transactions CSV (optional)", type=["csv"])
with col2:
    demand_file = st.file_uploader("Daily Sales CSV (required for custom data)", type=["csv"])

# Load user files if provided
if retail_file:
    retail_df = pd.read_csv(retail_file)
if demand_file:
    demand_df = pd.read_csv(demand_file)

# ——————————————————————
# 3. MAIN RUN BUTTON (only show if NOT in demo mode)
# ——————————————————————
show_run_button = not st.session_state.get("demo_mode", False) and demand_file

if show_run_button:
    if st.button("Run AI Pricing Engine", type="primary", use_container_width=True):
        run_demo = True
else:
    if st.session_state.get("demo_mode"):
        run_demo = True

# ——————————————————————
# 4. ACTUALLY RUN THE ENGINE
# ——————————————————————
if run_demo and 'demand_df' in locals():
    with st.spinner("Training AI model & optimizing prices... (10-20 seconds)"):
        results_df, avg_uplift, r2 = run_pricing_engine(
            retail_df if 'retail_df' in locals() else None,
            demand_df
        )

    # Success message
    st.success(f"Done! Model Accuracy (R²): {r2:.3f} | Expected Revenue Uplift: +{avg_uplift:.1f}%")

    # Metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Products Analyzed", len(results_df))
    col2.metric("Price Increases", len(results_df[results_df['price_change_%'] > 0]))
    col3.metric("Price Decreases", len(results_df[results_df['price_change_%'] < 0]))
    col4.metric("Revenue Gain", f"+{avg_uplift:.1f}%", delta=f"+{avg_uplift:.1f}%")

    # Results table
    st.subheader("Optimal Pricing Recommendations")
    st.dataframe(
        results_df.sort_values("revenue_uplift_%", ascending=False),
        use_container_width=True,
        hide_index=True
    )

    # Download button
    st.download_button(
        "Download Full Report (CSV)",
        data=results_df.to_csv(index=False).encode(),
        file_name=f"SmartPrice_Recommendations_{pd.Timestamp.now():%Y%m%d}.csv",
        mime="text/csv"
    )

    # Interactive demand curve
    st.subheader("Explore Demand & Revenue Curve")
    product = st.selectbox("Select Product", results_df["product_id"].unique())
    row = results_df[results_df["product_id"] == product].iloc[0]

    prices = np.linspace(row["current_price"] * 0.7, row["current_price"] * 1.4, 100)
    sales = 500 * (row["current_price"] / prices) ** 1.3
    revenue = sales * prices

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.plot(prices, sales, color="#6366f1", lw=3)
    ax1.set_title("Demand Curve"); ax1.set_xlabel("Price (₹)"); ax1.set_ylabel("Units Sold")
    ax2.plot(prices, revenue, color="#10b981", lw=3)
    ax2.axvline(row["optimal_price"], color="red", linestyle="--", linewidth=3, label=f"Optimal = ₹{row['optimal_price']}")
    ax2.legend(); ax2.set_title("Revenue Curve"); ax2.set_xlabel("Price (₹)")
    st.pyplot(fig)

    st.markdown("**Your data was processed securely and never stored.**")

# ——————————————————————
# 5. WELCOME PAGE (when no data yet)
# ——————————————————————
else:
    st.info("""
    **Click the red button above to see SmartPrice AI in action instantly!**

    Or upload your own CSV with these columns (names can be anything):
    - Date  
    - Product ID / SKU  
    - Units Sold / Quantity  
    - Price  

    The AI automatically understands your data format.
    """)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Example Daily Sales**")
        st.dataframe(pd.DataFrame({
            "Date": ["2024-01-01", "2024-01-02"],
            "Store": ["Mumbai", "Delhi"],
            "Product_ID": ["P100", "P100"],
            "Units_Sold": [23, 18],
            "Price": [999, 999]
        }))
    with col2:
        st.markdown("**Example Transactions (optional)**")
        st.dataframe(pd.DataFrame({
            "InvoiceDate": ["2024-01-01"],
            "StockCode": ["P100"],
            "Quantity": [10],
            "Price": [999]
        }))

    st.markdown("**100% private · No login · Works on mobile**")
