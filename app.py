# app.py – GORGEOUS PROFESSIONAL UI (2025 Edition)
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pricing_engine import run_pricing_engine

# —————— PAGE CONFIG & THEME ——————
st.set_page_config(
    page_title="SmartPrice AI – Dynamic Pricing",
    page_icon="chart_with_upwards_trend",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS – makes it look like a premium SaaS
st.markdown("""
<style>
    .main {background-color: #0e1117; color: #fafafa;}
    .stButton>button {
        background: linear-gradient(90deg, #ff4b6e, #ff8c42);
        color: white; font-weight: bold; border-radius: 12px;
        height: 60px; font-size: 20px; border: none;
        box-shadow: 0 4px 15px rgba(255, 75, 110, 0.4);
    }
    .metric-card {
        background: linear-gradient(135deg, #1e293b, #334155);
        padding: 20px; border-radius: 16px; text-align: center;
        box-shadow: 0 8px 20px rgba(0,0,0,0.3);
    }
    h1, h2, h3 {font-family: 'Segoe UI', sans-serif; text-align: center;}
    .success-box {background: #064e3b; padding: 20px; border-radius: 12px; text-align: center;}
</style>
""", unsafe_allow_html=True)

# —————— HEADER ——————
col1, col2, col3 = st.columns([1, 3, 1])
with col2:
    st.markdown("<h1 style='color:#ff4b6e;'>SmartPrice AI</h1>", unsafe_allow_html=True)
    st.markdown("<h3 style='color:#94a3b8;'>AI-Powered Dynamic Pricing Engine</h3>", unsafe_allow_html=True)

st.markdown("---")

# —————— ONE-CLICK DEMO ——————
if st.button("Try Demo Instantly (No Upload Needed)", type="primary", use_container_width=True):
    st.session_state.demo = True
    st.balloons()

if st.session_state.get("demo"):
    demand_df = pd.DataFrame({
        "date": pd.date_range("2024-01-01", periods=100),
        "store_id": np.random.choice(["Mumbai", "Delhi", "Bangalore", "Kolkata"], 100),
        "product_id": [f"P{str(i).zfill(4)}" for i in np.random.randint(1, 50, 100)],
        "units_sold": np.random.randint(20, 300, 100),
        "price": np.round(np.random.uniform(149, 4999, 100), 2)
    })
    retail_df = None
    auto_run = True
else:
    auto_run = False

# —————— FILE UPLOADERS (BEAUTIFUL) ——————
st.markdown("<h3 style='color:#e2e8f0;'>Upload Your Data</h3>", unsafe_allow_html=True)
col1, col2 = st.columns(2)
with col1:
    retail_file = st.file_uploader("Transactions CSV (optional)", type=["csv"])
with col2:
    demand_file = st.file_uploader("Daily Sales / Inventory CSV (required)", type=["csv"])

if retail_file:
    retail_df = pd.read_csv(retail_file)
if demand_file:
    demand_df = pd.read_csv(demand_file)

# —————— RUN ENGINE ——————
run_now = auto_run
if not auto_run and demand_file:
    if st.button("Run AI Pricing Engine", type="primary", use_container_width=True):
        run_now = True

# —————— RESULTS ——————
if run_now and 'demand_df' in locals() and not demand_df.empty:
    with st.spinner("Training AI model & calculating optimal prices..."):
        results_df, avg_uplift, r2 = run_pricing_engine(
            retail_df if 'retail_df' in locals() else None,
            demand_df
        )

    # SUCCESS + BALLOONS
    st.markdown(f"""
    <div class="success-box">
        <h2>Done! Model R²: {r2:.3f}</h2>
        <h1 style='color:#10b981;'>Expected Revenue Boost: +{avg_uplift:.1f}%</h1>
    </div>
    """, unsafe_allow_html=True)
    st.balloons()

    # METRICS – GORGEOUS CARDS
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"<div class='metric-card'><h3>Products</h3><h1 style='color:#60a5fa;'>{len(results_df)}</h1></div>", unsafe_allow_html=True)
    with col2:
        up = len(results_df[results_df['price_change_%'] > 0])
        st.markdown(f"<div class='metric-card'><h3>Price Increases</h3><h1 style='color:#34d399;'>{up}</h1></div>", unsafe_allow_html=True)
    with col3:
        down = len(results_df[results_df['price_change_%'] < 0])
        st.markdown(f"<div class='metric-card'><h3>Price Decreases</h3><h1 style='color:#f87171;'>{down}</h1></div>", unsafe_allow_html=True)
    with col4:
        st.markdown(f"<div class='metric-card'><h3>Revenue Gain</h3><h1 style='color:#fbbf24;'>+{avg_uplift:.1f}%</h1><p>per day</p></div>", unsafe_allow_html=True)

    st.markdown("---")

    # RECOMMENDATIONS TABLE
    st.markdown("<h2 style='color:#e2e8f0;'>Optimal Pricing Recommendations</h2>", unsafe_allow_html=True)
    styled_df = results_df.sort_values("revenue_uplift_%", ascending=False).round(2)
    st.dataframe(styled_df, use_container_width=True, hide_index=True)

    # DOWNLOAD
    st.download_button(
        "Download Full Report (CSV)",
        data=results_df.to_csv(index=False).encode(),
        file_name=f"SmartPrice_Recommendations_{pd.Timestamp.now():%Y%m%d}.csv",
        mime="text/csv",
        use_container_width=True
    )

    # DEMAND CURVE – INTERACTIVE & BEAUTIFUL
    st.markdown("<h2 style='color:#e2e8f0;'>Demand & Revenue Curve Explorer</h2>", unsafe_allow_html=True)
    pid = st.selectbox("Select Product to Explore", results_df["product_id"].unique(), key="curve")
    row = results_df[results_df["product_id"] == pid].iloc[0]

    prices = np.linspace(row["current_price"]*0.7, row["current_price"]*1.5, 200)
    elasticity = 1.3
    sales = row["predicted_daily_units"] * (row["current_price"]/prices)**elasticity
    revenue = sales * prices

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), facecolor='#1e293b')
    fig.patch.set_facecolor('#1e293b')

    ax1.plot(prices, sales, color="#60a5fa", linewidth=4)
    ax1.axvline(row["optimal_price"], color="#f87171", linestyle="--", linewidth=3, label=f"Optimal ₹{row['optimal_price']}")
    ax1.set_title("Demand Curve", color="white", fontsize=16)
    ax1.set_xlabel("Price (₹)", color="#94a3b8")
    ax1.set_ylabel("Predicted Units Sold", color="#94a3b8")
    ax1.grid(True, alpha=0.3)
    ax1.legend(facecolor="#1e293b", labelcolor="white")

    ax2.plot(prices, revenue, color="#34d399", linewidth=4)
    ax2.axvline(row["optimal_price"], color="#f87171", linestyle="--", linewidth=3)
    ax2.set_title("Revenue Curve", color="white", fontsize=16)
    ax2.set_xlabel("Price (₹)", color="#94a3b8")
    ax2.set_ylabel("Predicted Revenue (₹)", color="#94a3b8")
    ax2.grid(True, alpha=0.3)

    for ax in (ax1, ax2):
        ax.spines['bottom'].set_color('#475569')
        ax.spines['top'].set_color('#475569')
        ax.spines['right'].set_color('#475569')
        ax.spines['left'].set_color('#475569')
        ax.tick_params(colors='#94a3b8')

    st.pyplot(fig)

    st.markdown("<p style='text-align:center; color:#64748b;'><em>Your data was processed securely in memory — never stored.</em></p>", unsafe_allow_html=True)

# —————— WELCOME PAGE ——————
else:
    st.markdown("""
    <div style='text-align: center; padding: 40px; background: linear-gradient(135deg, #1e293b, #334155); border-radius: 20px;'>
        <h2 style='color:#ff4b6e;'>Welcome to SmartPrice AI</h2>
        <p style='font-size:18px; color:#cbd5e1;'>
            Click the red button above for an instant demo<br>
            Or upload your sales data to get AI-powered pricing recommendations
        </p>
        <br>
        <h3 style='color:#94a3b8;'>100% Private • No Login • Works on Mobile</h3>
    </div>
    """, unsafe_allow_html=True)
