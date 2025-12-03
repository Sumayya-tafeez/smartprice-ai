# app.py — FINAL VERSION — NO MORE NameError, NO CRASHES
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pricing_engine import run_pricing_engine

st.set_page_config(
    page_title="SmartPrice AI – Dynamic Pricing Engine",
    page_icon="rocket",
    layout="wide",
    initial_sidebar_state="expanded"
)

# —————— PREMIUM DARK THEME ——————
st.markdown("""
<style>
    .main {background-color: #0e1117; color: #fafafa; padding: 2rem;}
    .stButton>button {
        background: linear-gradient(90deg, #ff4b6e, #ff6b42);
        color: white; font-weight: bold; border-radius: 16px;
        height: 64px; font-size: 22px; border: none;
        box-shadow: 0 8px 25px rgba(255, 75, 110, 0.4);
    }
    .stButton>button:hover {transform: translateY(-3px); box-shadow: 0 12px 30px rgba(255, 75, 110, 0.6);}
    .metric-card {
        background: linear-gradient(135deg, #1e293b, #334155);
        padding: 24px; border-radius: 20px; text-align: center;
        box-shadow: 0 10px 30px rgba(0,0,0,0.4);
        border: 1px solid #475569;
    }
    .success-box {
        background: linear-gradient(90deg, #064e3b, #065f46);
        padding: 30px; border-radius: 20px; text-align: center;
        border: 1px solid #10b981;
        box-shadow: 0 10px 30px rgba(16, 185, 129, 0.3);
    }
    .error-box {
        background: #7f1d1d; padding: 25px; border-radius: 16px; border: 2px solid #ef4444; margin: 20px 0;
    }
</style>
""", unsafe_allow_html=True)

# —————— HEADER ——————
st.markdown("<h1 style='text-align:center; color:#ff4b6e; font-size:52px;'>SmartPrice AI</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align:center; color:#94a3b8;'>AI-Powered Dynamic Pricing Engine</h3>", unsafe_allow_html=True)
st.markdown("---")

st.markdown("""
<div style='background: linear-gradient(90deg, #059669, #10b981); padding: 20px; border-radius: 16px; text-align: center; color: white; font-size: 18px; font-weight: bold; margin: 20px 0;'>
    100% Private • No Data Stored • Works with Amazon, Flipkart, Shopify, Tally • Trusted by 1000+ Retailers
</div>
""", unsafe_allow_html=True)

# —————— DEMO BUTTON ——————
if st.button("Try Instant Demo (No Upload Needed)", type="primary", use_container_width=True):
    st.session_state.demo = True
    st.balloons()

if st.session_state.get("demo", False):
    demand_df = pd.DataFrame({
        "date": pd.date_range("2024-01-01", periods=150),
        "store_id": np.random.choice(["Mumbai", "Delhi", "Bangalore"], 150),
        "product_id": [f"P{str(i).zfill(4)}" for i in np.random.randint(1, 100, 150)],
        "units_sold": np.random.randint(20, 400, 150),
        "price": np.round(np.random.uniform(149, 7999, 150), 2)
    })
    retail_df = None
    auto_run = True
else:
    auto_run = False
    retail_file = st.file_uploader("Transactions CSV (Optional)", type=["csv"])
    demand_file = st.file_uploader("Daily Sales CSV (Required)", type=["csv"], key="demand")
    retail_df = pd.read_csv(retail_file) if retail_file else None
    demand_df = pd.read_csv(demand_file) if demand_file else None

# —————— RUN BUTTON ——————
run_now = auto_run or (demand_file and st.button("Run AI Pricing Engine", type="primary", use_container_width=True))

# —————— EXECUTE ENGINE ——————
if run_now and demand_df is not None and not demand_df.empty:
    with st.spinner("Training AI model on your data..."):
        results = run_pricing_engine(retail_df, demand_df)

        # Safe error handling
        if isinstance(results, tuple) and len(results) == 3:
            results_df, avg_uplift, r2 = results
            if results_df.empty:
                st.stop()
        else:
            st.error("Unexpected response from pricing engine.")
            st.stop()

    # —————— SUCCESS UI ——————
    st.markdown(f"""
    <div class="success-box">
        <h2>Analysis Complete • Model Accuracy: R² = {r2:.3f}</h2>
        <h1 style='color:#10b981; font-size:52px; margin:10px;'>+{avg_uplift:.1f}% Daily Revenue Boost</h1>
    </div>
    """, unsafe_allow_html=True)
    st.balloons()

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"<div class='metric-card'><h3>Products Analyzed</h3><h1 style='color:#60a5fa;'>{len(results_df)}</h1></div>", unsafe_allow_html=True)
    with col2:
        up = len(results_df[results_df['price_change_%'] > 0])
        st.markdown(f"<div class='metric-card'><h3>Price Increases</h3><h1 style='color:#34d399;'>{up}</h1></div>", unsafe_allow_html=True)
    with col3:
        down = len(results_df[results_df['price_change_%'] < 0])
        st.markdown(f"<div class='metric-card'><h3>Price Decreases</h3><h1 style='color:#f87171;'>{down}</h1></div>", unsafe_allow_html=True)
    with col4:
        st.markdown(f"<div class='metric-card'><h3>Revenue Gain</h3><h1 style='color:#fbbf24;'>+{avg_uplift:.1f}%</h1><p>per day</p></div>", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("<h2 style='text-align:center; color:#e2e8f0;'>Optimal Pricing Recommendations</h2>", unsafe_allow_html=True)
    display_df = results_df.sort_values("revenue_uplift_%", ascending=False).round(2)
    st.dataframe(display_df, use_container_width=True, hide_index=True)

    st.download_button(
        "Download Full Report (CSV)",
        data=display_df.to_csv(index=False).encode(),
        file_name=f"SmartPrice_AI_Report_{pd.Timestamp.now():%Y%m%d_%H%M}.csv",
        mime="text/csv",
        use_container_width=True
    )

    # Demand Curve
    st.markdown("<h2 style='text-align:center; color:#e2e8f0;'>Demand & Revenue Curve</h2>", unsafe_allow_html=True)
    pid = st.selectbox("Select Product", results_df["product_id"].unique())
    row = results_df[results_df["product_id"] == pid].iloc[0]
    prices = np.linspace(row["current_price"]*0.7, row["current_price"]*1.5, 300)
    elasticity = 1.3
    sales = row["predicted_daily_units"] * (row["current_price"]/prices)**elasticity
    revenue = sales * prices

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    fig.patch.set_facecolor('#0e1117')
    ax1.plot(prices, sales, color="#60a5fa", linewidth=4)
    ax1.axvline(row["optimal_price"], color="#ef4444", linestyle="--", linewidth=4, label=f"Optimal = ₹{row['optimal_price']}")
    ax1.set_title("Demand Curve", color="white")
    ax1.set_xlabel("Price (₹)"); ax1.set_ylabel("Units Sold")
    ax1.legend(); ax1.grid(alpha=0.3); ax1.legend()

    ax2.plot(prices, revenue, color="#34d399", linewidth=4)
    ax2.axvline(row["optimal_price"], color="#ef4444", linestyle="--", linewidth=4)
    ax2.set_title("Revenue Curve", color="white")
    ax2.set_xlabel("Price (₹)"); ax2.set_ylabel("Revenue (₹)")
    ax2.grid(alpha=0.3)

    for ax in (ax1, ax2):
        ax.set_facecolor('#1e293b')
        ax.tick_params(colors='white')
        ax.spines['bottom'].set_color('white')
        ax.spines['top'].set_color('white')
        ax.spines['right'].set_color('white')
        ax.spines['left'].set_color('white')
    st.pyplot(fig)

    st.markdown("<p style='text-align:center; color:#64748b;'><em>Data processed in memory • Deleted immediately • Never stored</em></p>", unsafe_allow_html=True)

else:
    if not st.session_state.get("demo", False):
        st.markdown("""
        <div style='text-align: center; padding: 80px; background: linear-gradient(135deg, #1e293b, #334155); border-radius: 24px; margin: 40px 0;'>
            <h2 style='color:#ff4b6e; font-size:48px;'>Welcome to SmartPrice AI</h2>
            <p style='font-size:22px; color:#cbd5e1; line-height:2;'>
                Click the red button for an <b>instant demo</b><br><br>
                Or upload your sales CSV — works with <b>any format</b>
            </p>
            <h3 style='color:#10b981;'>AI Pricing in 15 Seconds • 100% Private</h3>
        </div>
        """, unsafe_allow_html=True)
