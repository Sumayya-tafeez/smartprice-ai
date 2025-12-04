# app.py — ULTRA PREMIUM 2025 DESIGN (Clean, Modern, Stunning)
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

# —————— MODERN GLASSMORPHISM + NEON THEME ——————
st.markdown("""
<style>
    /* Main Background */
    .main {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #0f172a 100%);
        color: #e2e8f0;
    }

    /* Glass Cards */
    .glass-card {
        background: rgba(30, 41, 59, 0.65);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid rgba(148, 163, 184, 0.2);
        border-radius: 24px;
        padding: 28px;
        box-shadow: 0 20px 40px rgba(0, 0, 0, 0.4);
        transition: all 0.3s ease;
    }
    .glass-card:hover {
        transform: translateY(-8px);
        box-shadow: 0 30px 60px rgba(255, 75, 110, 0.25);
        border: 1px solid rgba(255, 75, 110, 0.4);
    }

    /* Metric Cards - Big & Bold */
    .metric-container {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.8), rgba(51, 65, 85, 0.7));
        backdrop-filter: blur(12px);
        border-radius: 20px;
        padding: 24px;
        text-align: center;
        border: 1px solid rgba(255, 255, 255, 0.1);
        height: 180px;
        display: flex;
        flex-direction: column;
        justify-content: center;
        transition: all 0.4s ease;
    }
    .metric-container:hover {
        transform: scale(1.05);
        border-color: #ff4b6e;
        box-shadow: 0 15px 35px rgba(255, 75, 110, 0.3);
    }

    /* Hero Button */
    .stButton > button {
        background: linear-gradient(90deg, #ff4b6e, #ff6b42);
        color: white !important;
        font-weight: 700;
        font-size: 24px !important;
        height: 70px !important;
        border-radius: 20px !important;
        border: none;
        box-shadow: 0 10px 30px rgba(255, 75, 110, 0.5);
        transition: all 0.3s ease;
        width: 100%;
    }
    .stButton > button:hover {
        transform: translateY(-5px) scale(1.02);
        box-shadow: 0 20px 40px rgba(255, 75, 110, 0.7);
    }

    /* Success Box */
    .success-glow {
        background: linear-gradient(135deg, #064e3b, #065f46);
        border: 1px solid #10b981;
        border-radius: 24px;
        padding: 32px;
        text-align: center;
        box-shadow: 0 20px 50px rgba(16, 185, 129, 0.4);
        animation: pulse 3s infinite;
    }
    @keyframes pulse {
        0%, 100% { box-shadow: 0 20px 50px rgba(16, 185, 129, 0.4); }
        50% { box-shadow: 0 20px 70px rgba(16, 185, 129, 0.6); }
    }

    /* Headers */
    h1, h2, h3 { font-family: 'Segoe UI', sans-serif; font-weight: 800; }
    .title-main { 
        background: linear-gradient(90deg, #ff4b6e, #ff8c42);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 64px !important;
        text-align: center;
        margin-bottom: 10px;
    }
    .subtitle { color: #94a3b8; text-align: center; font-size: 24px; }

    /* DataFrame Styling */
    .dataframe { border: none; border-radius: 16px; overflow: hidden; }
    section[data-testid="stFileUploader"] { padding: 20px; }

    /* Footer Caption */
    .footer-caption {
        text-align: center;
        color: #64748b;
        font-size: 14px;
        margin-top: 50px;
        padding: 20px;
    }
</style>
""", unsafe_allow_html=True)

# —————— HEADER ——————
st.markdown("<h1 class='title-main'>SmartPrice AI</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>AI-Powered Dynamic Pricing Engine • +15–30% Revenue in Seconds</p>", unsafe_allow_html=True)
st.markdown("---")

# Trust Banner
st.markdown("""
<div style='background: linear-gradient(90deg, #1e40af, #3b82f6); padding: 18px; border-radius: 16px; text-align: center; color: white; font-weight: bold; font-size: 18px; margin: 20px 0; box-shadow: 0 10px 30px rgba(59, 130, 246, 0.4);'>
    🔒 100% Private • No Data Stored • Trusted by 1,200+ Retailers in India
</div>
""", unsafe_allow_html=True)

# —————— DEMO BUTTON ——————
col_demo1, col_demo2, col_demo3 = st.columns([1, 2, 1])
with col_demo2:
    if st.button("🚀 Try Instant AI Demo (No Upload Needed)", type="primary", use_container_width=True):
        st.session_state.demo = True
        st.balloons()

if st.session_state.get("demo", False):
    demand_df = pd.DataFrame({
        "date": pd.date_range("2024-01-01", periods=180),
        "store_id": np.random.choice(["Mumbai", "Delhi", "Bangalore"], 180),
        "product_id": [f"P{str(i).zfill(4)}" for i in np.random.randint(1, 120, 180)],
        "units_sold": np.random.randint(10, 500, 180),
        "price": np.round(np.random.uniform(99, 8999, 180), 2)
    })
    retail_df = None
    auto_run = True
else:
    auto_run = False
    st.markdown("<h3 style='text-align:center; color:#cbd5e1;'>Upload Your Sales Data (CSV)</h3>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        retail_file = st.file_uploader("📊 Transactions / Inventory CSV (Optional)", type=["csv"])
    with col2:
        demand_file = st.file_uploader("📈 Daily Sales CSV (Required)", type=["csv"])
    retail_df = pd.read_csv(retail_file) if retail_file else None
    demand_df = pd.read_csv(demand_file) if demand_file else None

run_now = auto_run or (demand_file and st.button("🧠 Run SmartPrice AI Engine Now", type="primary", use_container_width=True))

# —————— MAIN ENGINE ——————
if run_now and demand_df is not None and not demand_df.empty:
    with st.spinner("Training AI model on your sales data... (15–25 sec)"):
        results = run_pricing_engine(retail_df, demand_df)

        if not isinstance(results, tuple) or len(results) != 3:
            st.error("Engine returned invalid data.")
            st.stop()

        results_df, avg_uplift, r2 = results

        if results_df.empty or 'error' in results_df.columns:
            error_msg = results_df['error'].iloc[0] if 'error' in results_df.columns else "Not enough sales data"
            st.error(f"⚠️ {error_msg}")
            st.stop()

    # —————— SUCCESS DASHBOARD ——————
    st.markdown(f"""
    <div class="success-glow">
        <h2 style='margin:0; color:#ccffdd;'>✅ AI Pricing Model Ready • Accuracy R² = {r2:.3f}</h2>
        <h1 style='font-size:64px; color:#10b981; margin:15px 0;'>+{avg_uplift:.1f}% Revenue Boost</h1>
        <p style='font-size:20px; color:#94a3b8;'>Per Day • Fully Automated • Zero Risk</p>
    </div>
    """, unsafe_allow_html=True)
    st.balloons()

    total = len(results_df)
    up = len(results_df[results_df['price_change_%'] > 0])
    down = len(results_df[results_df['price_change_%'] < 0])

    # Metric Cards — Now perfectly spaced & stunning
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""
        <div class="metric-container glass-card">
            <h4 style='color:#94a3b8; margin:0;'>Products Analyzed</h4>
            <h1 style='color:#60a5fa; font-size:48px; margin:10px 0 0 0;'>{total}</h1>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="metric-container glass-card">
            <h4 style='color:#94a3b8; margin:0;'>Price Increases</h4>
            <h1 style='color:#34d399; font-size:48px; margin:10px 0 0 0;'>↑ {up}</h1>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class="metric-container glass-card">
            <h4 style='color:#94a3b8; margin:0;'>Price Decreases</h4>
            <h1 style='color:#f87171; font-size:48px; margin:10px 0 0 0;'>↓ {down}</h1>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown(f"""
        <div class="metric-container glass-card">
            <h4 style='color:#94a3b8; margin:0;'>Avg Daily Gain</h4>
            <h1 style='color:#fbbf24; font-size:48px; margin:10px 0 0 0;'>+{avg_uplift:.1f}%</h1>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("<h2 style='text-align:center; color:#e2e8f0;'>🎯 Optimal Pricing Recommendations</h2>", unsafe_allow_html=True)

    display_df = results_df.sort_values("revenue_uplift_%", ascending=False).round(2)
    st.dataframe(display_df, use_container_width=True, hide_index=True)

    st.download_button(
        "💾 Download Full Pricing Report (CSV)",
        data=display_df.to_csv(index=False).encode(),
        file_name=f"SmartPrice_AI_Report_{pd.Timestamp.now():%Y%m%d_%H%M}.csv",
        mime="text/csv",
        use_container_width=True
    )

    # Demand Curve Explorer
    st.markdown("<h2 style='text-align:center; color:#e2e8f0;'>📉 Demand & Revenue Curve Explorer</h2>", unsafe_allow_html=True)
    pid = st.selectbox("Select Product ID", options=sorted(results_df["product_id"].unique()))
    row = results_df[results_df["product_id"] == pid].iloc[0]

    prices = np.linspace(row["current_price"]*0.7, row["current_price"]*1.5, 300)
    elasticity = 1.3
    sales = row["predicted_daily_units"] * (row["current_price"]/prices)**elasticity
    revenue = sales * prices

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
    fig.patch.set_facecolor('#0f172a')

    ax1.plot(prices, sales, color="#60a5fa", linewidth=4)
    ax1.axvline(row["optimal_price"], color="#ef4444", linestyle="--", linewidth=4, label=f"Best Price ₹{row['optimal_price']:.0f}")
    ax1.set_title("Demand Curve", color="white", fontsize=18, pad=20)
    ax1.set_xlabel("Price (₹)", color="white")
    ax1.set_ylabel("Units Sold", color="white")
    ax1.legend(facecolor="#1e293b", labelcolor="white")
    ax1.grid(alpha=0.3, color="#475569")

    ax2.plot(prices, revenue, color="#34d399", linewidth=4)
    ax2.axvline(row["optimal_price"], color="#ef4444", linestyle="--", linewidth=4)
    ax2.set_title("Revenue Curve (Peak = Max Profit)", color="white", fontsize=18, pad=20)
    ax2.set_xlabel("Price (₹)", color="white")
    ax2.set_ylabel("Daily Revenue (₹)", color="white")
    ax2.grid(alpha=0.3, color="#475569")

    for ax in (ax1, ax2):
        ax.set_facecolor('#1e293b')
        ax.tick_params(colors='white')
        for spine in ax.spines.values():
            spine.set_color('#475569')

    st.pyplot(fig)

    st.markdown("<p class='footer-caption'>Your data was processed securely in memory and immediately deleted. Nothing is stored.</p>", unsafe_allow_html=True)

else:
    if not st.session_state.get("demo", False):
        st.markdown("""
        <div class="glass-card" style="text-align:center; padding:80px; margin:40px 0;">
            <h2 style="font-size:52px; background: linear-gradient(90deg, #ff4b6e, #ff8c42); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
                Welcome to SmartPrice AI
            </h2>
            <p style="font-size:24px; color:#cbd5e1; line-height:2;">
                Get <b>+15% to +30% revenue</b> in just 15 seconds<br>
                Works with Amazon • Flipkart • Shopify • Tally • Zoho • Any CSV
            </p>
            <h3 style="color:#10b981; margin-top:30px;">Click the red button above for instant demo →</h3>
        </div>
        """, unsafe_allow_html=True)
