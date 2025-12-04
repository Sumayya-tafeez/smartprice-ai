# app.py — FINAL COLORFUL PROFESSIONAL + DEMO BUTTON
import streamlit as st
import pandas as pd
import numpy as np
from pricing_engine import run_pricing_engine

st.set_page_config(page_title="Dynamic Pricing Engine", layout="wide", page_icon="chart_with_upwards_trend")

# Beautiful modern styling
st.markdown("""
<style>
    .main {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        color: #e2e8f0;
    }
    .title {
        font-size: 54px;
        font-weight: 800;
        text-align: center;
        background: linear-gradient(90deg, #60a5fa, #a78bfa);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 10px;
    }
    .subtitle {
        text-align: center;
        color: #94a3b8;
        font-size: 22px;
        margin-bottom: 40px;
    }
    .demo-btn {
        background: linear-gradient(90deg, #3b82f6, #8b5cf6) !important;
        color: white !important;
        font-size: 20px !important;
        height: 65px !important;
        border-radius: 16px !important;
        box-shadow: 0 8px 25px rgba(59, 130, 246, 0.4) !important;
    }
    .run-btn {
        background: linear-gradient(90deg, #10b981, #34d399) !important;
        color: white !important;
    }
    .segment-card {
        background: rgba(30, 41, 59, 0.8);
        backdrop-filter: blur(10px);
        border-radius: 16px;
        padding: 24px;
        margin: 20px 0;
        border-left: 6px solid;
        box-shadow: 0 10px 30px rgba(0,0,0,0.3);
    }
    .vip { border-left-color: #f59e0b; }
    .hunter { border-left-color: #ef4444; }
    .risk { border-left-color: #f97316; }
    .sleeping { border-left-color: #8b5cf6; }
</style>
""", unsafe_allow_html=True)

st.markdown("<h1 class='title'>Dynamic Pricing Engine</h1>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Customer Segmentation + Intelligent Price Optimization</div>", unsafe_allow_html=True)

# Demo Button
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    if st.button("Try Live Demo (See Results Instantly)", key="demo", use_container_width=True):
        st.session_state.demo = True
        st.rerun()

# File uploader
uploaded_file = st.file_uploader("Upload your sales data (CSV with customer_id)", type=["csv"])

if st.button("Run Analysis", type="primary", use_container_width=True) or st.session_state.get("demo"):
    with st.spinner("Training AI model and optimizing prices..." if not st.session_state.get("demo") else "Loading demo results..."):

        if st.session_state.get("demo"):
            # === REALISTIC DEMO DATA (Looks 100% real) ===
            demo_results = pd.DataFrame([
                {"product_id": "P0015", "customer_segment": "VIP Loyal", "customers_in_segment": 189, "current_price": 363.46, "optimal_price": 438.20, "price_change_%": 20.6, "revenue_uplift_%": 32.8},
                {"product_id": "P0008", "customer_segment": "VIP Loyal", "customers_in_segment": 189, "current_price": 329.73, "optimal_price": 395.60, "price_change_%": 20.0, "revenue_uplift_%": 30.1},
                {"product_id": "P0012", "customer_segment": "Price Hunters", "customers_in_segment": 312, "current_price": 26.75, "optimal_price": 21.40, "price_change_%": -20.0, "revenue_uplift_%": 44.3},
                {"product_id": "P0005", "customer_segment": "Price Hunters", "customers_in_segment": 312, "current_price": 73.64, "optimal_price": 58.91, "price_change_%": -20.0, "revenue_uplift_%": 41.7},
                {"product_id": "P0018", "customer_segment": "At Risk", "customers_in_segment": 98, "current_price": 147.27, "optimal_price": 125.18, "price_change_%": -15.0, "revenue_uplift_%": 28.4},
                {"product_id": "P0020", "customer_segment": "Sleeping Giants", "customers_in_segment": 67, "current_price": 115.23, "optimal_price": 132.51, "price_change_%": 15.0, "revenue_uplift_%": 24.9},
            ])
            results_df, avg_uplift, r2 = demo_results, 33.7, 0.94
        else:
            df = pd.read_csv(uploaded_file)
            results_df, avg_uplift, r2 = run_pricing_engine(df)
            if 'error' in results_df.columns:
                st.error(results_df['error'].iloc[0])
                st.stop()

    # === RESULTS ===
    st.success(f"Analysis Complete — Model Accuracy R²: {r2} — Average Revenue Increase: +{avg_uplift}%")

    segments = {"VIP Loyal": "vip", "Price Hunters": "hunter", "At Risk": "risk", "Sleeping Giants": "sleeping"}
    
    for segment_name, css_class in segments.items():
        seg_data = results_df[results_df['customer_segment'] == segment_name]
        if len(seg_data) == 0:
            continue
        uplift = seg_data['revenue_uplift_%'].mean()
        customers = seg_data['customers_in_segment'].iloc[0]

        st.markdown(f"""
        <div class="segment-card {css_class}">
            <h2 style="margin:0; color:white;">{segment_name}</h2>
            <h3 style="margin:5px 0; color:#60a5fa;">{customers} customers • Average Uplift: <span style="color:#34d399; font-size:28px;">+{uplift:.1f}%</span></h3>
            <p style="color:#94a3b8; margin:10px 0;">
                { "Increase price by 10–25% — high loyalty" if "VIP" in segment_name else
                  "Decrease price by 10–20% — volume explosion" if "Price Hunters" in segment_name else
                  "Apply 12–18% discount — win-back strategy" if "At Risk" in segment_name else
                  "15–20% premium pricing — high past value" }
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.dataframe(
            seg_data[['product_id', 'current_price', 'optimal_price', 'price_change_%', 'revenue_uplift_%']].head(8),
            use_container_width=True,
            hide_index=True
        )

    st.download_button(
        label="Download Full Pricing Report (CSV)",
        data=results_df.to_csv(index=False).encode(),
        file_name=f"pricing_report_{pd.Timestamp.now().strftime('%Y%m%d')}.csv",
        mime="text/csv",
        use_container_width=True
    )

    if st.session_state.get("demo"):
        st.info("This was a demo using sample data. Upload your file for real results.")
