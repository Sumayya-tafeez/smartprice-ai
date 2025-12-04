# app.py — CLEAN ENTERPRISE VERSION (No fluff, no balloons, no marketing lines)
import streamlit as st
import pandas as pd
from pricing_engine import run_pricing_engine

st.set_page_config(page_title="Dynamic Pricing Engine", layout="wide")

# Minimal professional styling
st.markdown("""
<style>
    .main {background-color: #0f172a; color: #e2e8f0;}
    .stButton>button {
        background: #1e40af; color: white; font-size: 18px; height: 60px; border-radius: 12px;
    }
    .title {font-size: 48px; text-align: center; font-weight: 700; color: #e2e8f0;}
    .segment-box {
        background: #1e293b; padding: 20px; border-radius: 12px; margin: 15px 0;
        border-left: 5px solid #3b82f6;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("<h1 class='title'>Dynamic Pricing Engine</h1>", unsafe_allow_html=True)
st.markdown("### Customer Segmentation + Price Optimization")

uploaded_file = st.file_uploader("Upload sales data (CSV with customer_id)", type=["csv"])

if uploaded_file is not None:
    if st.button("Run Analysis", type="primary", use_container_width=True):
        with st.spinner("Processing data and training model..."):
            df = pd.read_csv(uploaded_file)
            results_df, avg_uplift, r2 = run_pricing_engine(df)

            if 'error' in results_df.columns:
                st.error(f"Error: {results_df['error'].iloc[0]}")
            else:
                st.success(f"Analysis Complete — Model Accuracy (R²): {r2} — Projected Revenue Increase: +{avg_uplift}%")

                for segment in results_df['customer_segment'].unique():
                    seg_data = results_df[results_df['customer_segment'] == segment].head(10)
                    uplift = seg_data['revenue_uplift_%'].mean()
                    st.markdown(f"""
                    <div class="segment-box">
                        <h3>{segment} — {seg_data['customers_in_segment'].iloc[0]} customers — Avg. Uplift: +{uplift:.1f}%</h3>
                    </div>
                    """, unsafe_allow_html=True)
                    st.dataframe(seg_data[['product_id', 'current_price', 'optimal_price', 'price_change_%', 'revenue_uplift_%']],
                                 use_container_width=True, hide_index=True)

                st.download_button(
                    label="Download Complete Pricing Report",
                    data=results_df.to_csv(index=False).encode(),
                    file_name="pricing_optimization_report.csv",
                    mime="text/csv"
                )
