# app.py — FINAL 100% STABLE VERSION (works with latest pricing_engine.py)
import streamlit as st
import pandas as pd

# === Initialize session state ===
if "demo" not in st.session_state:
    st.session_state.demo = False

# === Styling ===
st.set_page_config(page_title="Dynamic Pricing Engine", layout="wide", page_icon="chart_with_upwards_trend")
st.markdown("""
<style>
    .main {background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%); color: #e2e8f0;}
    .title {font-size: 54px; font-weight: 800; text-align: center;
            background: linear-gradient(90deg, #60a5fa, #a78bfa);
            -webkit-background-clip: text; -webkit-text-fill-color: transparent;}
    .subtitle {text-align: center; color: #94a3b8; font-size: 22px; margin-bottom: 40px;}
    .demo-btn > button {background: linear-gradient(90deg, #3b82f6, #8b5cf6) !important;
                        color: white !important; font-size: 20px !important; height: 65px !important;
                        border-radius: 16px !important; box-shadow: 0 8px 25px rgba(59,130,246,0.4) !important;}
    .run-btn > button {background: linear-gradient(90deg, #10b981, #34d399) !important; color: white !important;}
    .segment-card {background: rgba(30,41,59,0.8); backdrop-filter: blur(10px);
                   border-radius: 16px; padding: 24px; margin: 20px 0;
                   border-left: 6px solid; box-shadow: 0 10px 30px rgba(0,0,0,0.3);}
    .vip {border-left-color: #f59e0b;}
    .hunter {border-left-color: #ef4444;}
    .risk {border-left-color: #f97316;}
    .sleeping {border-left-color: #8b5cf6;}
</style>
""", unsafe_allow_html=True)

st.markdown("<h1 class='title'>Dynamic Pricing Engine</h1>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Customer Segmentation + Intelligent Price Optimization</div>", unsafe_allow_html=True)

col1, col2, col3 = st.columns([1,2,1])
with col2:
    if st.button("Try Live Demo (Instant Results)", key="demo_btn", use_container_width=True):
        st.session_state.demo = True
        st.rerun()

uploaded_file = st.file_uploader("Upload your sales data (CSV with customer_id)", type=["csv"])

if st.button("Run Analysis", type="primary", key="run", use_container_width=True) or st.session_state.demo:
    with st.spinner("Analyzing data..." if not st.session_state.demo else "Loading demo..."):
        
        if st.session_state.demo:
            # Hard-coded beautiful demo result
            results_df = pd.DataFrame([
                {"product_id": "P0015", "customer_segment": "VIP Loyal", "segment_size": 189, "current_price": 363.46, "optimal_price": 438.20, "price_change_%": 20.6, "revenue_uplift_%": 32.8},
                {"product_id": "P0008", "customer_segment": "VIP Loyal", "segment_size": 189, "current_price": 329.73, "optimal_price": 395.60, "price_change_%": 20.0, "revenue_uplift_%": 30.1},
                {"product_id": "P0012", "customer_segment": "Price Hunters", "segment_size": 312, "current_price": 26.75, "optimal_price": 21.40, "price_change_%": -20.0, "revenue_uplift_%": 44.3},
                {"product_id": "P0005", "customer_segment": "Price Hunters", "segment_size": 312, "current_price": 73.64, "optimal_price": 58.91, "price_change_%": -20.0, "revenue_uplift_%": 41.7},
                {"product_id": "P0018", "customer_segment": "At Risk", "segment_size": 98, "current_price": 147.27, "optimal_price": 125.18, "price_change_%": -15.0, "revenue_uplift_%": 28.4},
                {"product_id": "P0020", "customer_segment": "Sleeping Giants", "segment_size": 67, "current_price": 115.23, "optimal_price": 132.51, "price_change_%": 15.0, "revenue_uplift_%": 24.9},
            ])
            avg_uplift = 33.7
            r2 = 0.94
        else:
            from pricing_engine import run_pricing_engine
            df = pd.read_csv(uploaded_file)
            results_df, avg_uplift, r2 = run_pricing_engine(df)
            if 'error' in results_df.columns:
                st.error(results_df['error'].iloc[0])
                st.stop()

    st.success(f"Analysis Complete — Model Accuracy R²: {r2} — Average Revenue Increase: +{avg_uplift}%")

    # FIXED: Use 'segment_size' instead of 'customers_in_segment'
    for segment in ["VIP Loyal", "Price Hunters", "At Risk", "Sleeping Giants"]:
        seg_data = results_df[results_df['customer_segment'] == segment]
        if len(seg_data) == 0:
            continue
        customers = seg_data['segment_size'].iloc[0] if 'segment_size' in seg_data.columns else "N/A"
        uplift = seg_data['revenue_uplift_%'].mean()
        color_class = {"VIP Loyal":"vip", "Price Hunters":"hunter", "At Risk":"risk", "Sleeping Giants":"sleeping"}[segment]

        st.markdown(f"""
        <div class="segment-card {color_class}">
            <h2 style="margin:0; color:white;">{segment}</h2>
            <h3 style="margin:5px 0;">{customers} customers • Avg. Uplift: <span style="color:#34d399; font-size:28px;">+{uplift:.1f}%</span></h3>
        </div>
        """, unsafe_allow_html=True)

        st.dataframe(seg_data[['product_id', 'current_price', 'optimal_price', 'price_change_%', 'revenue_uplift_%']].head(8),
                     use_container_width=True, hide_index=True)

    st.download_button("Download Full Pricing Report (CSV)",
                       data=results_df.to_csv(index=False).encode(),
                       file_name=f"pricing_report_{pd.Timestamp.now().strftime('%Y%m%d')}.csv",
                       mime="text/csv", use_container_width=True)

    if st.session_state.demo:
        st.info("This was a demo. Upload your real data for actual results.")
        if st.button("Clear Demo"):
            st.session_state.demo = False
            st.rerun()
