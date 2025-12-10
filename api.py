# api.py — FINAL WORKING VERSION (DEC 2025)
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pricing_engine import run_pricing_engine
import pandas as pd

app = FastAPI(title="SmartPrice AI – Live Pricing Engine")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load sample data once
df = pd.read_csv("sample_data.csv")

@app.get("/")
def home():
    return {"message": "SmartPrice AI Engine is LIVE"}

@app.post("/optimize")
def optimize(scenario: str = "normal"):
    data = df.copy()
    
    # Simulate real-world demand scenarios
    multiplier = {"normal": 1.0, "blackfriday": 2.1, "low_demand": 0.4}
    data['Units Sold'] = (data['Units Sold'] * multiplier.get(scenario, 1.0)).astype(int)

    # Run your full engine
    results_df, avg_uplift, r2 = run_pricing_engine(data)

    # Add product name to results (for beautiful display)
    product_map = df[['Product ID', 'Product Name']].drop_duplicates().set_index('Product ID')['Product Name'].to_dict()
    results_df['Product Name'] = results_df['product_id'].map(product_map)
    results_df['Current Price'] = results_df['current_price']
    results_df['Optimal Price'] = results_df['optimal_price']
    results_df['Revenue Uplift %'] = results_df['revenue_uplift_%']

    return {
        "products": results_df[[
            'Product ID', 'Product Name', 'Current Price', 'Optimal Price', 
            'Revenue Uplift %'
        ]].round(2).to_dict(orient="records"),
        "summary": {
            "avg_uplift": float(avg_uplift),
            "r2_score": float(r2),
            "total_products": len(results_df),
            "scenario": scenario.title()
        }
    }
