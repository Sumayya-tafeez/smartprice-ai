from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pricing_engine import run_pricing_engine  # your existing function
import pandas as pd

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

df = pd.read_csv("sample_data.csv")

@app.get("/")
def home():
    return {"message": "SmartPrice AI Engine Running – Live Demo Ready"}

@app.post("/optimize")
def optimize(scenario: str = "normal"):
    data = df.copy()
    if scenario == "blackfriday":
        data['Units Sold'] *= 1.9
    elif scenario == "low_demand":
        data['Units Sold'] *= 0.5
    data['Units Sold'] = data['Units Sold'].astype(int)

    results = run_pricing_engine(data)

    # Make sure these columns exist in your results
    results['Revenue Uplift %'] = results.get('Revenue Uplift %', 0)
    results['Recommendation'] = results.get('Recommendation', 'Hold')

    return {
        "products": results[[
            'Product ID', 'Product Name', 'Price', 'Optimal Price', 
            'Revenue Uplift %', 'Recommendation'
        ]].round(2).to_dict(orient="records"),
        "summary": {"avg_uplift": round(results['Revenue Uplift %'].mean(), 1)}
    }
