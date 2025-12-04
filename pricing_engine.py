# pricing_engine.py — FINAL BULLETPROOF VERSION (DEC 2025)
# Works with ANY column naming: customer_id, CustomerID, price, Price, qty, etc.

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import warnings
warnings.filterwarnings('ignore')

def run_pricing_engine(df):
    try:
        df = df.copy()

        # Step 1: Normalize all column names
        df.columns = [c.strip().lower().replace(' ', '_').replace('-', '_') for c in df.columns]

        # Step 2: SUPER FLEXIBLE COLUMN MAPPING (this is the magic)
        df.rename(columns={
            # Customer ID variations
            'customerid': 'customer_id', 'cust_id': 'customer_id', 'customer': 'customer_id',
            'custid': 'customer_id', 'client_id': 'customer_id', 'cust': 'customer_id',
            'customer_id': 'customer_id',  # already correct

            # Price variations
            'selling_price': 'price', 'unit_price': 'price', 'rate': 'price',
            'amount': 'price', 'sale_price': 'price', 'sellingprice': 'price',
            'unitprice': 'price', 'price': 'price',  # already correct

            # Quantity variations
            'quantity': 'units_sold', 'qty': 'units_sold', 'units': 'units_sold',
            'sales_qty': 'units_sold', 'unit_sold': 'units_sold', 'units_sold': 'units_sold',

            # Date variations
            'order_date': 'date', 'invoice_date': 'date', 'transaction_date': 'date',
            'orderdate': 'date', 'date': 'date',

            # Product ID variations
            'sku': 'product_id', 'item_code': 'product_id', 'productid': 'product_id',
            'item_id': 'product_id', 'product': 'product_id', 'product_id': 'product_id'
        }, inplace=True)

        # REQUIRED COLUMNS
        required = ['date', 'customer_id', 'product_id', 'units_sold', 'price']
        missing = [col for col in required if col not in df.columns]
        if missing:
            return pd.DataFrame({'error': [f"Missing columns: {', '.join(missing)}"]}), 0, 0

        # Clean data
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df = df.dropna(subset=required)
        if len(df) < 5:
            return pd.DataFrame({'error': ['Need at least 5 transactions']}), 0, 0

        # === 1. RFM + KMeans Customer Segmentation ===
        latest = df['date'].max()
        rfm = df.groupby('customer_id').agg({
            'date': lambda x: (latest - x.max()).days,
            'customer_id': 'count',
            'price': lambda x: (x * df.loc[x.index, 'units_sold']).sum()
        }).rename(columns={'date': 'recency', 'customer_id': 'frequency', 'price': 'monetary'})

        rfm['monetary'] = rfm['monetary'].replace(0, 1)
        scaler = StandardScaler()
        rfm_scaled = scaler.fit_transform(rfm[['recency', 'frequency', 'monetary']])
        kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
        rfm['segment'] = kmeans.fit_predict(rfm_scaled)

        segment_names = {0: "VIP Loyal", 1: "Price Hunters", 2: "At Risk", 3: "Sleeping Giants"}
        rfm['segment_name'] = rfm['segment'].map(segment_names)
        df = df.merge(rfm[['segment_name']], on='customer_id', how='left')

        # === 2. Daily aggregation per product + segment ===
        daily = df.groupby(['date', 'product_id', 'segment_name'], as_index=False).agg({
            'units_sold': 'sum',
            'price': 'median'
        }).rename(columns={'units_sold': 'sales'})

        daily['dow'] = daily['date'].dt.dayofweek
        daily['month'] = daily['date'].dt.month
        daily['price_log'] = np.log1p(daily['price'])

        # === 3. Train XGBoost model ===
        encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore', drop='first')
        X_cat = encoder.fit_transform(daily[['product_id', 'segment_name']])
        X_num = daily[['price', 'price_log', 'dow', 'month']].values
        X = np.hstack([X_num, X_cat])
        y = daily['sales'].values

        model = xgb.XGBRegressor(n_estimators=600, max_depth=6, learning_rate=0.05,
                                 subsample=0.9, colsample_bytree=0.9, random_state=42, n_jobs=-1)

        # ======= MODIFIED: Train-test split for realistic R² =======
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model.fit(X_train, y_train)
        r2 = round(r2_score(y_test, model.predict(X_test)), 3)
        # ============================================================

        # === 4. Generate personalized pricing recommendations ===
        results = []
        baseline = daily.groupby(['product_id', 'segment_name'], as_index=False).agg({
            'price': 'median', 'sales': 'mean'
        }).rename(columns={'price': 'current_price', 'sales': 'current_sales'})

        for _, row in baseline.iterrows():
            prices = np.linspace(row['current_price']*0.7, row['current_price']*1.4, 30)
            best_price = row['current_price']
            best_rev = row['current_price'] * row['current_sales']

            cat_vec = encoder.transform(pd.DataFrame([{
                'product_id': row['product_id'],
                'segment_name': row['segment_name']
            }]))[0]
            base_num = daily[['price', 'price_log', 'dow', 'month']].median()

            for p in prices:
                vec = base_num.copy()
                vec['price'] = p
                vec['price_log'] = np.log1p(p)
                pred = max(1, model.predict(np.hstack([vec.values, cat_vec]).reshape(1, -1))[0])
                if p * pred > best_rev:
                    best_rev = p * pred
                    best_price = p

            uplift = round((best_rev / (row['current_price'] * row['current_sales']) - 1) * 100, 1)
            results.append({
                'product_id': row['product_id'],
                'customer_segment': row['segment_name'],
                'segment_size': len(df[df['segment_name'] == row['segment_name']]['customer_id'].unique()),
                'current_price': round(row['current_price'], 2),
                'optimal_price': round(best_price, 2),
                'price_change_%': round((best_price / row['current_price'] - 1) * 100, 1),
                'revenue_uplift_%': uplift
            })

        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values(['customer_segment', 'revenue_uplift_%'], ascending=[True, False])
        avg_uplift = round(results_df['revenue_uplift_%'].mean(), 1)

        return results_df, avg_uplift, r2

    except Exception as e:
        return pd.DataFrame({'error': [f"Error: {str(e)}"]}), 0, 0
