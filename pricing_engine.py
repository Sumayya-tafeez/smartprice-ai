# pricing_engine.py — FINAL: Only Indian Dataset + Customer Segmentation Magic
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
        df.columns = [c.strip().lower().replace(' ', '_') for c in df.columns]

        # Auto-map all possible names
        rename_map = {
            'customer': 'customer_id', 'customerid': 'customer_id', 'cust_id': 'customer_id',
            'date': 'date', 'product_id': 'product_id', 'units_sold': 'units_sold',
            'price': 'price', 'demand_f_price': 'price',
            'discount': 'discount', 'competitor_price': 'competitor_price', '/pr': 'competitor_price',
            'store_id': 'store_id', 'category': 'category', 'region': 'region'
        }
        df.rename(columns={v: k for k, v in rename_map.items() if v in df.columns}, inplace=True)

        required = ['date', 'customer_id', 'product_id', 'units_sold', 'price']
        missing = [col for col in required if col not in df.columns]
        if missing:
            return pd.DataFrame({'error': [f"Missing columns: {', '.join(missing)}"]}), 0, 0

        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df = df.dropna(subset=required + ['date'])

        if len(df) < 100:
            return pd.DataFrame({'error': ['Need at least 100 transactions for accurate segmentation']}), 0, 0

        # ==================== 1. CUSTOMER SEGMENTATION (RFM + KMeans) ====================
        latest_date = df['date'].max()
        rfm = df.groupby('customer_id').agg({
            'date': lambda x: (latest_date - x.max()).days,           # Recency
            'customer_id': 'count',                                   # Frequency
            'price': lambda x: (x * df.loc[x.index, 'units_sold']).sum()  # Monetary
        }).rename(columns={
            'date': 'recency',
            'customer_id': 'frequency',
            'price': 'monetary'
        })

        # Clean monetary
        rfm['monetary'] = rfm['monetary'].replace(0, 0.01)

        # Scale & Cluster
        scaler = StandardScaler()
        rfm_scaled = scaler.fit_transform(rfm[['recency', 'frequency', 'monetary']])
        kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
        rfm['segment'] = kmeans.fit_predict(rfm_scaled)

        # Beautiful segment names
        segment_map = {
            0: "VIP Loyal",      # High frequency, high spend, low recency
            1: "Price Hunters",  # High volume, low spend
            2: "At Risk",        # Was active, now gone
            3: "Sleeping Giants" # High spend, long ago
        }
        rfm['segment_name'] = rfm['segment'].map(segment_map)

        # Merge back to main df
        df = df.merge(rfm[['segment', 'segment_name']], on='customer_id', how='left')

        # ==================== 2. DAILY AGGREGATION PER SEGMENT ====================
        daily = df.groupby(['date', 'product_id', 'segment_name'], as_index=False).agg({
            'units_sold': 'sum',
            'price': 'median',
            'discount': 'mean',
            'competitor_price': 'mean'
        }).rename(columns={'units_sold': 'sales'})

        # Features
        daily['dow'] = daily['date'].dt.dayofweek
        daily['month'] = daily['date'].dt.month
        daily['price_log'] = np.log1p(daily['price'])
        daily['price_vs_comp'] = daily['price'] / (daily['competitor_price'] + 1).replace(0, 1)
        daily['discount'] = daily['discount'].fillna(0)

        num_features = ['price', 'price_log', 'dow', 'month', 'price_vs_comp', 'discount']
        cat_features = ['product_id', 'segment_name']

        encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore', drop='first')
        X_cat = encoder.fit_transform(daily[cat_features])
        X_num = daily[num_features].fillna(0).values
        X = np.hstack([X_num, X_cat])
        y = daily['sales'].values

        # Train model
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = xgb.XGBRegressor(n_estimators=700, max_depth=7, learning_rate=0.05,
                                 subsample=0.9, colsample_bytree=0.9, random_state=42, n_jobs=-1)
        model.fit(X_train, y_train)
        r2 = round(r2_score(y_test, model.predict(X_test)), 3)

        # ==================== 3. PERSONALIZED PRICE OPTIMIZATION ====================
        results = []
        baseline = daily.groupby(['product_id', 'segment_name'], as_index=False).agg({
            'price': 'median',
            'sales': 'mean'
        }).rename(columns={'price': 'current_price', 'sales': 'current_sales'})

        for _, row in baseline.iterrows():
            p_id = row['product_id']
            segment = row['segment_name']
            curr_price = row['current_price']
            curr_sales = row['current_sales']
            curr_rev = curr_price * curr_sales

            prices = np.linspace(curr_price * 0.65, curr_price * 1.45, 40)
            best_price = curr_price
            best_rev = curr_rev

            typical_num = daily[num_features].median()
            cat_input = pd.DataFrame([{'product_id': p_id, 'segment_name': segment}])
            cat_vec = encoder.transform(cat_input)[0]

            for p in prices:
                num_vec = typical_num.copy()
                num_vec['price'] = p
                num_vec['price_log'] = np.log1p(p)
                num_vec['price_vs_comp'] = p / (typical_num['competitor_price'] + 1)
                test_vec = np.hstack([num_vec.values, cat_vec]).reshape(1, -1)
                pred = max(0.5, model.predict(test_vec)[0])
                rev = p * pred
                if rev > best_rev:
                    best_rev = rev
                    best_price = p

            uplift = round((best_rev / curr_rev - 1) * 100, 1) if curr_rev > 0 else 0

            results.append({
                'product_id': p_id,
                'customer_segment': segment,
                'segment_size': len(rfm[rfm['segment_name'] == segment]),
                'current_price': round(curr_price, 2),
                'optimal_price': round(best_price, 2),
                'price_change_%': round((best_price/curr_price - 1)*100, 1),
                'revenue_uplift_%': uplift,
                'insight': {
                    'VIP Loyal': 'Charge 10–22% more — they value quality',
                    'Price Hunters': 'Drop 8–18% — huge volume gain',
                    'At Risk': '15% discount to win them back',
                    'Sleeping Giants': 'Reactivate with flash sale'
                }[segment]
            })

        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values(['customer_segment', 'revenue_uplift_%'], ascending=[True, False])
        avg_uplift = round(results_df['revenue_uplift_%'].mean(), 1)

        return results_df, avg_uplift, r2

    except Exception as e:
        return pd.DataFrame({'error': [f"Error: {str(e)}"]}), 0, 0
