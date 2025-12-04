# pricing_engine.py — FINAL ULTRA-ADVANCED (Only 1 dataset, All Features, Real R²)
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import warnings
warnings.filterwarnings('ignore')

def run_pricing_engine(demand_df):
    try:
        df = demand_df.copy()
        df.columns = [c.strip().lower().replace(' ', '_') for c in df.columns]

        # Auto-map all possible column names
        col_map = {
            'date': 'date', 'store_id': 'store_id', 'product_id': 'product_id',
            'units_sold': 'units_sold', 'units_ordered': 'units_sold', 'inventory': 'inventory',
            'price': 'price', 'demand_f_price': 'price', 'selling_price': 'price',
            'discount': 'discount', 'competitor_price': 'competitor_price', '/pr': 'competitor_price',
            'category': 'category', 'region': 'region', 'weather': 'weather',
            'holiday': 'holiday', 'seasonality': 'seasonality'
        }
        df.rename(columns={v: k for k, v in col_map.items() if v in df.columns}, inplace=True)

        required = ['date', 'product_id', 'units_sold', 'price']
        if not all(col in df.columns for col in required):
            return pd.DataFrame({'error': ['Missing required columns: date, product_id, units_sold, price']}), 0, 0

        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df = df.dropna(subset=required)

        if len(df) < 20:
            return pd.DataFrame({'error': ['Need at least 20 sales records']}), 0, 0

        # Add store_id if missing
        if 'store_id' not in df.columns:
            df['store_id'] = 'Main_Store'

        # Aggregate to daily
        agg_dict = {
            'units_sold': 'sum',
            'price': 'median',
            'inventory': 'mean',
            'discount': 'mean',
            'competitor_price': 'mean'
        }
        for col in ['inventory', 'discount', 'competitor_price']:
            if col not in df.columns:
                df[col] = 0

        daily = df.groupby(['date', 'store_id', 'product_id'], as_index=False).agg(agg_dict)
        daily.rename(columns={'units_sold': 'sales'}, inplace=True)

        # Rich Features
        daily['dow'] = daily['date'].dt.dayofweek
        daily['month'] = daily['date'].dt.month
        daily['is_weekend'] = (daily['dow'] >= 5).astype(int)
        daily['is_holiday'] = daily['date'].isin(df[df['holiday'] == 1]['date'].unique()).astype(int) if 'holiday' in df.columns else 0
        daily['price_log'] = np.log1p(daily['price'])
        daily['price_ratio_vs_comp'] = daily['price'] / (daily['competitor_price'] + 1)
        daily['discount_pct'] = daily['discount']

        # Merge categorical modes per product
        cat_cols = ['category', 'region', 'weather', 'seasonality']
        for col in cat_cols:
            if col in df.columns:
                mode_map = df.groupby('product_id')[col].agg(lambda x: x.mode().iloc[0] if not x.empty else 'Unknown').to_dict()
                daily[col] = daily['product_id'].map(mode_map).fillna('Unknown')
            else:
                daily[col] = 'Unknown'

        # Final features
        num_features = ['price', 'price_log', 'dow', 'month', 'is_weekend', 'is_holiday',
                        'inventory', 'discount_pct', 'price_ratio_vs_comp']
        cat_features = ['store_id', 'product_id', 'category', 'region', 'weather', 'seasonality']

        encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore', drop='first')
        X_cat = encoder.fit_transform(daily[cat_features])
        X_num = daily[num_features].fillna(0).values
        X = np.hstack([X_num, X_cat])
        y = daily['sales'].values

        # Train with real validation
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = xgb.XGBRegressor(
            n_estimators=600,
            max_depth=7,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=42,
            n_jobs=-1
        )
        model.fit(X_train, y_train)
        r2 = round(r2_score(y_test, model.predict(X_test)), 3)

        # Baseline (last 7 days)
        baseline = daily.sort_values('date').groupby(['product_id', 'store_id']).tail(7)
        baseline = baseline.groupby(['product_id', 'store_id'], as_index=False).agg({
            'price': 'median', 'sales': 'mean'
        }).rename(columns={'price': 'current_price', 'sales': 'avg_sales'})

        results = []
        for _, row in baseline.iterrows():
            curr_price = row['current_price']
            curr_sales = row['avg_sales']
            curr_rev = curr_price * curr_sales

            prices = np.linspace(curr_price * 0.65, curr_price * 1.45, 40)
            best_rev = curr_rev
            best_price = curr_price
            best_pred = curr_sales

            # Build typical context
            typical = daily[num_features].median()
            typical_cat = daily[cat_features].mode().iloc[0]
            cat_vec = encoder.transform(typical_cat.to_frame().T)[0]

            for p in prices:
                num_vec = typical.copy()
                num_vec['price'] = p
                num_vec['price_log'] = np.log1p(p)
                num_vec['price_ratio_vs_comp'] = p / (typical['competitor_price'] + 1)
                test_vec = np.hstack([num_vec.values, cat_vec]).reshape(1, -1)
                pred = max(0.5, model.predict(test_vec)[0])
                rev = p * pred
                if rev > best_rev:
                    best_rev = rev
                    best_price = p
                    best_pred = pred

            uplift = (best_rev / curr_rev - 1) * 100 if curr_rev > 0 else 0
            results.append({
                'product_id': row['product_id'],
                'store_id': row['store_id'],
                'current_price': round(curr_price, 2),
                'optimal_price': round(best_price, 2),
                'price_change_%': round((best_price/curr_price - 1)*100, 1),
                'current_daily_revenue': round(curr_rev, 2),
                'optimal_daily_revenue': round(best_rev, 2),
                'revenue_uplift_%': round(uplift, 1),
                'predicted_daily_units': round(best_pred, 1)
            })

        results_df = pd.DataFrame(results).sort_values('revenue_uplift_%', ascending=False)
        avg_uplift = results_df['revenue_uplift_%'].mean()

        return results_df, round(avg_uplift, 1), r2

    except Exception as e:
        return pd.DataFrame({'error': [f"Error: {str(e)}"]}), 0, 0
