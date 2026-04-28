import os
import json
import pandas as pd
import django

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'config.settings')
django.setup()

from dashboard.ml_engine import MLEngine

def generate():
    csv_path = 'Bajulmati_Dataset_2018_2026_Imputed.csv'
    if not os.path.exists(csv_path):
        print("CSV not found.")
        return

    # Read the last 500 rows to ensure we have enough for look_back
    df = pd.read_csv(csv_path)
    df = df.tail(300).reset_index(drop=True)
    
    engine = MLEngine()
    
    # We need to simulate a continuous batch prediction.
    # The batch predict needs all feature cols.
    # Let's ensure the df has the right cols.
    missing = [c for c in engine.feature_cols if c not in df.columns]
    if missing:
        print(f"Missing columns in CSV: {missing}")
        return

    print("Running batch prediction on subset...")
    # predict_batch adds 'tma_predicted'
    # Wait, the engine's predict_batch uses seed_history if we don't pass continuous data.
    # Actually, df has 300 rows. predict_batch will use its own rows for look_back (90).
    # So the first 89 rows will have NaN predictions. 
    # We can take the last 150 rows.
    result_df = engine.predict_batch(df)
    
    # Filter out NaNs
    valid_df = result_df.dropna(subset=['tma_predicted']).tail(150)
    
    if valid_df.empty:
        print("No valid predictions generated.")
        return
        
    labels = []
    actuals = []
    predicteds = []
    
    for _, row in valid_df.iterrows():
        # Prefer datetime column if available, else use index
        dt = row.get('waktu', str(_))
        if isinstance(dt, str) and ' ' in dt:
            dt = dt.split(' ')[0] # just the date or time
            
        labels.append(dt)
        actuals.append(round(row['tma_m'], 3) if 'tma_m' in row else 0)
        predicteds.append(round(row['tma_predicted'], 3))
        
    output_data = {
        'labels': labels,
        'actuals': actuals,
        'predicteds': predicteds
    }
    
    out_path = os.path.join('models', 'static_test_results.json')
    with open(out_path, 'w') as f:
        json.dump(output_data, f)
        
    print(f"Generated {len(labels)} data points and saved to {out_path}.")

if __name__ == '__main__':
    generate()
