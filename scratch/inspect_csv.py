import pandas as pd
import os

csv_path = 'Bajulmati_Dataset_2018_2026_Imputed.csv'
if os.path.exists(csv_path):
    df = pd.read_csv(csv_path)
    print("CSV columns:", list(df.columns))
    print("First 3 rows:")
    print(df.head(3).to_string())
    print("Is delta_tma in CSV?", 'delta_tma' in df.columns)
else:
    print(f"CSV path {csv_path} not found.")
