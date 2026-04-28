import os
import django

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'config.settings')
django.setup()

from dashboard.ml_engine import MLEngine

engine = MLEngine()
print("Model loaded:", engine.is_loaded)

feature_dict = {
    'curah_hujan_mm': 1.0,
    'cuaca_kode': 1,
    'smd_kanan_q_ls': 2.0,
    'smd_kiri_q_ls': 3.0,
    'jam_kode': 12,
    'tma_lag1': 86.0,
    'tma_lag2': 86.0,
    'tma_lag3': 86.0,
    'delta_tma': 0.0,
    'tma_rolling_mean_3': 86.0
}

try:
    print(engine.predict_single(feature_dict))
except Exception as e:
    import traceback
    traceback.print_exc()
