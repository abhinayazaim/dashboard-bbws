import os
import django

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "dashboard_project.settings")
django.setup()

import pandas as pd
from dashboard.models import DataBendungan, LogPrediksi

print(f"DataBendungan count: {DataBendungan.objects.count()}")
print(f"LogPrediksi count: {LogPrediksi.objects.count()}")

dataset_path = 'Bajulmati_Dataset_2018_2026_Imputed.csv'
df_base = pd.read_csv(dataset_path) if os.path.exists(dataset_path) else pd.DataFrame()
print(f"Base CSV count: {len(df_base)}")
