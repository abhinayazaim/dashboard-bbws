from django.urls import path
from . import views

urlpatterns = [
    path('', views.index_view, name='index'),
    path('input-data/', views.predict_view, name='input_data'),
    path('input-data/delete/<int:id>/', views.delete_observation_view, name='delete_observation'),
    path('prediksi/', views.batch_predict_view, name='prediksi'),
    path('history/', views.history_view, name='history'),
    path('model/', views.model_info_view, name='model_info'),
    path('api/system-status/', views.system_status_api, name='system_status'),
    path('api/predict-esok/', views.predict_esok_api, name='predict_esok_api'),
    path('api/retrain-model/', views.retrain_model_api, name='retrain_model_api'),
    path('export/csv/', views.export_csv_view, name='export_csv'),
    path('export/pdf/', views.export_pdf_view, name='export_pdf'),
    path('historical-data/', views.historical_data_view, name='historical_data'),
    path('reset/', views.reset_data_view, name='reset_data'),
]
