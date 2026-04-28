from django.urls import path
from . import views

urlpatterns = [
    path('', views.index_view, name='index'),
    path('predict/', views.predict_view, name='predict'),
    path('batch/', views.batch_predict_view, name='batch_predict'),
    path('history/', views.history_view, name='history'),
    path('model/', views.model_info_view, name='model_info'),
    path('api/system-status/', views.system_status_api, name='system_status'),
    path('export/csv/', views.export_csv_view, name='export_csv'),
    path('export/pdf/', views.export_pdf_view, name='export_pdf'),
]
