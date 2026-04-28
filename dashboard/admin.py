from django.contrib import admin
from .models import PredictionRecord, BatchSession, ModelVersion


@admin.register(PredictionRecord)
class PredictionRecordAdmin(admin.ModelAdmin):
    list_display = ('created_at', 'tma_predicted', 'status', 'source', 'curah_hujan_mm')
    list_filter = ('status', 'source', 'created_at')
    search_fields = ('notes',)


@admin.register(BatchSession)
class BatchSessionAdmin(admin.ModelAdmin):
    list_display = ('session_id', 'file_name', 'upload_date', 'total_rows', 'danger_count', 'normal_count')


@admin.register(ModelVersion)
class ModelVersionAdmin(admin.ModelAdmin):
    list_display = ('version_name', 'training_date', 'rmse', 'is_active')
