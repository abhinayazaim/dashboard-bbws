from django.db import models
import uuid


class ModelVersion(models.Model):
    version_name = models.CharField(max_length=100, unique=True)
    training_date = models.DateTimeField(auto_now_add=True)
    rmse = models.FloatField(null=True, blank=True)
    mae = models.FloatField(null=True, blank=True)
    r2_score = models.FloatField(null=True, blank=True)
    look_back = models.IntegerField(default=90)
    threshold = models.FloatField(default=88.3794)
    is_active = models.BooleanField(default=False)

    def __str__(self):
        return f"{self.version_name} {'(Active)' if self.is_active else ''}"


class BatchSession(models.Model):
    session_id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    upload_date = models.DateTimeField(auto_now_add=True)
    file_name = models.CharField(max_length=255)
    total_rows = models.IntegerField(default=0)
    danger_count = models.IntegerField(default=0)
    normal_count = models.IntegerField(default=0)

    def __str__(self):
        return f"Batch {self.file_name} ({self.upload_date.strftime('%Y-%m-%d %H:%M')})"


class PredictionRecord(models.Model):
    """Stores each prediction with feature values matching the trained model columns."""
    # Timestamp of the observation
    waktu = models.DateTimeField(null=True, blank=True)

    # Input features (names match training_metadata.json feature_cols)
    curah_hujan_mm = models.FloatField(default=0.0)
    cuaca_kode = models.FloatField(default=0.0)
    smd_kanan_q_ls = models.FloatField(default=0.0)
    smd_kiri_q_ls = models.FloatField(default=0.0)
    tma_lag1 = models.FloatField(default=0.0)
    tma_lag2 = models.FloatField(default=0.0)
    tma_lag3 = models.FloatField(default=0.0)
    delta_tma = models.FloatField(default=0.0)
    tma_rolling_mean_3 = models.FloatField(default=0.0)
    jam_kode = models.FloatField(default=0.0)

    # Results
    tma_predicted = models.FloatField(default=0.0)
    status = models.CharField(max_length=20, default='Pending')  # 'Bahaya' / 'Normal'
    threshold_used = models.FloatField(default=88.3794)
    source = models.CharField(max_length=50, default='Manual')  # 'Manual' / 'Batch'
    batch_session = models.ForeignKey(
        BatchSession, on_delete=models.CASCADE,
        null=True, blank=True, related_name='predictions'
    )
    notes = models.TextField(blank=True, default='')
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-created_at']

    def __str__(self):
        return f"Pred TMA: {self.tma_predicted:.3f} - {self.status}"
