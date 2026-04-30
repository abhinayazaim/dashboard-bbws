import json
import pandas as pd
from datetime import datetime, timedelta

from django.shortcuts import render, redirect, get_object_or_404
from django.http import JsonResponse, HttpResponse
from django.contrib import messages
from django.core.paginator import Paginator
from django.utils import timezone
from django.db.models import Q

from .models import PredictionRecord, BatchSession
from .forms import ManualPredictionForm, BatchUploadForm
from .ml_engine import MLEngine
from .export_utils import export_history_to_csv, export_history_to_pdf


def _compute_lag_features(last_records):
    """
    Auto-compute lag1/lag2/lag3, delta_tma, and rolling_mean from the last 3 predictions.
    Returns dict of computed features.
    """
    tma_values = []
    for r in last_records:
        tma_values.append(r.tma_predicted)

    # Pad if not enough history
    while len(tma_values) < 3:
        tma_values.insert(0, tma_values[0] if tma_values else 86.0)

    lag1 = tma_values[-1]  # most recent
    lag2 = tma_values[-2]
    lag3 = tma_values[-3]
    delta_tma = lag1 - lag2
    rolling_mean = sum(tma_values[-3:]) / 3.0

    return {
        'tma_lag1': lag1,
        'tma_lag2': lag2,
        'tma_lag3': lag3,
        'delta_tma': delta_tma,
        'tma_rolling_mean_3': rolling_mean,
    }


def index_view(request):
    """Dashboard home page."""
    engine = MLEngine()

    # Stats
    total_predictions = PredictionRecord.objects.count()
    today = timezone.now().date()
    today_predictions = PredictionRecord.objects.filter(created_at__date=today).count()
    danger_count = PredictionRecord.objects.filter(status='Bahaya').count()

    # Last 50 for chart
    # Order by created_at then id to keep batch rows in sequence even if created at same second
    last_50 = list(PredictionRecord.objects.order_by('-created_at', '-id')[:50])
    last_50.reverse()

    chart_labels = []
    chart_data = []
    for p in last_50:
        # Priority: use 'waktu' (actual observation time) if available, fallback to 'created_at'
        label_time = p.waktu if p.waktu else p.created_at
        chart_labels.append(label_time.strftime('%d %b %H:%M'))
        chart_data.append(round(p.tma_predicted, 3))

    # Last 4 for log table
    last_4 = PredictionRecord.objects.order_by('-created_at')[:4]

    # Model info
    threshold = engine.get_threshold()
    metrics = engine.get_model_metrics()

    # Manual prediction form
    form = ManualPredictionForm()
    result = None
    result_status = None

    # Load static test results for the model performance chart
    static_chart_labels = []
    static_chart_actuals = []
    static_chart_predicteds = []
    import os
    from django.conf import settings
    static_results_path = os.path.join(settings.BASE_DIR, 'models', 'static_test_results.json')
    if os.path.exists(static_results_path):
        with open(static_results_path, 'r') as f:
            static_data = json.load(f)
            # Send the entire dataset to the frontend so the slider can navigate 2018-2026
            static_chart_labels = static_data.get('labels', [])
            static_chart_actuals = static_data.get('actuals', [])
            static_chart_predicteds = static_data.get('predicteds', [])

    if request.method == 'POST':
        form = ManualPredictionForm(request.POST)
        if form.is_valid():
            cd = form.cleaned_data

            # Compute lag features from history
            last_records = list(PredictionRecord.objects.order_by('-created_at')[:3])
            lag_features = _compute_lag_features(last_records)

            feature_dict = {
                'curah_hujan_mm': cd['curah_hujan_mm'],
                'cuaca_kode': float(cd['cuaca_kode']),
                'smd_kanan_q_ls': cd['smd_kanan_q_ls'],
                'smd_kiri_q_ls': cd['smd_kiri_q_ls'],
                'jam_kode': float(cd['jam_kode']),
                **lag_features,
            }

            tma_pred, pred_status, th = engine.predict_single(feature_dict)

            # Save to DB
            PredictionRecord.objects.create(
                waktu=cd['waktu'],
                curah_hujan_mm=cd['curah_hujan_mm'],
                cuaca_kode=float(cd['cuaca_kode']),
                smd_kanan_q_ls=cd['smd_kanan_q_ls'],
                smd_kiri_q_ls=cd['smd_kiri_q_ls'],
                jam_kode=float(cd['jam_kode']),
                tma_lag1=lag_features['tma_lag1'],
                tma_lag2=lag_features['tma_lag2'],
                tma_lag3=lag_features['tma_lag3'],
                delta_tma=lag_features['delta_tma'],
                tma_rolling_mean_3=lag_features['tma_rolling_mean_3'],
                tma_predicted=tma_pred,
                status=pred_status,
                threshold_used=th,
                source='Manual',
            )

            result = round(tma_pred, 3)
            result_status = pred_status
            messages.success(request, 'Prediksi berhasil dilakukan.')

    context = {
        'total_predictions': total_predictions,
        'today_predictions': today_predictions,
        'danger_count': danger_count,
        'chart_labels': json.dumps(chart_labels),
        'chart_data': json.dumps(chart_data),
        'threshold': threshold,
        'last_4': last_4,
        'metrics': metrics,
        'is_loaded': engine.is_loaded,
        'form': form,
        'result': result,
        'result_status': result_status,
        'static_chart_labels': json.dumps(static_chart_labels),
        'static_chart_actuals': json.dumps(static_chart_actuals),
        'static_chart_predicteds': json.dumps(static_chart_predicteds),
    }
    return render(request, 'dashboard/index.html', context)


def predict_view(request):
    """Dedicated manual prediction page."""
    engine = MLEngine()
    form = ManualPredictionForm()
    result = None
    result_status = None

    if request.method == 'POST':
        form = ManualPredictionForm(request.POST)
        if form.is_valid():
            cd = form.cleaned_data
            last_records = list(PredictionRecord.objects.order_by('-created_at')[:3])
            lag_features = _compute_lag_features(last_records)

            feature_dict = {
                'curah_hujan_mm': cd['curah_hujan_mm'],
                'cuaca_kode': float(cd['cuaca_kode']),
                'smd_kanan_q_ls': cd['smd_kanan_q_ls'],
                'smd_kiri_q_ls': cd['smd_kiri_q_ls'],
                'jam_kode': float(cd['jam_kode']),
                **lag_features,
            }

            tma_pred, pred_status, th = engine.predict_single(feature_dict)

            PredictionRecord.objects.create(
                waktu=cd['waktu'],
                curah_hujan_mm=cd['curah_hujan_mm'],
                cuaca_kode=float(cd['cuaca_kode']),
                smd_kanan_q_ls=cd['smd_kanan_q_ls'],
                smd_kiri_q_ls=cd['smd_kiri_q_ls'],
                jam_kode=float(cd['jam_kode']),
                tma_lag1=lag_features['tma_lag1'],
                tma_lag2=lag_features['tma_lag2'],
                tma_lag3=lag_features['tma_lag3'],
                delta_tma=lag_features['delta_tma'],
                tma_rolling_mean_3=lag_features['tma_rolling_mean_3'],
                tma_predicted=tma_pred,
                status=pred_status,
                threshold_used=th,
                source='Manual',
            )

            result = round(tma_pred, 3)
            result_status = pred_status
            messages.success(request, 'Prediksi berhasil dilakukan.')

    context = {
        'form': form,
        'result': result,
        'result_status': result_status,
        'threshold': engine.get_threshold(),
    }
    return render(request, 'dashboard/predict.html', context)


def batch_predict_view(request):
    """Batch upload and prediction page."""
    if request.method == 'POST':
        form = BatchUploadForm(request.POST, request.FILES)
        if form.is_valid():
            uploaded_file = request.FILES['csv_file']
            file_name = uploaded_file.name

            try:
                # Exclusively read CSV
                try:
                    df = pd.read_csv(uploaded_file, encoding='utf-8')
                except UnicodeDecodeError:
                    uploaded_file.seek(0)
                    df = pd.read_csv(uploaded_file, encoding='latin1')

                engine = MLEngine()

                # Run batch prediction
                # predict_batch will automatically compute V2 features (log, avg) if missing
                result_df = engine.predict_batch(df)

                # Create batch session
                session = BatchSession.objects.create(
                    file_name=file_name,
                    total_rows=len(result_df),
                )

                # Bulk create prediction records
                records = []
                danger_count = 0
                normal_count = 0

                # Identify time column from original df BEFORE preprocessing
                time_col = None
                for col in ['datetime', 'waktu', 'date', 'tanggal', 'time', 'timestamp']:
                    # Case-insensitive column match
                    match = next((c for c in df.columns if c.strip().lower() == col), None)
                    if match:
                        time_col = match
                        break

                for _, row in result_df.iterrows():
                    pred_status = row.get('status', 'Pending')
                    if pred_status == 'Bahaya':
                        danger_count += 1
                    elif pred_status == 'Normal':
                        normal_count += 1

                    pred_val = row.get('tma_predicted', 0.0)
                    if pd.isna(pred_val):
                        pred_val = 0.0

                    # Handle observation time from original uploaded data
                    obs_time = None  # None = unknown, will display as "-" in table
                    if time_col and time_col in result_df.columns:
                        try:
                            raw_t = row.get(time_col)
                            if raw_t is not None and not (isinstance(raw_t, float) and pd.isna(raw_t)):
                                parsed = pd.to_datetime(raw_t, errors='coerce')
                                if not pd.isna(parsed):
                                    obs_time = parsed.to_pydatetime()
                        except Exception:
                            pass

                    records.append(PredictionRecord(
                        waktu=obs_time,
                        curah_hujan_mm=row.get('curah_hujan_mm', 0),
                        cuaca_kode=row.get('cuaca_kode', 0),
                        smd_kanan_q_ls=row.get('smd_kanan_q_ls', 0),
                        smd_kiri_q_ls=row.get('smd_kiri_q_ls', 0),
                        tma_lag1=row.get('tma_lag1', 0),
                        tma_lag2=row.get('tma_lag2', 0),
                        tma_lag3=row.get('tma_lag3', 0),
                        delta_tma=row.get('delta_tma', 0),
                        tma_rolling_mean_3=row.get('tma_rolling_mean_3', 0),
                        jam_kode=row.get('jam_kode', 0),
                        tma_predicted=pred_val,
                        status=pred_status,
                        threshold_used=engine.get_threshold(),
                        source='Batch',
                        batch_session=session,
                    ))

                PredictionRecord.objects.bulk_create(records)

                session.danger_count = danger_count
                session.normal_count = normal_count
                session.save()

                messages.success(
                    request,
                    f'Batch berhasil diproses: {normal_count} Normal, {danger_count} Bahaya.'
                )
                return redirect('history')

            except Exception as e:
                messages.error(request, f"Error: {str(e)}")
    else:
        form = BatchUploadForm()

    return render(request, 'dashboard/batch.html', {'form': form})


def history_view(request):
    """Prediction history page with filters."""
    query = PredictionRecord.objects.all()

    # Date range filter
    date_from = request.GET.get('date_from', '')
    date_to = request.GET.get('date_to', '')
    status_filter = request.GET.get('status', '')
    search_query = request.GET.get('q', '')
    
    if date_from:
        try:
            from_date = datetime.strptime(date_from, '%Y-%m-%d').date()
            query = query.filter(created_at__date__gte=from_date)
        except ValueError:
            pass
    if date_to:
        try:
            to_date = datetime.strptime(date_to, '%Y-%m-%d').date()
            query = query.filter(created_at__date__lte=to_date)
        except ValueError:
            pass
            
    if status_filter and status_filter != 'all':
        query = query.filter(status=status_filter)
        
    if search_query:
        try:
            obs_date = datetime.strptime(search_query, '%Y-%m-%d').date()
            query = query.filter(waktu__date=obs_date)
        except ValueError:
            # If not a valid date, fall back to text search on source/status
            query = query.filter(
                Q(source__icontains=search_query) |
                Q(status__icontains=search_query)
            )

    total_count = query.count()
    paginator = Paginator(query, 10)
    page_number = request.GET.get('page')
    page_obj = paginator.get_page(page_number)

    context = {
        'page_obj': page_obj,
        'total_count': total_count,
        'date_from': date_from,
        'date_to': date_to,
        'status_filter': status_filter,
        'search_query': search_query,
    }
    return render(request, 'dashboard/history.html', context)


def model_info_view(request):
    """Model information and transparency page."""
    engine = MLEngine()

    # Features and weights from engine (now following metadata/validated image)
    features = engine.feature_cols
    weights = engine.attention_weights if engine.attention_weights is not None else []

    # Feature display names for V2 model
    feature_display_names = {
        'curah_hujan_log': 'Curah Hujan (Log)',
        'cuaca_kode': 'Kode Cuaca',
        'smd_avg': 'Debit Rata-rata (SMD)',
        'delta_tma_lag1': 'Delta TMA (Lag 1)',
        'jam_kode': 'Jam Kode',
    }

    feature_data = []
    if len(weights) > 0:
        max_w = max(weights)
        for i, feat in enumerate(features):
            w = weights[i] if i < len(weights) else 0
            feature_data.append({
                'name': feature_display_names.get(feat, feat),
                'raw_name': feat,
                'weight': round(w, 4),
                'bar_width': round((w / max_w) * 100, 1) if max_w > 0 else 0,
                'is_lag': 'lag' in feat.lower(),
            })

    metrics = engine.get_model_metrics()
    model_info = engine.get_model_info()


    context = {
        'feature_data': feature_data,
        'metrics': metrics,
        'model_info': model_info,
        'is_loaded': engine.is_loaded,
    }

    return render(request, 'dashboard/model_info.html', context)


def system_status_api(request):
    """API endpoint for system status badge."""
    engine = MLEngine()
    return JsonResponse({
        'status': 'operational' if engine.is_loaded else 'offline',
        'model_loaded': engine.is_loaded,
        'threshold': engine.get_threshold(),
    })


def export_csv_view(request):
    """Export filtered history to CSV."""
    return export_history_to_csv(request)


def export_pdf_view(request):
    """Export filtered history to PDF."""
    return export_history_to_pdf(request)

def historical_data_view(request):
    """View to query and display historical data from the original dataset."""
    target_date = request.GET.get('target_date', '')
    data = []
    
    if target_date:
        engine = MLEngine()
        data = engine.get_historical_data(target_date)
        
    context = {
        'target_date': target_date,
        'historical_data': data,
    }
    return render(request, 'dashboard/historical_data.html', context)

def reset_data_view(request):
    """Deletes all prediction records and batch sessions."""
    if request.method == 'POST':
        # Delete all records
        PredictionRecord.objects.all().delete()
        BatchSession.objects.all().delete()
        
        messages.success(request, "Seluruh riwayat data telah berhasil direset ke nol.")
        return redirect('index')
    return redirect('index')
