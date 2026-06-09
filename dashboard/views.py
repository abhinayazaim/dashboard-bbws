import json
import pandas as pd
from datetime import datetime, timedelta

from django.shortcuts import render, redirect, get_object_or_404
from django.http import JsonResponse, HttpResponse
from django.contrib import messages
from django.core.paginator import Paginator
from django.utils import timezone
from django.db.models import Q

from .models import LogPrediksi, BatchSession, DataBendungan
from .forms import DataBendunganForm, BatchUploadForm, ManualPredictionForm
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
    total_predictions = LogPrediksi.objects.count()
    today = timezone.now().date()
    today_predictions = LogPrediksi.objects.filter(created_at__date=today).count()
    
    aman_count = LogPrediksi.objects.filter(status='Aman').count()
    waspada_count = LogPrediksi.objects.filter(status='Waspada').count()
    siaga_count = LogPrediksi.objects.filter(status='Siaga').count()
    awas_count = LogPrediksi.objects.filter(status='Awas').count()

    # Last 50 for chart
    last_50 = list(LogPrediksi.objects.order_by('-created_at', '-id')[:50])
    last_50.reverse()

    chart_labels = []
    chart_data = []
    for p in last_50:
        label_time = p.waktu if p.waktu else p.created_at
        chart_labels.append(label_time.strftime('%d %b %H:%M'))
        chart_data.append(round(p.tma_predicted, 3))

    last_4 = LogPrediksi.objects.order_by('-created_at')[:4]
    threshold = engine.get_threshold()
    metrics = engine.get_model_metrics()

    result = None
    result_status = None

    static_chart_labels = []
    static_chart_actuals = []
    static_chart_predicteds = []
    import os
    from django.conf import settings
    static_results_path = os.path.join(settings.BASE_DIR, 'models', 'static_test_results.json')
    if os.path.exists(static_results_path):
        with open(static_results_path, 'r') as f:
            static_data = json.load(f)
            static_chart_labels = static_data.get('labels', [])
            static_chart_actuals = static_data.get('actuals', [])
            static_chart_predicteds = static_data.get('predicteds', [])

    context = {
        'total_predictions': total_predictions,
        'today_predictions': today_predictions,
        'aman_count': aman_count,
        'waspada_count': waspada_count,
        'siaga_count': siaga_count,
        'awas_count': awas_count,
        'chart_labels': json.dumps(chart_labels),
        'chart_data': json.dumps(chart_data),
        'threshold': threshold,
        'last_4': last_4,
        'metrics': metrics,
        'is_loaded': engine.is_loaded,
        'result': result,
        'result_status': result_status,
        'static_chart_labels': json.dumps(static_chart_labels),
        'static_chart_actuals': json.dumps(static_chart_actuals),
        'static_chart_predicteds': json.dumps(static_chart_predicteds),
    }
    return render(request, 'dashboard/index.html', context)


def predict_esok_api(request):
    if request.method == 'POST':
        from .models import DataBendungan
        latest = DataBendungan.objects.order_by('-tanggal').first()
        if not latest:
            return JsonResponse({'error': 'Data harian kosong. Silakan input data terlebih dahulu.'}, status=400)
            
        engine = MLEngine()
        last_records = list(LogPrediksi.objects.order_by('-created_at')[:3])
        lag_features = _compute_lag_features(last_records)
        
        feature_dict = {
            'curah_hujan_mm': latest.curah_hujan_mm,
            'cuaca_kode': latest.cuaca_kode,
            'smd_kanan_q_ls': latest.smd_kanan_q_ls,
            'smd_kiri_q_ls': latest.smd_kiri_q_ls,
            'jam_kode': latest.jam_kode,
            **lag_features,
        }
        
        tma_pred, pred_status, th = engine.predict_single(feature_dict)
        
        # Simpan ke LogPrediksi
        import datetime
        pred_waktu = datetime.datetime.combine(latest.tanggal + datetime.timedelta(days=1), datetime.time(12, 0))
        
        LogPrediksi.objects.create(
            waktu=pred_waktu,
            curah_hujan_mm=latest.curah_hujan_mm,
            cuaca_kode=latest.cuaca_kode,
            smd_kanan_q_ls=latest.smd_kanan_q_ls,
            smd_kiri_q_ls=latest.smd_kiri_q_ls,
            jam_kode=latest.jam_kode,
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
        return JsonResponse({'tma_predicted': round(tma_pred, 3), 'status': pred_status})
    return JsonResponse({'error': 'Invalid request'}, status=400)


def retrain_model_api(request):
    if request.method == 'POST':
        engine = MLEngine()
        msg = engine.train_candidate_model()
        return JsonResponse({'message': msg})
    return JsonResponse({'error': 'Invalid request'}, status=400)


def predict_view(request):
    """Input Data page (menyimpan record data harian)."""
    form = DataBendunganForm()
    if request.method == 'POST':
        form = DataBendunganForm(request.POST)
        if form.is_valid():
            cd = form.cleaned_data
            DataBendungan.objects.update_or_create(
                tanggal=cd['tanggal'],
                defaults={
                    'tma': cd['tma'],
                    'curah_hujan_mm': cd['curah_hujan_mm'],
                    'cuaca_kode': cd['cuaca_kode'],
                    'smd_kanan_q_ls': cd['smd_kanan_q_ls'],
                    'smd_kiri_q_ls': cd['smd_kiri_q_ls'],
                    'jam_kode': cd['jam_kode'],
                }
            )
            messages.success(request, 'Data harian berhasil disimpan ke database.')
            return redirect('input_data')

    # Query all DataBendungan records, ordered by date descending
    records_list = DataBendungan.objects.all().order_by('-tanggal')
    paginator = Paginator(records_list, 10)  # 10 records per page
    page_number = request.GET.get('page')
    page_obj = paginator.get_page(page_number)

    return render(request, 'dashboard/input_data.html', {
        'form': form,
        'page_obj': page_obj,
    })


def delete_observation_view(request, id):
    """Delete a DataBendungan observation record."""
    if request.method == 'POST':
        record = get_object_or_404(DataBendungan, id=id)
        tanggal_str = record.tanggal.strftime('%Y-%m-%d')
        record.delete()
        messages.success(request, f'Data observasi tanggal {tanggal_str} berhasil dihapus.')
    return redirect('input_data')


def batch_predict_view(request):
    """Batch upload and manual prediction page."""
    form_batch = BatchUploadForm()
    form_manual = ManualPredictionForm()
    
    if request.method == 'POST' and 'csv_file' in request.FILES:
        form_batch = BatchUploadForm(request.POST, request.FILES)
        if form_batch.is_valid():
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
                darurat_count = 0
                danger_count = 0
                waspada_count = 0
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
                    if pred_status == 'Awas':
                        darurat_count += 1
                    elif pred_status == 'Siaga':
                        danger_count += 1
                    elif pred_status == 'Waspada':
                        waspada_count += 1
                    elif pred_status == 'Aman':
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

                    records.append(LogPrediksi(
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

                LogPrediksi.objects.bulk_create(records)

                session.danger_count = danger_count + darurat_count # aggregate danger for session
                session.normal_count = normal_count + waspada_count
                session.save()

                messages.success(
                    request,
                    f'Batch berhasil diproses: {normal_count} Aman, {waspada_count} Waspada, {danger_count} Siaga, {darurat_count} Awas.'
                )
                return redirect('history')

            except Exception as e:
                messages.error(request, f"Error: {str(e)}")

    elif request.method == 'POST' and 'curah_hujan_mm' in request.POST:
        form_manual = ManualPredictionForm(request.POST)
        if form_manual.is_valid():
            engine = MLEngine()
            cd = form_manual.cleaned_data
            
            last_records = list(LogPrediksi.objects.order_by('-created_at')[:3])
            lag_features = _compute_lag_features(last_records)
            
            feature_dict = {
                'curah_hujan_mm': cd['curah_hujan_mm'],
                'cuaca_kode': int(cd['cuaca_kode']),
                'smd_kanan_q_ls': cd['smd_kanan_q_ls'],
                'smd_kiri_q_ls': cd['smd_kiri_q_ls'],
                'jam_kode': int(cd['jam_kode']),
                **lag_features,
            }
            
            tma_pred, pred_status, th = engine.predict_single(feature_dict)
            
            # Save to log but without marking as 'Manual' from DB, mark as 'Manual Test'
            LogPrediksi.objects.create(
                waktu=timezone.now(),
                curah_hujan_mm=cd['curah_hujan_mm'],
                cuaca_kode=int(cd['cuaca_kode']),
                smd_kanan_q_ls=cd['smd_kanan_q_ls'],
                smd_kiri_q_ls=cd['smd_kiri_q_ls'],
                jam_kode=int(cd['jam_kode']),
                tma_lag1=lag_features['tma_lag1'],
                tma_lag2=lag_features['tma_lag2'],
                tma_lag3=lag_features['tma_lag3'],
                delta_tma=lag_features['delta_tma'],
                tma_rolling_mean_3=lag_features['tma_rolling_mean_3'],
                tma_predicted=tma_pred,
                status=pred_status,
                threshold_used=th,
                source='Manual Test',
            )
            
            return render(request, 'dashboard/prediksi.html', {
                'form_batch': form_batch,
                'form_manual': form_manual,
                'result': round(tma_pred, 3),
                'result_status': pred_status,
                'threshold': th,
            })

    return render(request, 'dashboard/prediksi.html', {
        'form_batch': form_batch,
        'form_manual': form_manual,
    })


def history_view(request):
    """Prediction history page with filters."""
    query = LogPrediksi.objects.all()

    # Filters
    year_filter = request.GET.get('year', '')
    status_filter = request.GET.get('status', '')
    search_query = request.GET.get('q', '')
    
    if year_filter and year_filter != 'all':
        try:
            year_int = int(year_filter)
            # Filter by observation time (waktu) year
            query = query.filter(waktu__year=year_int)
        except ValueError:
            pass
            
    if status_filter and status_filter != 'all':
        query = query.filter(status=status_filter)
        
    if search_query:
        try:
            obs_date = datetime.strptime(search_query, '%Y-%m-%d')
            obs_date_end = obs_date.replace(hour=23, minute=59, second=59)
            query = query.filter(waktu__gte=obs_date, waktu__lte=obs_date_end)
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
        'year_filter': year_filter,
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

def export_dataset_csv_view(request):
    """Export all prediction logs as a single dataset CSV, dropping duplicates."""
    import pandas as pd
    from django.http import HttpResponse
    from django.contrib import messages
    from django.shortcuts import redirect
    
    # Ambil semua data dari LogPrediksi (mencakup Batch Upload dan Manual Predict)
    # Ini merepresentasikan semua data yang pernah diproses/dimasukkan user.
    log_records = LogPrediksi.objects.all().order_by('waktu', '-created_at')
    
    if not log_records.exists():
        messages.error(request, "Tidak ada data yang dapat diekspor.")
        return redirect('history')
        
    log_data = []
    for r in log_records:
        dt_str = r.waktu.strftime('%Y-%m-%d %H:%M') if r.waktu else ''
        log_data.append({
            'datetime': dt_str,
            'tma_m': r.tma_predicted,
            'curah_hujan_mm': r.curah_hujan_mm,
            'cuaca_kode': r.cuaca_kode,
            'smd_kanan_q_ls': r.smd_kanan_q_ls,
            'smd_kiri_q_ls': r.smd_kiri_q_ls,
            'tma_lag1': r.tma_lag1,
            'tma_lag2': r.tma_lag2,
            'tma_lag3': r.tma_lag3,
            'delta_tma': r.delta_tma,
            'tma_rolling_mean_3': r.tma_rolling_mean_3,
            'jam_kode': r.jam_kode,
            'status': r.status,
            'source': r.source,
            'created_at': r.created_at
        })
        
    df = pd.DataFrame(log_data)
    
    # Hindari duplikasi jika datanya sama.
    subset_cols = ['datetime', 'curah_hujan_mm', 'cuaca_kode', 'smd_kanan_q_ls', 'smd_kiri_q_ls', 'jam_kode', 'tma_m']
    df = df.drop_duplicates(subset=subset_cols, keep='first')
    
    # Konversi ke datetime asli untuk sorting kronologis
    df['datetime_dt'] = pd.to_datetime(df['datetime'], errors='coerce')
    df = df.sort_values(by=['datetime_dt', 'created_at'])
    
    # Clean up temporary columns
    df = df.drop(columns=['datetime_dt', 'created_at'])
    
    response = HttpResponse(content_type='text/csv')
    response['Content-Disposition'] = 'attachment; filename="Bajulmati_Dataset_Gabungan.csv"'
    
    df.to_csv(response, index=False)
    return response

def export_actual_dataset_csv_view(request):
    """Export the actual dataset (Base CSV + DataBendungan manual inputs)."""
    import pandas as pd
    from django.http import HttpResponse
    from django.contrib import messages
    from django.shortcuts import redirect
    
    engine = MLEngine()
    try:
        data = engine.get_historical_data('')
    except Exception as e:
        messages.error(request, f"Gagal mengambil data historis: {e}")
        return redirect('historical_data')

    for item in data:
        item['source'] = 'Base_CSV'
        
    db_records = DataBendungan.objects.all()
    for r in db_records:
        if r.jam_kode == 0 or r.jam_kode == 6:
            time_str = "06:00"
        elif r.jam_kode == 1 or r.jam_kode == 12:
            time_str = "12:00"
        elif r.jam_kode == 2 or r.jam_kode == 18 or r.jam_kode == 17:
            time_str = "18:00"
        else:
            time_str = f"{int(r.jam_kode):02d}:00"
        
        datetime_str = f"{r.tanggal.strftime('%Y-%m-%d')} {time_str}"
        
        db_item = {
            'datetime': datetime_str,
            'tma_m': r.tma,
            'curah_hujan_mm': r.curah_hujan_mm,
            'smd_kanan_q_ls': r.smd_kanan_q_ls,
            'smd_kiri_q_ls': r.smd_kiri_q_ls,
            'jam_kode': r.jam_kode,
            'cuaca_kode': r.cuaca_kode,
            'source': 'DataBendungan_Manual'
        }
        
        existing_idx = next((i for i, item in enumerate(data) if item.get('datetime') == datetime_str), None)
        if existing_idx is not None:
            data[existing_idx].update(db_item)
        else:
            data.append(db_item)
            
    df = pd.DataFrame(data)
    
    if not df.empty and 'datetime' in df.columns:
        df['datetime_dt'] = pd.to_datetime(df['datetime'], errors='coerce')
        df = df.sort_values(by='datetime_dt')
        df = df.drop(columns=['datetime_dt'])
        
    response = HttpResponse(content_type='text/csv')
    response['Content-Disposition'] = 'attachment; filename="Bajulmati_Dataset_Aktual.csv"'
    
    df.to_csv(response, index=False)
    return response

def historical_data_view(request):
    """View to query and display historical data from the original dataset and manual DB logs."""
    target_date = request.GET.get('target_date', '')
    data = []
    
    engine = MLEngine()
    # 1. Fetch CSV records
    data = engine.get_historical_data(target_date)
    # Ensure all CSV items have a source tag
    for item in data:
        item['source'] = 'CSV'

    # 2. Fetch database records
    try:
        if target_date:
            db_records = DataBendungan.objects.filter(tanggal=target_date)
        else:
            db_records = DataBendungan.objects.all()
            
        for r in db_records:
            # Map jam_kode to standard time formats
            if r.jam_kode == 0 or r.jam_kode == 6:
                time_str = "06:00"
            elif r.jam_kode == 1 or r.jam_kode == 12:
                time_str = "12:00"
            elif r.jam_kode == 2 or r.jam_kode == 18 or r.jam_kode == 17:
                time_str = "18:00"
            else:
                time_str = f"{int(r.jam_kode):02d}:00"
            
            datetime_str = f"{r.tanggal.strftime('%Y-%m-%d')} {time_str}"
            
            # Determine status from TMA
            tma_val = r.tma
            if tma_val < 87.60:
                status_val = "Aman"
            elif tma_val < 89.487:
                status_val = "Waspada"
            elif tma_val < 91.30:
                status_val = "Siaga"
            else:
                status_val = "Awas"
                
            db_item = {
                'datetime': datetime_str,
                'tma_m': tma_val,
                'curah_hujan_mm': r.curah_hujan_mm,
                'smd_kanan_q_ls': r.smd_kanan_q_ls,
                'smd_kiri_q_ls': r.smd_kiri_q_ls,
                'status': status_val,
                'source': 'Database'
            }
            
            # Override if datetime already exists in CSV
            existing_idx = next((i for i, item in enumerate(data) if item['datetime'] == datetime_str), None)
            if existing_idx is not None:
                data[existing_idx] = db_item
            else:
                data.append(db_item)
        
        # Sort by datetime string descending
        data.sort(key=lambda x: x['datetime'], reverse=True)
    except Exception as e:
        print(f"Error merging DB records in historical_data_view: {e}")
        
    paginator = Paginator(data, 50)
    page_number = request.GET.get('page')
    page_obj = paginator.get_page(page_number)

    context = {
        'target_date': target_date,
        'page_obj': page_obj,
    }
    return render(request, 'dashboard/historical_data.html', context)

def reset_data_view(request):
    """Deletes all prediction records and batch sessions."""
    if request.method == 'POST':
        # Delete all records
        LogPrediksi.objects.all().delete()
        BatchSession.objects.all().delete()
        
        messages.success(request, "Seluruh riwayat data telah berhasil direset ke nol.")
        return redirect('index')
    return redirect('index')
