import csv
import io
from django.http import HttpResponse
from .models import PredictionRecord


def export_history_to_csv(request):
    """Export prediction history to CSV, respecting current filters."""
    query = PredictionRecord.objects.all()

    date_from = request.GET.get('date_from', '')
    date_to = request.GET.get('date_to', '')
    status_filter = request.GET.get('status', '')

    if date_from:
        query = query.filter(created_at__date__gte=date_from)
    if date_to:
        query = query.filter(created_at__date__lte=date_to)
    if status_filter:
        query = query.filter(status=status_filter)

    response = HttpResponse(content_type='text/csv')
    response['Content-Disposition'] = 'attachment; filename="bajulmati_history.csv"'

    writer = csv.writer(response)
    writer.writerow([
        'No', 'Waktu Prediksi', 'TMA Prediksi (m)',
        'Curah Hujan (mm)', 'Debit Kanan (L/s)', 'Debit Kiri (L/s)',
        'Status', 'Sumber'
    ])

    for i, p in enumerate(query, 1):
        writer.writerow([
            i,
            p.created_at.strftime('%Y-%m-%d %H:%M:%S'),
            round(p.tma_predicted, 3),
            p.curah_hujan_mm,
            p.smd_kanan_q_ls,
            p.smd_kiri_q_ls,
            p.status,
            p.source,
        ])

    return response


def export_history_to_pdf(request):
    """Export prediction history to PDF."""
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
        from reportlab.lib.styles import getSampleStyleSheet
        from reportlab.lib import colors
    except ImportError:
        return HttpResponse("reportlab not installed. Use CSV export instead.", status=500)

    query = PredictionRecord.objects.all()

    date_from = request.GET.get('date_from', '')
    date_to = request.GET.get('date_to', '')
    status_filter = request.GET.get('status', '')

    if date_from:
        query = query.filter(created_at__date__gte=date_from)
    if date_to:
        query = query.filter(created_at__date__lte=date_to)
    if status_filter:
        query = query.filter(status=status_filter)

    response = HttpResponse(content_type='application/pdf')
    response['Content-Disposition'] = 'attachment; filename="laporan_bajulmati.pdf"'

    doc = SimpleDocTemplate(response, pagesize=A4)
    elements = []
    styles = getSampleStyleSheet()

    elements.append(Paragraph("Laporan Prediksi TMA Bendungan Bajulmati", styles['Title']))
    elements.append(Spacer(1, 12))

    filter_text = "Filter: "
    if date_from:
        filter_text += f"Dari {date_from} "
    if date_to:
        filter_text += f"Sampai {date_to} "
    if status_filter:
        filter_text += f"Status: {status_filter}"
    if filter_text == "Filter: ":
        filter_text += "Semua data"
    elements.append(Paragraph(filter_text, styles['Normal']))
    elements.append(Spacer(1, 12))

    data = [['No', 'Waktu', 'TMA (m)', 'Status']]
    for i, p in enumerate(query[:500], 1):
        data.append([
            str(i),
            p.created_at.strftime('%Y-%m-%d %H:%M'),
            f"{p.tma_predicted:.3f}",
            p.status,
        ])

    if query.count() > 500:
        data.append(['...', f'Total: {query.count()} data', '...', '...'])

    table = Table(data, colWidths=[40, 150, 80, 80])
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1a1d27')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#f5f5f5')),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f0f0f0')]),
    ]))

    elements.append(table)
    doc.build(elements)

    return response
