import csv
import io
from datetime import datetime
from django.http import HttpResponse
from .models import PredictionRecord


def _get_filtered_query(request):
    """Helper to apply common filters from GET params."""
    query = PredictionRecord.objects.all()
    date_from = request.GET.get('date_from', '')
    date_to = request.GET.get('date_to', '')
    status_filter = request.GET.get('status', '')
    search_query = request.GET.get('q', '')

    if date_from:
        query = query.filter(created_at__date__gte=date_from)
    if date_to:
        query = query.filter(created_at__date__lte=date_to)
    if status_filter and status_filter != 'all':
        query = query.filter(status=status_filter)
    if search_query:
        try:
            obs_date = datetime.strptime(search_query, '%Y-%m-%d').date()
            query = query.filter(waktu__date=obs_date)
        except ValueError:
            pass
    return query


def export_history_to_csv(request):
    """Export prediction history to CSV, respecting current filters."""
    query = _get_filtered_query(request)

    response = HttpResponse(content_type='text/csv')
    response['Content-Disposition'] = 'attachment; filename="bajulmati_history.csv"'

    writer = csv.writer(response)
    writer.writerow([
        'No', 'Waktu Observasi', 'Waktu Prediksi', 'TMA Prediksi (m)',
        'Curah Hujan (mm)', 'Debit Kanan (L/s)', 'Debit Kiri (L/s)',
        'Status', 'Sumber'
    ])

    for i, p in enumerate(query, 1):
        writer.writerow([
            i,
            p.waktu.strftime('%Y-%m-%d %H:%M:%S') if p.waktu else '-',
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
    """Export a professional analytical PDF report."""
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer,
                                        Table, TableStyle, HRFlowable)
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib import colors
        from reportlab.lib.units import cm
        from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
    except ImportError:
        return HttpResponse("reportlab tidak terinstall.", status=500)

    query = _get_filtered_query(request)
    records = list(query.order_by('waktu', 'created_at'))

    # --- Analytics ---
    total = len(records)
    danger_count = sum(1 for r in records if r.status == 'Bahaya')
    normal_count = sum(1 for r in records if r.status == 'Normal')
    danger_pct = (danger_count / total * 100) if total > 0 else 0

    tma_values = [r.tma_predicted for r in records if r.tma_predicted]
    tma_avg = sum(tma_values) / len(tma_values) if tma_values else 0
    tma_max = max(tma_values) if tma_values else 0
    tma_min = min(tma_values) if tma_values else 0
    threshold_val = records[0].threshold_used if records else 88.3794

    rain_values = [r.curah_hujan_mm for r in records if r.curah_hujan_mm is not None]
    rain_avg = sum(rain_values) / len(rain_values) if rain_values else 0

    date_from = request.GET.get('date_from', '')
    date_to = request.GET.get('date_to', '')
    generated_at = datetime.now().strftime('%d %B %Y, %H:%M WIB')

    # --- Colors ---
    DARK_BLUE = colors.HexColor('#1F3864')
    MID_BLUE = colors.HexColor('#2d5a9e')
    LIGHT_GRAY = colors.HexColor('#f4f6f9')
    DANGER_RED = colors.HexColor('#dc3545')
    NORMAL_GREEN = colors.HexColor('#27a644')
    TEXT_DARK = colors.HexColor('#1a1a2e')
    BORDER_COLOR = colors.HexColor('#dee2e6')
    ROW_ALT = colors.HexColor('#f0f4fa')
    DANGER_ROW = colors.HexColor('#fff0f0')

    # --- Styles ---
    styles = getSampleStyleSheet()

    def style(name, **kwargs):
        return ParagraphStyle(name, parent=styles['Normal'], **kwargs)

    title_s = style('T', fontSize=18, fontName='Helvetica-Bold', textColor=DARK_BLUE,
                    spaceAfter=2, alignment=TA_CENTER)
    sub_s = style('Sub', fontSize=9, textColor=colors.HexColor('#6c757d'),
                  alignment=TA_CENTER, spaceAfter=2)
    section_s = style('Sec', fontSize=12, fontName='Helvetica-Bold', textColor=DARK_BLUE,
                      spaceBefore=12, spaceAfter=6)
    body_s = style('Body', fontSize=9, textColor=TEXT_DARK, leading=15)
    cell_s = style('Cell', fontSize=8, textColor=TEXT_DARK, alignment=TA_CENTER)
    cell_l = style('CellL', fontSize=8, textColor=TEXT_DARK, alignment=TA_LEFT)
    danger_s = style('Dan', fontSize=8, fontName='Helvetica-Bold', textColor=DANGER_RED,
                     alignment=TA_CENTER)
    normal_s = style('Nor', fontSize=8, fontName='Helvetica-Bold', textColor=NORMAL_GREEN,
                     alignment=TA_CENTER)
    kpi_label_s = style('KL', fontSize=7, fontName='Helvetica', textColor=colors.HexColor('#6c757d'),
                         alignment=TA_CENTER)
    kpi_val_s = style('KV', fontSize=16, fontName='Helvetica-Bold', textColor=DARK_BLUE,
                       alignment=TA_CENTER)
    kpi_danger_s = style('KD', fontSize=16, fontName='Helvetica-Bold', textColor=DANGER_RED,
                          alignment=TA_CENTER)
    footer_s = style('Foot', fontSize=7, textColor=colors.HexColor('#adb5bd'),
                     alignment=TA_CENTER, spaceBefore=4)
    analysis_s = style('Ana', fontSize=9, textColor=TEXT_DARK, leading=15,
                        leftIndent=6, rightIndent=6)

    # --- Build ---
    response = HttpResponse(content_type='application/pdf')
    response['Content-Disposition'] = 'attachment; filename="laporan_bajulmati.pdf"'
    buf = io.BytesIO()

    doc = SimpleDocTemplate(buf, pagesize=A4,
                            rightMargin=2*cm, leftMargin=2*cm,
                            topMargin=2*cm, bottomMargin=2*cm)
    E = []  # elements

    # === HEADER BLOCK ===
    hdr = Table([
        [Paragraph("LAPORAN PREDIKSI TMA", title_s)],
        [Paragraph("Bendungan Bajulmati — Sistem Prediksi Hidrologi LSTM-Attention", sub_s)],
        [Paragraph(f"Digenerate: {generated_at}", sub_s)],
    ], colWidths=[17*cm])
    hdr.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), LIGHT_GRAY),
        ('TOPPADDING', (0, 0), (-1, -1), 8),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
    ]))
    E.append(hdr)
    E.append(Spacer(1, 0.3*cm))

    # Filter info
    filter_parts = []
    if date_from: filter_parts.append(f"Dari: {date_from}")
    if date_to: filter_parts.append(f"Sampai: {date_to}")
    filter_parts.append(f"Total: {total:,} record")
    E.append(Paragraph("  |  ".join(filter_parts), sub_s))
    E.append(HRFlowable(width="100%", thickness=1.5, color=DARK_BLUE, spaceAfter=8))

    # === KPI SUMMARY ===
    E.append(Paragraph("Ringkasan Analisis", section_s))

    def kpi_cell(label, value, val_style=kpi_val_s):
        return Table([
            [Paragraph(label, kpi_label_s)],
            [Paragraph(value, val_style)],
        ], colWidths=[3.2*cm])

    kpi_row = [[
        kpi_cell("TOTAL DATA", f"{total:,}"),
        kpi_cell("STATUS BAHAYA", f"{danger_count:,}\n({danger_pct:.1f}%)", kpi_danger_s),
        kpi_cell("STATUS NORMAL", f"{normal_count:,}"),
        kpi_cell("RATA-RATA TMA (m)", f"{tma_avg:.3f}"),
        kpi_cell("MAKS / MIN (m)", f"{tma_max:.3f} /\n{tma_min:.3f}"),
    ]]
    kpi_table = Table(kpi_row, colWidths=[3.4*cm]*5, hAlign='CENTER')
    kpi_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), LIGHT_GRAY),
        ('BOX', (0, 0), (0, -1), 0.5, BORDER_COLOR),
        ('BOX', (1, 0), (1, -1), 0.5, BORDER_COLOR),
        ('BOX', (2, 0), (2, -1), 0.5, BORDER_COLOR),
        ('BOX', (3, 0), (3, -1), 0.5, BORDER_COLOR),
        ('BOX', (4, 0), (4, -1), 0.5, BORDER_COLOR),
        ('TOPPADDING', (0, 0), (-1, -1), 8),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ('LEFTPADDING', (0, 0), (-1, -1), 4),
        ('RIGHTPADDING', (0, 0), (-1, -1), 4),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ]))
    E.append(kpi_table)
    E.append(Spacer(1, 0.4*cm))

    # === ANALYSIS TEXT ===
    E.append(Paragraph("Interpretasi & Analisis", section_s))

    lines = []
    if total == 0:
        lines.append("Tidak ada data yang sesuai dengan filter yang dipilih.")
    else:
        lines.append(
            f"Dari <b>{total:,}</b> data prediksi yang dianalisis, ditemukan <b>{danger_count:,} record "
            f"({danger_pct:.1f}%)</b> berstatus <b>BAHAYA</b> dan <b>{normal_count:,} record</b> berstatus <b>NORMAL</b>."
        )
        if tma_avg > 0:
            lines.append(
                f"Rata-rata Tinggi Muka Air (TMA) tercatat sebesar <b>{tma_avg:.3f} m</b>, dengan nilai tertinggi "
                f"<b>{tma_max:.3f} m</b> dan terendah <b>{tma_min:.3f} m</b>. "
                f"Ambang batas siaga yang digunakan: <b>{threshold_val:.4f} m</b>."
            )
        if rain_avg > 0:
            lines.append(f"Rata-rata curah hujan pada periode ini adalah <b>{rain_avg:.2f} mm</b>.")

        if danger_pct > 50:
            rec = ("<b>PERHATIAN:</b> Lebih dari separuh data (>50%) menunjukkan kondisi BAHAYA. "
                   "Diperlukan pemantauan intensif dan tindakan mitigasi segera.")
        elif danger_pct > 20:
            rec = ("Persentase status BAHAYA cukup signifikan. "
                   "Disarankan untuk meningkatkan frekuensi pemantauan pada periode ini.")
        else:
            rec = ("Kondisi umum pada periode ini tergolong aman. "
                   "Tetap lakukan pemantauan rutin sesuai prosedur operasional.")
        lines.append(rec)

    ana_block = Table([[Paragraph("<br/><br/>".join(lines), analysis_s)]],
                       colWidths=[17*cm])
    ana_block.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#eef4ff')),
        ('BOX', (0, 0), (-1, -1), 1, MID_BLUE),
        ('TOPPADDING', (0, 0), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 10),
        ('LEFTPADDING', (0, 0), (-1, -1), 14),
        ('RIGHTPADDING', (0, 0), (-1, -1), 14),
    ]))
    E.append(ana_block)
    E.append(Spacer(1, 0.4*cm))

    # === DATA TABLE ===
    E.append(Paragraph("Detail Data Prediksi", section_s))

    hdr_s = ParagraphStyle('Hdr', fontSize=8, fontName='Helvetica-Bold',
                            textColor=colors.white, alignment=TA_CENTER)

    tbl_data = [[
        Paragraph('No', hdr_s),
        Paragraph('Waktu Observasi', hdr_s),
        Paragraph('Waktu Prediksi', hdr_s),
        Paragraph('TMA (m)', hdr_s),
        Paragraph('Hujan (mm)', hdr_s),
        Paragraph('Status', hdr_s),
        Paragraph('Sumber', hdr_s),
    ]]

    display = records[:500]
    for i, p in enumerate(display, 1):
        is_d = p.status == 'Bahaya'
        obs = p.waktu.strftime('%d/%m/%Y %H:%M') if p.waktu else '-'
        pred_time = p.created_at.strftime('%d/%m/%Y %H:%M')
        tbl_data.append([
            Paragraph(str(i), cell_s),
            Paragraph(obs, cell_l),
            Paragraph(pred_time, cell_l),
            Paragraph(f"{p.tma_predicted:.3f}", danger_s if is_d else cell_s),
            Paragraph(f"{p.curah_hujan_mm:.1f}", cell_s),
            Paragraph(p.status, danger_s if is_d else normal_s),
            Paragraph(p.source, cell_s),
        ])

    if len(records) > 500:
        note_s = ParagraphStyle('N', fontSize=8, textColor=colors.HexColor('#6c757d'),
                                 fontName='Helvetica-Oblique', alignment=TA_CENTER)
        tbl_data.append([
            Paragraph(f'Menampilkan 500 dari {len(records):,} data total', note_s),
            '', '', '', '', '', ''
        ])

    col_w = [0.8*cm, 3.5*cm, 3.5*cm, 2*cm, 2*cm, 2*cm, 2*cm]
    dtbl = Table(tbl_data, colWidths=col_w, repeatRows=1)
    ts = TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), DARK_BLUE),
        ('LINEBELOW', (0, 0), (-1, 0), 1.5, DARK_BLUE),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, ROW_ALT]),
        ('GRID', (0, 0), (-1, -1), 0.3, BORDER_COLOR),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('TOPPADDING', (0, 0), (-1, -1), 5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
    ])
    # Danger row highlight
    for i, p in enumerate(display, 1):
        if p.status == 'Bahaya':
            ts.add('BACKGROUND', (0, i), (-1, i), DANGER_ROW)
    # "More rows" note spans all columns
    if len(records) > 500:
        last_row = len(tbl_data) - 1
        ts.add('SPAN', (0, last_row), (-1, last_row))
        ts.add('BACKGROUND', (0, last_row), (-1, last_row), LIGHT_GRAY)

    dtbl.setStyle(ts)
    E.append(dtbl)

    # === FOOTER ===
    E.append(Spacer(1, 0.4*cm))
    E.append(HRFlowable(width="100%", thickness=0.5, color=BORDER_COLOR))
    E.append(Paragraph(
        f"Laporan digenerate otomatis oleh Sistem Prediksi Hidrologi Bendungan Bajulmati  |  {generated_at}",
        footer_s
    ))

    doc.build(E)
    buf.seek(0)
    response.write(buf.read())
    return response
