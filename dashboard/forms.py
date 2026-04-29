from django import forms

# Cuaca choices for dropdown
CUACA_CHOICES = [
    (0, 'Cerah'),
    (1, 'Berawan'),
    (2, 'Hujan'),
]

# Jam pengukuran choices
JAM_CHOICES = [
    (0, '06:00'),
    (1, '12:00'),
    (2, '18:00'),
]


class ManualPredictionForm(forms.Form):
    """User-facing manual prediction form. Lag/delta/rolling features are auto-computed."""
    waktu = forms.DateTimeField(
        widget=forms.DateTimeInput(attrs={
            'type': 'datetime-local',
            'class': 'form-input w-full bg-[#121212] border border-[#333333] text-[#e3e2e6] rounded py-2 px-3 focus:ring-[#4a7cff] focus:border-[#4a7cff]',
            'id': 'id_waktu',
        }),
        label='WAKTU',
    )
    curah_hujan_mm = forms.FloatField(
        widget=forms.NumberInput(attrs={
            'class': 'form-input w-full bg-[#121212] border border-[#333333] text-[#e3e2e6] rounded py-2 px-3 focus:ring-[#4a7cff] focus:border-[#4a7cff]',
            'step': '0.01',
            'placeholder': '0.00',
            'id': 'id_curah_hujan',
        }),
        label='CURAH HUJAN (MM)',
        initial=0.0,
    )
    cuaca_kode = forms.ChoiceField(
        choices=CUACA_CHOICES,
        widget=forms.Select(attrs={
            'class': 'form-input w-full bg-[#121212] border border-[#333333] text-[#e3e2e6] rounded py-2 px-3 focus:ring-[#4a7cff] focus:border-[#4a7cff]',
            'id': 'id_cuaca_kode',
        }),
        label='KODE CUACA',
    )
    jam_kode = forms.ChoiceField(
        choices=JAM_CHOICES,
        widget=forms.Select(attrs={
            'class': 'form-input w-full bg-[#121212] border border-[#333333] text-[#e3e2e6] rounded py-2 px-3 focus:ring-[#4a7cff] focus:border-[#4a7cff]',
            'id': 'id_jam_kode',
        }),
        label='JAM PENGUKURAN',
    )
    smd_kanan_q_ls = forms.FloatField(
        widget=forms.NumberInput(attrs={
            'class': 'form-input w-full bg-[#121212] border border-[#333333] text-[#e3e2e6] rounded py-2 px-3 focus:ring-[#4a7cff] focus:border-[#4a7cff]',
            'step': '0.1',
            'placeholder': '0.0',
            'id': 'id_smd_kanan',
        }),
        label='DEBIT SMD KANAN (L/S)',
        initial=0.0,
    )
    smd_kiri_q_ls = forms.FloatField(
        widget=forms.NumberInput(attrs={
            'class': 'form-input w-full bg-[#121212] border border-[#333333] text-[#e3e2e6] rounded py-2 px-3 focus:ring-[#4a7cff] focus:border-[#4a7cff]',
            'step': '0.1',
            'placeholder': '0.0',
            'id': 'id_smd_kiri',
        }),
        label='DEBIT SMD KIRI (L/S)',
        initial=0.0,
    )


class BatchUploadForm(forms.Form):
    csv_file = forms.FileField(
        label='Upload File',
        widget=forms.FileInput(attrs={
            'class': 'form-input w-full bg-[#121212] border border-[#333333] text-[#e3e2e6] rounded py-2 px-3 focus:ring-[#4a7cff] focus:border-[#4a7cff]',
            'accept': '.csv',
            'id': 'id_csv_file',
        })
    )
