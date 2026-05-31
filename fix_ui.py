import os

base_dir = r'c:\Users\abhinaya\.gemini\antigravity\scratch\bajulmati_dashboard'

# 1. urls.py
urls_path = os.path.join(base_dir, 'dashboard', 'urls.py')
with open(urls_path, 'r', encoding='utf-8') as f:
    urls = f.read()

urls = urls.replace("'predict/'", "'input-data/'")
urls = urls.replace("name='predict'", "name='input_data'")
urls = urls.replace("'batch/'", "'prediksi/'")
urls = urls.replace("name='batch_predict'", "name='prediksi'")

with open(urls_path, 'w', encoding='utf-8') as f:
    f.write(urls)

# 2. base.html
base_path = os.path.join(base_dir, 'dashboard', 'templates', 'base.html')
with open(base_path, 'r', encoding='utf-8') as f:
    base = f.read()

# Sidebar renames
base = base.replace("Beranda", "Beranda") # It's already Beranda or Dashboard. Ah, earlier it was Dashboard in UI text but I changed it.
base = base.replace(">Dashboard<", ">Beranda<")
base = base.replace("url 'predict'", "url 'input_data'")
base = base.replace("url_name == 'predict'", "url_name == 'input_data'")
base = base.replace(">Input Manual<", ">Input Data<")

base = base.replace("url 'batch_predict'", "url 'prediksi'")
base = base.replace("url_name == 'batch_predict'", "url_name == 'prediksi'")
base = base.replace(">Upload File<", ">Prediksi (Upload File)<")

# Add Retrain Model to Settings
settings_content = '''
                <div class="pt-4 border-t border-outline-variant/30">
                    <p class="text-on-surface font-medium mb-1">Pelatihan Model Lanjutan</p>
                    <p class="text-xs text-outline mb-3">Latih ulang model di latar belakang menggunakan data terbaru.</p>
                    <button id="btnRetrainModelSettings" type="button" class="w-full py-2 bg-secondary-container/20 text-secondary border border-secondary/50 rounded hover:bg-secondary hover:text-on-secondary transition-all font-medium flex items-center justify-center gap-2">
                        <span class="material-symbols-outlined text-sm">model_training</span>
                        Retrain Model
                    </button>
                </div>
'''
if 'Pelatihan Model Lanjutan' not in base:
    base = base.replace('<div class="pt-4 border-t border-outline-variant/30">', settings_content + '\n                <div class="pt-4 border-t border-outline-variant/30">', 1)

# Update Help text
old_help = '''<li>Gunakan menu <strong>Input Manual</strong> untuk memprediksi data tunggal secara real-time.</li>
                        <li>Gunakan menu <strong>Unggah Batch</strong> untuk memprediksi banyak data sekaligus menggunakan file CSV.</li>'''
new_help = '''<li>Gunakan menu <strong>Input Data</strong> untuk menyimpan catatan TMA dan curah hujan harian ke dalam database historis.</li>
                        <li>Gunakan menu <strong>Prediksi (Upload File)</strong> untuk memprediksi TMA hari esok menggunakan data terbaru atau memprediksi banyak data sekaligus melalui unggah file CSV.</li>'''
base = base.replace(old_help, new_help)

# Add retrain JS handler in base.html
retrain_js = '''
            document.getElementById('btnRetrainModelSettings')?.addEventListener('click', function() {
                const btn = this;
                btn.disabled = true;
                btn.innerHTML = 'Memulai Training...';
                
                // Get CSRF from anywhere, e.g. the reset form
                const csrf = document.querySelector('[name=csrfmiddlewaretoken]')?.value;
                if(!csrf) { alert('CSRF token missing'); return; }

                fetch("/api/retrain-model/", {
                    method: 'POST',
                    headers: { 'X-CSRFToken': csrf }
                })
                .then(r => r.json())
                .then(d => {
                    btn.innerHTML = 'Training Berjalan';
                    alert(d.message || 'Proses retraining dimulai di background.');
                })
                .catch(e => {
                    btn.disabled = false;
                    btn.innerHTML = 'Retrain Model';
                    alert('Gagal memulai retraining.');
                });
            });
'''
if 'btnRetrainModelSettings' not in base[base.rfind('<script>'):]:
    base = base.replace('// Bindings', '// Bindings\n' + retrain_js)

with open(base_path, 'w', encoding='utf-8') as f:
    f.write(base)

# 3. Rename templates
templates_dir = os.path.join(base_dir, 'dashboard', 'templates', 'dashboard')
if os.path.exists(os.path.join(templates_dir, 'predict.html')):
    os.rename(os.path.join(templates_dir, 'predict.html'), os.path.join(templates_dir, 'input_data.html'))
if os.path.exists(os.path.join(templates_dir, 'batch.html')):
    os.rename(os.path.join(templates_dir, 'batch.html'), os.path.join(templates_dir, 'prediksi.html'))

print('Renames done')
