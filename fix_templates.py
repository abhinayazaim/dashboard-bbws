import os, glob

template_dir = r'c:\Users\abhinaya\.gemini\antigravity\scratch\bajulmati_dashboard\dashboard\templates\dashboard'
files = glob.glob(os.path.join(template_dir, '*.html'))

for file_path in files:
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Generic replaces for status conditionals
    content = content.replace("'Darurat' or result_status == 'Bahaya'", "'Awas'")
    content = content.replace("result_status == 'Darurat'", "result_status == 'Awas'")
    content = content.replace("result_status == 'Bahaya'", "result_status == 'Siaga'")
    content = content.replace("result_status == 'Normal'", "result_status == 'Aman'")
    
    content = content.replace("'Darurat' or p.status == 'Bahaya'", "'Awas'")
    content = content.replace("p.status == 'Darurat'", "p.status == 'Awas'")
    content = content.replace("p.status == 'Bahaya'", "p.status == 'Siaga'")
    content = content.replace("p.status == 'Normal'", "p.status == 'Aman'")
    
    content = content.replace("'Darurat' or row.status == 'Bahaya'", "'Awas'")
    content = content.replace("row.status == 'Darurat'", "row.status == 'Awas'")
    content = content.replace("row.status == 'Bahaya'", "row.status == 'Siaga'")
    content = content.replace("row.status == 'Normal'", "row.status == 'Aman'")
    
    content = content.replace("status_filter == 'Darurat'", "status_filter == 'Awas'")
    content = content.replace('value="Darurat"', 'value="Awas"')
    content = content.replace('>Darurat</option>', '>Awas</option>')
    
    content = content.replace("status_filter == 'Bahaya'", "status_filter == 'Siaga'")
    content = content.replace('value="Bahaya"', 'value="Siaga"')
    content = content.replace('>Bahaya</option>', '>Siaga</option>')
    
    content = content.replace("status_filter == 'Normal'", "status_filter == 'Aman'")
    content = content.replace('value="Normal"', 'value="Aman"')
    content = content.replace('>Normal</option>', '>Aman</option>')

    # Fix Aman (white), Waspada (green), Siaga (yellow), Awas (red)
    # The previous code had text-error, border-error, text-orange-400, border-orange-500
    # Let's fix that directly by finding where colors are applied.
    # In index.html, we have:
    # {% if p.status == 'Awas' %}font-bold text-error{% elif p.status == 'Waspada' %}font-bold text-orange-400{% else %}text-on-surface{% endif %}
    
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)
