import json

notebook_path = r'c:\Users\abhinaya\.gemini\antigravity\scratch\bajulmati_dashboard\models\LSTM_Bajulmati.ipynb'
output_path = r'c:\Users\abhinaya\.gemini\antigravity\scratch\bajulmati_dashboard\scratch\notebook_extracted.txt'

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

with open(output_path, 'w', encoding='utf-8') as out:
    out.write(f"Total cells: {len(nb.get('cells', []))}\n")
    for i, cell in enumerate(nb.get('cells', [])):
        if cell.get('cell_type') == 'code':
            source = "".join(cell.get('source', []))
            out.write(f"\n======================================\n")
            out.write(f"CELL {i} (Code)\n")
            out.write(f"======================================\n")
            out.write(source)
            out.write("\n")

print("Notebook code cells written to scratch/notebook_extracted.txt")
