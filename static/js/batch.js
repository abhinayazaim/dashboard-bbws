document.addEventListener('DOMContentLoaded', function() {
    const uploadZone = document.getElementById('uploadZone');
    const batchForm = document.getElementById('batchForm');
    const submitBtn = document.getElementById('submitBatchBtn');
    const progressContainer = document.getElementById('progressContainer');
    const progressBar = document.getElementById('progressBar');
    const progressText = document.getElementById('progressText');
    const fileInput = document.getElementById('id_csv_file');

    if (!uploadZone) return;

    // Drag & drop events with visual feedback
    ['dragenter', 'dragover'].forEach(e => {
        uploadZone.addEventListener(e, (ev) => { 
            ev.preventDefault(); 
            uploadZone.style.borderColor = 'var(--primary-color)';
            uploadZone.style.background = 'rgba(74, 124, 255, 0.05)';
        });
    });
    
    ['dragleave', 'drop'].forEach(e => {
        uploadZone.addEventListener(e, (ev) => { 
            ev.preventDefault(); 
            uploadZone.style.borderColor = '';
            uploadZone.style.background = '';
        });
    });

    uploadZone.addEventListener('drop', (ev) => {
        const files = ev.dataTransfer.files;
        if (files.length > 0) {
            fileInput.files = files;
            handleFileSelect(fileInput);
        }
    });

    // Make handleFileSelect globally available if called from HTML onclick
    window.handleFileSelect = function(input) {
        const file = input.files[0];
        if (!file) return;

        // Client-side Validation (Extension and Size)
        const validExtensions = ['.csv', '.xlsx', '.xls'];
        const fileName = file.name.toLowerCase();
        const isValidExtension = validExtensions.some(ext => fileName.endsWith(ext));
        
        if (!isValidExtension) {
            alert('Format file tidak didukung. Harap unggah file CSV atau Excel.');
            resetUpload();
            return;
        }

        const maxSizeMB = 10;
        if (file.size > maxSizeMB * 1024 * 1024) {
            alert(`Ukuran file terlalu besar (Maksimal ${maxSizeMB}MB).`);
            resetUpload();
            return;
        }

        document.getElementById('fileName').textContent = file.name;
        document.getElementById('previewSection').style.display = 'block';
        document.getElementById('actionBar').style.display = 'flex';
        uploadZone.style.display = 'none';

        // CSV preview
        if (file.name.endsWith('.csv')) {
            const reader = new FileReader();
            reader.onload = function(e) {
                const lines = e.target.result.split('\n').filter(l => l.trim());
                const headers = lines[0].split(',');
                const headRow = '<tr>' + headers.map(h => '<th>' + h.trim() + '</th>').join('') + '</tr>';
                document.getElementById('previewHead').innerHTML = headRow;

                let bodyHTML = '';
                const previewCount = Math.min(lines.length - 1, 4);
                for (let i = 1; i <= previewCount; i++) {
                    const cols = lines[i].split(',');
                    bodyHTML += '<tr>' + cols.map(c => '<td>' + c.trim() + '</td>').join('') + '</tr>';
                }
                document.getElementById('previewBody').innerHTML = bodyHTML;
                document.getElementById('rowCount').textContent = 'Showing ' + previewCount + ' of ' + (lines.length - 1) + ' rows';
            };
            reader.readAsText(file);
        } else {
            document.getElementById('previewHead').innerHTML = '<tr><th>File Excel terdeteksi - preview tidak tersedia</th></tr>';
            document.getElementById('previewBody').innerHTML = '';
            document.getElementById('rowCount').textContent = file.name;
        }
    };

    window.resetUpload = function() {
        fileInput.value = '';
        document.getElementById('previewSection').style.display = 'none';
        document.getElementById('actionBar').style.display = 'none';
        progressContainer.style.display = 'none';
        uploadZone.style.display = 'block';
        if (submitBtn) {
            submitBtn.disabled = false;
            submitBtn.innerHTML = '📈 Mulai Prediksi';
        }
    };

    // Upload Simulation on form submit
    if (batchForm && submitBtn) {
        batchForm.addEventListener('submit', function(e) {
            // Prevent immediate submission to show simulation
            e.preventDefault();
            
            submitBtn.disabled = true;
            submitBtn.innerHTML = '⏳ Mengunggah...';
            progressContainer.style.display = 'block';
            
            let progress = 0;
            const interval = setInterval(() => {
                progress += Math.random() * 15; // Random jump
                if (progress >= 100) {
                    progress = 100;
                    clearInterval(interval);
                    progressBar.style.width = '100%';
                    progressText.textContent = '100%';
                    submitBtn.innerHTML = 'Memproses Data...';
                    
                    // Actually submit the form after simulation
                    setTimeout(() => {
                        batchForm.submit();
                    }, 500);
                } else {
                    progressBar.style.width = progress + '%';
                    progressText.textContent = Math.round(progress) + '%';
                }
            }, 200);
        });
    }
});
