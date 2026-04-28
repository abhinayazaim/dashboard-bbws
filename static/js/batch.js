document.addEventListener('DOMContentLoaded', function() {
    const dropZone = document.getElementById('drop-zone');
    const batchForm = document.getElementById('batch-form');
    const submitBtn = document.getElementById('submit-btn');
    const fileInput = document.getElementById('file-upload');
    const fileNameDisplay = document.getElementById('file-name-display');
    const browseBtn = document.getElementById('browse-btn');
    const uploadContent = document.getElementById('upload-content');
    const uploadProgress = document.getElementById('upload-progress');
    const progressBarFill = document.getElementById('progress-bar-fill');
    const cancelBtn = document.getElementById('cancel-btn');

    if (!dropZone) return;

    // Trigger file input when browse button or drop zone is clicked
    browseBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        fileInput.click();
    });

    dropZone.addEventListener('click', () => {
        fileInput.click();
    });

    // Drag & drop events with visual feedback
    ['dragenter', 'dragover'].forEach(e => {
        dropZone.addEventListener(e, (ev) => { 
            ev.preventDefault(); 
            dropZone.style.borderColor = '#4a7cff';
            dropZone.style.background = 'rgba(74, 124, 255, 0.05)';
        });
    });
    
    ['dragleave', 'drop'].forEach(e => {
        dropZone.addEventListener(e, (ev) => { 
            ev.preventDefault(); 
            dropZone.style.borderColor = '';
            dropZone.style.background = '';
        });
    });

    dropZone.addEventListener('drop', (ev) => {
        const files = ev.dataTransfer.files;
        if (files.length > 0) {
            fileInput.files = files;
            handleFileSelect(fileInput);
        }
    });

    fileInput.addEventListener('change', () => {
        if (fileInput.files.length > 0) {
            handleFileSelect(fileInput);
        }
    });

    function handleFileSelect(input) {
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

        // Update UI
        fileNameDisplay.textContent = file.name;
        fileNameDisplay.classList.add('text-primary');
        
        // Enable buttons
        submitBtn.disabled = false;
        submitBtn.classList.remove('opacity-50', 'cursor-not-allowed');
        submitBtn.classList.add('opacity-100');
        
        cancelBtn.classList.remove('hidden');
    }

    function resetUpload() {
        fileInput.value = '';
        fileNameDisplay.textContent = 'Tarik & Lepas File di Sini';
        fileNameDisplay.classList.remove('text-primary');
        
        submitBtn.disabled = true;
        submitBtn.classList.add('opacity-50', 'cursor-not-allowed');
        submitBtn.classList.remove('opacity-100');
        
        cancelBtn.classList.add('hidden');
        uploadProgress.classList.add('hidden');
        uploadContent.classList.remove('opacity-0');
    }

    cancelBtn.addEventListener('click', (e) => {
        e.preventDefault();
        e.stopPropagation();
        resetUpload();
    });

    // Upload Simulation on form submit
    if (batchForm && submitBtn) {
        batchForm.addEventListener('submit', function(e) {
            // Prevent immediate submission to show simulation
            e.preventDefault();
            
            submitBtn.disabled = true;
            submitBtn.innerHTML = '<span class="material-symbols-outlined animate-spin">sync</span> Mengunggah...';
            cancelBtn.classList.add('hidden');
            
            uploadContent.classList.add('opacity-0');
            setTimeout(() => {
                uploadProgress.classList.remove('hidden');
                uploadProgress.classList.add('flex');
            }, 300);
            
            let progress = 0;
            const interval = setInterval(() => {
                progress += Math.random() * 20; // Random jump
                if (progress >= 100) {
                    progress = 100;
                    clearInterval(interval);
                    progressBarFill.style.width = '100%';
                    submitBtn.innerHTML = '<span class="material-symbols-outlined animate-spin">sync</span> Memproses...';
                    
                    // Actually submit the form after simulation
                    setTimeout(() => {
                        batchForm.submit();
                    }, 500);
                } else {
                    progressBarFill.style.width = progress + '%';
                }
            }, 200);
        });
    }
});
