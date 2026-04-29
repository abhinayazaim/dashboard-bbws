document.addEventListener('DOMContentLoaded', function() {
    // 1. Clickable Rows
    const rows = document.querySelectorAll('.clickable-row');
    rows.forEach(row => {
        row.addEventListener('click', function(e) {
            // Prevent if clicking on something that shouldn't trigger row click
            if (e.target.tagName.toLowerCase() === 'a' || e.target.tagName.toLowerCase() === 'button') {
                return;
            }
            
            // Highlight row temporarily to show interaction
            const originalBg = this.style.backgroundColor;
            this.style.backgroundColor = 'rgba(74, 124, 255, 0.1)';
            
            setTimeout(() => {
                this.style.backgroundColor = originalBg;
                
                // Show detail mockup (could be replaced with actual modal/navigation)
                const id = this.getAttribute('data-id') || (forloop_counter + 1);
                const tma = this.querySelector('td:nth-child(4)').innerText;
                const status = this.querySelector('td:nth-child(7)').innerText.trim();
                alert(`Detail Prediksi\nTMA: ${tma}\nStatus: ${status}\n\nFitur detail lengkap akan segera hadir!`);
            }, 200);
        });

        // Hover effect
        row.addEventListener('mouseenter', function() {
            this.style.backgroundColor = 'var(--bg-lighter)';
        });
        row.addEventListener('mouseleave', function() {
            this.style.backgroundColor = '';
        });
    });

    // 2. Export Confirmation
    const exportBtns = document.querySelectorAll('.export-btn');
    exportBtns.forEach(btn => {
        btn.addEventListener('click', function(e) {
            const type = this.getAttribute('data-type');
            const confirmMsg = `Anda yakin ingin mengunduh riwayat data dalam format ${type}?`;
            if (!confirm(confirmMsg)) {
                e.preventDefault();
            }
        });
    });

    // 3. Quick Date Filters
    const quickBtns = document.querySelectorAll('.quick-filter-btn');
    const dateFrom = document.getElementById('date_from');
    const dateTo = document.getElementById('date_to');
    
    if (quickBtns.length > 0 && dateFrom && dateTo) {
        quickBtns.forEach(btn => {
            btn.addEventListener('click', function() {
                const days = parseInt(this.getAttribute('data-days'), 10);
                const today = new Date();
                
                // Format YYYY-MM-DD
                const toDateString = today.toISOString().split('T')[0];
                dateTo.value = toDateString;
                
                if (days === 0) {
                    dateFrom.value = toDateString;
                } else {
                    const fromDate = new Date();
                    fromDate.setDate(today.getDate() - days);
                    dateFrom.value = fromDate.toISOString().split('T')[0];
                }
                
                // Auto submit form
                this.closest('form').submit();
            });
        });
    }
});
