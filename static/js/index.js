document.addEventListener('DOMContentLoaded', function() {
    // 1. Number Counter Animation
    const stats = document.querySelectorAll('.stat-card .value');
    stats.forEach(stat => {
        // Only animate numbers, not text like 'v2.0'
        const text = stat.innerText.replace(/,/g, '');
        if (!isNaN(text) && text.trim() !== '') {
            const target = parseInt(text, 10);
            let current = 0;
            const increment = target / 30; // 30 frames
            if (target > 0) {
                const interval = setInterval(() => {
                    current += increment;
                    if (current >= target) {
                        stat.innerText = target;
                        clearInterval(interval);
                    } else {
                        stat.innerText = Math.ceil(current);
                    }
                }, 30);
            }
        }
    });

    // 2. Form Loading State
    const form = document.querySelector('form');
    const predictBtn = document.getElementById('predict-btn');
    
    if (form && predictBtn) {
        form.addEventListener('submit', function() {
            predictBtn.disabled = true;
            predictBtn.innerHTML = '⏳ Memproses...';
            predictBtn.style.opacity = '0.8';
        });
    }

    // 3. Auto-Refresh Toggle
    const autoRefreshCb = document.getElementById('auto-refresh-cb');
    let refreshInterval = null;

    if (autoRefreshCb) {
        // Load state from localStorage
        const savedState = localStorage.getItem('autoRefresh');
        if (savedState === 'true') {
            autoRefreshCb.checked = true;
            startAutoRefresh();
        }

        autoRefreshCb.addEventListener('change', function() {
            if (this.checked) {
                localStorage.setItem('autoRefresh', 'true');
                startAutoRefresh();
            } else {
                localStorage.setItem('autoRefresh', 'false');
                stopAutoRefresh();
            }
        });
    }

    function startAutoRefresh() {
        if (!refreshInterval) {
            refreshInterval = setInterval(() => {
                window.location.reload();
            }, 60000); // 60 seconds
        }
    }

    function stopAutoRefresh() {
        if (refreshInterval) {
            clearInterval(refreshInterval);
            refreshInterval = null;
        }
    }
});
