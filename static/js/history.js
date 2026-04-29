document.addEventListener('DOMContentLoaded', function() {
    // Select form and hidden inputs
    const filterForm = document.getElementById('filter-form');
    const dateFromInput = document.getElementById('id_date_from');
    const dateToInput = document.getElementById('id_date_to');

    if (!filterForm) return;

    // 1. Handle Filter Buttons
    const filterBtns = document.querySelectorAll('.filter-btn');
    filterBtns.forEach(btn => {
        btn.addEventListener('click', function() {
            const type = this.getAttribute('data-filter');
            const today = new Date();
            
            const formatDate = (date) => {
                const year = date.getFullYear();
                const month = String(date.getMonth() + 1).padStart(2, '0');
                const day = String(date.getDate()).padStart(2, '0');
                return `${year}-${month}-${day}`;
            };

            if (type === 'all') {
                dateFromInput.value = '';
                dateToInput.value = '';
            } else if (type === 'today') {
                const d = formatDate(today);
                dateFromInput.value = d;
                dateToInput.value = d;
            } else if (type === '7days') {
                const from = new Date();
                from.setDate(today.getDate() - 7);
                dateFromInput.value = formatDate(from);
                dateToInput.value = formatDate(today);
            } else if (type === '30days') {
                const from = new Date();
                from.setDate(today.getDate() - 30);
                dateFromInput.value = formatDate(from);
                dateToInput.value = formatDate(today);
            }
            
            filterForm.submit();
        });
    });

    // 2. Highlight Active Button
    const urlParams = new URLSearchParams(window.location.search);
    const dateFrom = urlParams.get('date_from');
    const dateTo = urlParams.get('date_to');
    const todayStr = new Date().toISOString().split('T')[0];

    function setBtnActive(type) {
        filterBtns.forEach(b => {
            if (b.getAttribute('data-filter') === type) {
                b.classList.remove('bg-[#121212]', 'text-on-surface');
                b.classList.add('bg-primary-container', 'text-white');
            } else {
                b.classList.add('bg-[#121212]', 'text-on-surface');
                b.classList.remove('bg-primary-container', 'text-white');
            }
        });
    }

    if (!dateFrom) {
        setBtnActive('all');
    } else if (dateFrom === todayStr) {
        setBtnActive('today');
    } else {
        // Simple logic for 7/30 highlight
        const diff = Math.ceil((new Date() - new Date(dateFrom)) / (1000 * 60 * 60 * 24));
        if (diff >= 6 && diff <= 8) setBtnActive('7days');
        else if (diff >= 28 && diff <= 32) setBtnActive('30days');
    }
});
