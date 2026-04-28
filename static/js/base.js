document.addEventListener('DOMContentLoaded', function() {
    // 1. Real-time Clock
    const clockElement = document.getElementById('system-clock');
    if (clockElement) {
        setInterval(() => {
            const now = new Date();
            clockElement.textContent = now.toLocaleTimeString('id-ID');
        }, 1000);
    }

    // 2. Alert Auto-dismiss
    const alerts = document.querySelectorAll('.alert');
    alerts.forEach(alert => {
        setTimeout(() => {
            alert.style.transition = 'opacity 0.5s ease';
            alert.style.opacity = '0';
            setTimeout(() => alert.remove(), 500);
        }, 5000); // 5 seconds
    });

    // 3. Scroll-to-Top Button
    const scrollTopBtn = document.createElement('button');
    scrollTopBtn.innerHTML = '⬆️';
    scrollTopBtn.className = 'scroll-top-btn';
    scrollTopBtn.style.cssText = `
        position: fixed;
        bottom: 30px;
        right: 30px;
        background: var(--primary-color);
        color: white;
        border: none;
        border-radius: 50%;
        width: 45px;
        height: 45px;
        font-size: 20px;
        cursor: pointer;
        display: none;
        box-shadow: 0 4px 12px rgba(0,0,0,0.2);
        z-index: 1000;
        transition: transform 0.3s, background 0.3s;
    `;
    document.body.appendChild(scrollTopBtn);

    window.addEventListener('scroll', () => {
        if (window.scrollY > 300) {
            scrollTopBtn.style.display = 'block';
        } else {
            scrollTopBtn.style.display = 'none';
        }
    });

    scrollTopBtn.addEventListener('click', () => {
        window.scrollTo({ top: 0, behavior: 'smooth' });
    });
    
    scrollTopBtn.addEventListener('mouseover', () => {
        scrollTopBtn.style.transform = 'translateY(-3px)';
        scrollTopBtn.style.background = 'var(--primary-hover)';
    });
    scrollTopBtn.addEventListener('mouseout', () => {
        scrollTopBtn.style.transform = 'translateY(0)';
        scrollTopBtn.style.background = 'var(--primary-color)';
    });

    // 4. Notifications Dropdown Mock
    const notifIcon = document.getElementById('notif-icon');
    if (notifIcon) {
        notifIcon.addEventListener('click', () => {
            alert('Belum ada notifikasi baru.');
        });
    }
});
