document.addEventListener('DOMContentLoaded', function() {
    const form = document.getElementById('predict-form');
    const predictBtn = document.getElementById('predict-btn');
    const clearBtn = document.getElementById('clear-btn');
    const copyBtn = document.getElementById('copy-result-btn');

    // 1. Real-time Validation
    const inputs = document.querySelectorAll('#predict-form input[type="number"], #predict-form input[type="text"], #predict-form select');
    
    inputs.forEach(input => {
        input.addEventListener('input', function() {
            if (this.checkValidity()) {
                this.style.borderColor = '#22c55e'; // Green for valid
            } else {
                this.style.borderColor = '#ef4444'; // Red for invalid
            }
        });
        
        // Remove styling on blur if empty to not overwhelm user
        input.addEventListener('blur', function() {
            if (this.value === '') {
                this.style.borderColor = ''; 
            }
        });
    });

    // 2. Form Loading State
    if (form && predictBtn) {
        form.addEventListener('submit', function() {
            predictBtn.disabled = true;
            predictBtn.innerHTML = '⏳ Memproses...';
            predictBtn.style.opacity = '0.8';
        });
    }

    // 3. Clear Form
    if (clearBtn && form) {
        clearBtn.addEventListener('click', function() {
            form.reset();
            inputs.forEach(input => {
                input.style.borderColor = ''; // Reset borders
            });
        });
    }

    // 4. Copy Result
    if (copyBtn) {
        copyBtn.addEventListener('click', function() {
            const resultText = this.getAttribute('data-result');
            navigator.clipboard.writeText(resultText).then(() => {
                const originalText = this.innerHTML;
                this.innerHTML = '✅ Disalin';
                this.style.color = '#22c55e';
                this.style.borderColor = '#22c55e';
                
                setTimeout(() => {
                    this.innerHTML = originalText;
                    this.style.color = '';
                    this.style.borderColor = '';
                }, 2000);
            }).catch(err => {
                console.error('Failed to copy text: ', err);
            });
        });
    }
    
    // Result Animation (Fade In)
    const resultCard = document.querySelector('.result-card');
    if (resultCard && resultCard.querySelector('.result-value')) {
        resultCard.style.opacity = '0';
        resultCard.style.transform = 'translateY(10px)';
        resultCard.style.transition = 'opacity 0.5s ease, transform 0.5s ease';
        
        setTimeout(() => {
            resultCard.style.opacity = '1';
            resultCard.style.transform = 'translateY(0)';
        }, 100);
    }
});
