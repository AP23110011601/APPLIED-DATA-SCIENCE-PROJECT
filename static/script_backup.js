// DOM Elements with null checks
let predictionForm, resultCard, loadingSpinner, predictionBadge, predictionText, confidenceScore, epaCategoryText, probGood, probNotGood;
let aqiInput, monthSelect, stateSelect, parameterSelect, seasonSelect;
let isSubmitting = false; // Prevent multiple submissions

// Initialize DOM elements
function initializeDOMElements() {
    predictionForm = document.getElementById('predictionForm');
    resultCard = document.getElementById('resultCard');
    loadingSpinner = document.getElementById('loadingSpinner');
    predictionBadge = document.getElementById('predictionBadge');
    predictionText = document.getElementById('predictionText');
    confidenceScore = document.getElementById('confidenceScore');
    epaCategoryText = document.getElementById('epaCategoryText');
    probGood = document.getElementById('probGood');
    probNotGood = document.getElementById('probNotGood');
    
    // Form elements
    aqiInput = document.getElementById('aqi');
    monthSelect = document.getElementById('month');
    stateSelect = document.getElementById('state');
    parameterSelect = document.getElementById('parameter');
    seasonSelect = document.getElementById('season');
    
    // Check if all required elements exist
    if (!predictionForm || !aqiInput || !monthSelect) {
        console.error('Required form elements not found');
        return false;
    }
    return true;
}

// Event Listeners
document.addEventListener('DOMContentLoaded', function() {
    // Initialize DOM elements
    if (!initializeDOMElements()) {
        return;
    }
    
    // Set default month to current month
    if (monthSelect) {
        const currentMonth = new Date().getMonth() + 1;
        monthSelect.value = currentMonth;
    }
    
    // Add form submit event listener
    if (predictionForm) {
        predictionForm.addEventListener('submit', handlePrediction);
    }
    
    // Add input validation
    if (aqiInput) {
        aqiInput.addEventListener('input', validateAQI);
    }
});

// Validate AQI input
function validateAQI() {
    const value = parseFloat(aqiInput.value);
    if (value < 0) {
        aqiInput.value = 0;
    } else if (value > 500) {
        aqiInput.value = 500;
    }
}

// Handle prediction form submission
async function handlePrediction(event) {
    event.preventDefault();
    
    // Prevent multiple submissions
    if (isSubmitting) {
        return;
    }
    
    isSubmitting = true;
    
    // Show loading spinner
    showLoading();
    
    try {
        // Get form data with validation
        if (!aqiInput || !monthSelect || !stateSelect || !parameterSelect || !seasonSelect) {
            throw new Error('Form elements not found');
        }
        
        const formData = {
            aqi: parseFloat(aqiInput.value) || 0,
            month: parseInt(monthSelect.value) || 1,
            state: stateSelect.value || '',
            parameter: parameterSelect.value || '',
            season: seasonSelect.value || ''
        };
        
        // Send prediction request
        const response = await fetch('/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(formData)
        });
        
        if (!response.ok) {
            let errorData;
            try {
                errorData = await response.json();
            } catch (e) {
                errorData = { error: 'Server error occurred' };
            }
            throw new Error(errorData.error || `HTTP ${response.status}: Prediction failed`);
        }
        
        const result = await response.json();
        
        // Display results
        displayResults(result);
        
    } catch (error) {
        console.error('Prediction error:', error);
        showError(error.message || 'An unexpected error occurred');
    } finally {
        // Hide loading spinner
        hideLoading();
        isSubmitting = false;
    }
}

// Display prediction results
function displayResults(result) {
    // Validate result data
    if (!result || typeof result !== 'object') {
        showError('Invalid prediction result received');
        return;
    }
    
    // Update prediction badge
    const isGood = result.prediction === 'Good';
    if (predictionBadge && predictionText) {
        predictionBadge.className = `prediction-badge ${isGood ? 'good' : 'not-good'}`;
        predictionText.textContent = result.prediction || 'Unknown';
    }
    
    // Update confidence score
    if (confidenceScore) {
        confidenceScore.textContent = result.confidence || '0';
    }
    
    // Update EPA category
    if (epaCategoryText) {
        epaCategoryText.textContent = result.epa_category || 'Unknown';
    }
    
    // Update probabilities
    if (probGood) {
        probGood.textContent = result.probability_good || '0';
    }
    if (probNotGood) {
        probNotGood.textContent = result.probability_not_good || '0';
    }
    
    // Show result card with animation
    if (resultCard) {
        resultCard.style.display = 'block';
        resultCard.classList.add('fade-in');
        
        // Scroll to results
        setTimeout(() => {
            resultCard.scrollIntoView({ 
                behavior: 'smooth', 
                block: 'center' 
            });
        }, 100);
        
        // Remove animation class after animation completes
        setTimeout(() => {
            resultCard.classList.remove('fade-in');
        }, 500);
    }
}

// Show loading spinner
function showLoading() {
    if (loadingSpinner) {
        loadingSpinner.style.display = 'flex';
    }
    // Disable form during loading
    if (predictionForm) {
        predictionForm.style.opacity = '0.6';
        predictionForm.style.pointerEvents = 'none';
    }
}

// Hide loading spinner
function hideLoading() {
    if (loadingSpinner) {
        loadingSpinner.style.display = 'none';
    }
    // Re-enable form
    if (predictionForm) {
        predictionForm.style.opacity = '1';
        predictionForm.style.pointerEvents = 'auto';
    }
}

// Show error message with cleanup
let existingErrorAlert = null;

function showError(message) {
    // Remove existing error alert if present
    if (existingErrorAlert && existingErrorAlert.parentNode) {
        existingErrorAlert.parentNode.removeChild(existingErrorAlert);
    }
    
    // Create error alert
    const errorAlert = document.createElement('div');
    errorAlert.className = 'alert alert-danger alert-dismissible fade show position-fixed';
    errorAlert.style.cssText = 'top: 20px; right: 20px; z-index: 9999; min-width: 300px; max-width: 400px;';
    errorAlert.innerHTML = `
        <strong>Error!</strong> ${message}
        <button type="button" class="btn-close" data-bs-dismiss="alert" aria-label="Close"></button>
    `;
    
    // Add close button functionality
    const closeBtn = errorAlert.querySelector('.btn-close');
    if (closeBtn) {
        closeBtn.addEventListener('click', () => {
            if (errorAlert.parentNode) {
                errorAlert.parentNode.removeChild(errorAlert);
            }
            if (existingErrorAlert === errorAlert) {
                existingErrorAlert = null;
            }
        });
    }
    
    // Add to body
    document.body.appendChild(errorAlert);
    existingErrorAlert = errorAlert;
    
    // Auto remove after 5 seconds
    setTimeout(() => {
        if (errorAlert.parentNode) {
            errorAlert.parentNode.removeChild(errorAlert);
        }
        if (existingErrorAlert === errorAlert) {
            existingErrorAlert = null;
        }
    }, 5000);
}


// Add hover effects to form elements
function addFormInteractions() {
    const formControls = document.querySelectorAll('.form-control, .form-select');
    
    formControls.forEach(element => {
        element.addEventListener('focus', function() {
            this.parentElement.classList.add('focused');
        });
        
        element.addEventListener('blur', function() {
            this.parentElement.classList.remove('focused');
        });
    });
}

// Initialize form interactions and reset button
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => {
        addFormInteractions();
        addResetButton();
    });
} else {
    addFormInteractions();
    addResetButton();
}

// Add keyboard shortcuts
document.addEventListener('keydown', function(event) {
    // Ctrl/Cmd + Enter to submit form
    if ((event.ctrlKey || event.metaKey) && event.key === 'Enter') {
        if (document.activeElement && document.activeElement.form === predictionForm) {
            event.preventDefault();
            if (predictionForm && !isSubmitting) {
                predictionForm.dispatchEvent(new Event('submit'));
            }
        }
    }
    
    // Escape to hide results
    if (event.key === 'Escape' && resultCard && resultCard.style.display === 'block') {
        resultCard.style.display = 'none';
    }
});

// Add smooth scroll behavior
document.querySelectorAll('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener('click', function (e) {
        e.preventDefault();
        const target = document.querySelector(this.getAttribute('href'));
        if (target) {
            target.scrollIntoView({
                behavior: 'smooth'
            });
        }
    });
});

// Function to reset form
function resetForm() {
    if (!predictionForm) return;
    
    predictionForm.reset();
    if (resultCard) {
        resultCard.style.display = 'none';
    }
    
    // Set default values
    if (monthSelect) {
        const currentMonth = new Date().getMonth() + 1;
        monthSelect.value = currentMonth;
    }
    if (aqiInput) {
        aqiInput.value = '50';
    }
}

// Add reset button functionality (if needed)
function addResetButton() {
    const predictBtn = document.querySelector('.predict-btn');
    if (!predictBtn) return;
    
    const resetBtn = document.createElement('button');
    resetBtn.type = 'button';
    resetBtn.className = 'btn btn-secondary btn-sm mt-2';
    resetBtn.textContent = '🔄 Reset Form';
    resetBtn.addEventListener('click', resetForm);
    
    // Add reset button after predict button
    predictBtn.parentElement.appendChild(resetBtn);
}

// Console log for debugging
console.log('AQI Predictor JavaScript loaded successfully!');
