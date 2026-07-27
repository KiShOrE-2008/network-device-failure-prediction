// Preset Configurations
const PRESETS = {
    "healthy-router": {
        "Device_Type": "Router",
        "CPU_Usage": 22.0,
        "Memory_Usage": 35.0,
        "Temperature": 38.0,
        "Uptime": 120.0,
        "Interface_Errors": 3,
        "Packet_Loss": 0.05,
        "Bandwidth_Usage": 45.0,
        "Log_Errors": 1
    },
    "healthy-switch": {
        "Device_Type": "Switch",
        "CPU_Usage": 15.0,
        "Memory_Usage": 28.0,
        "Temperature": 34.0,
        "Uptime": 245.0,
        "Interface_Errors": 1,
        "Packet_Loss": 0.0,
        "Bandwidth_Usage": 30.0,
        "Log_Errors": 0
    },
    "high-cpu": {
        "Device_Type": "Router",
        "CPU_Usage": 96.5,
        "Memory_Usage": 82.0,
        "Temperature": 65.0,
        "Uptime": 42.0,
        "Interface_Errors": 18,
        "Packet_Loss": 0.9,
        "Bandwidth_Usage": 94.0,
        "Log_Errors": 4
    },
    "thermal-alert": {
        "Device_Type": "Router",
        "CPU_Usage": 62.0,
        "Memory_Usage": 55.0,
        "Temperature": 87.5,
        "Uptime": 195.0,
        "Interface_Errors": 12,
        "Packet_Loss": 0.4,
        "Bandwidth_Usage": 68.0,
        "Log_Errors": 5
    },
    "packet-loss": {
        "Device_Type": "Router",
        "CPU_Usage": 45.0,
        "Memory_Usage": 48.0,
        "Temperature": 41.0,
        "Uptime": 320.0,
        "Interface_Errors": 210,
        "Packet_Loss": 8.5,
        "Bandwidth_Usage": 92.5,
        "Log_Errors": 11
    },
    "critical-failure": {
        "Device_Type": "Switch",
        "CPU_Usage": 95.0,
        "Memory_Usage": 94.0,
        "Temperature": 85.0,
        "Uptime": 410.0,
        "Interface_Errors": 412,
        "Packet_Loss": 9.8,
        "Bandwidth_Usage": 97.0,
        "Log_Errors": 28
    }
};

// DOM Elements
const navItems = document.querySelectorAll('.nav-item');
const tabContents = document.querySelectorAll('.tab-content');
const pageTitle = document.getElementById('page-title');
const activeModelSidebar = document.getElementById('active-model-sidebar');

// Stats DOM Elements
const statTotalDevices = document.getElementById('stat-total-devices');
const statHealthRate = document.getElementById('stat-health-rate');
const statFailedCount = document.getElementById('stat-failed-count');

// Form & Sliders Elements
const telemetryForm = document.getElementById('telemetry-form');
const cpuSlider = document.getElementById('cpu_usage');
const cpuVal = document.getElementById('cpu_val');
const memorySlider = document.getElementById('memory_usage');
const memoryVal = document.getElementById('memory_val');
const temperatureSlider = document.getElementById('temperature');
const temperatureVal = document.getElementById('temperature_val');
const uptimeSlider = document.getElementById('uptime');
const uptimeVal = document.getElementById('uptime_val');
const errorsSlider = document.getElementById('interface_errors');
const errorsVal = document.getElementById('errors_val');
const lossSlider = document.getElementById('packet_loss');
const lossVal = document.getElementById('loss_val');
const bandwidthSlider = document.getElementById('bandwidth_usage');
const bandwidthVal = document.getElementById('bandwidth_val');
const logSlider = document.getElementById('log_errors');
const logVal = document.getElementById('log_val');

// Results DOM Elements
const resultPercent = document.getElementById('result-percent');
const riskBadge = document.getElementById('risk-badge-element');
const riskIcon = document.getElementById('risk-icon');
const riskText = document.getElementById('risk-text');
const resultStatusSummary = document.getElementById('result-status-summary');
const advisoryList = document.getElementById('advisory-list-element');
const gaugeFillCircle = document.getElementById('gauge-fill-circle');

// Lightbox Elements
const lightbox = document.getElementById('lightbox');
const lightboxImg = document.getElementById('lightbox-img');
const lightboxCaption = document.getElementById('lightbox-caption');

// Reset Button Element
const resetBtn = document.getElementById('reset-btn');

// Active rows highlights (Model Performance table)
const rowXGB = document.getElementById('row-xgb');
const rowRF = document.getElementById('row-rf');
const rowLR = document.getElementById('row-lr');

// Default Nominal Telemetry Values
const DEFAULTS = {
    "Device_Type": "Router",
    "CPU_Usage": 50.0,
    "Memory_Usage": 60.0,
    "Temperature": 45.0,
    "Uptime": 100.0,
    "Interface_Errors": 10,
    "Packet_Loss": 0.5,
    "Bandwidth_Usage": 40.0,
    "Log_Errors": 2
};

// Global Debounce Timeout
let predictionTimeout = null;

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    setupTabSwitcher();
    setupSlidersListener();
    setupPresetsListener();
    setupResetListener();
    loadNetworkStats();
    
    // Run initial prediction based on default slider states
    triggerPrediction();
});

// Setup Reset Button Click Handler
function setupResetListener() {
    if (!resetBtn) return;
    
    resetBtn.addEventListener('click', () => {
        // Clear all selected states from presets
        document.querySelectorAll('.preset-btn').forEach(btn => btn.classList.remove('selected'));
        
        // Reset Device Type Switch
        if (DEFAULTS.Device_Type === "Router") {
            document.getElementById('type-router').checked = true;
        } else {
            document.getElementById('type-switch').checked = true;
        }
        
        // Reset Sliders & labels
        cpuSlider.value = DEFAULTS.CPU_Usage;
        cpuVal.innerText = DEFAULTS.CPU_Usage;
        
        memorySlider.value = DEFAULTS.Memory_Usage;
        memoryVal.innerText = DEFAULTS.Memory_Usage;
        
        temperatureSlider.value = DEFAULTS.Temperature;
        temperatureVal.innerText = DEFAULTS.Temperature;
        
        uptimeSlider.value = DEFAULTS.Uptime;
        uptimeVal.innerText = DEFAULTS.Uptime;
        
        errorsSlider.value = DEFAULTS.Interface_Errors;
        errorsVal.innerText = DEFAULTS.Interface_Errors;
        
        lossSlider.value = DEFAULTS.Packet_Loss;
        lossVal.innerText = DEFAULTS.Packet_Loss;
        
        bandwidthSlider.value = DEFAULTS.Bandwidth_Usage;
        bandwidthVal.innerText = DEFAULTS.Bandwidth_Usage;
        
        logSlider.value = DEFAULTS.Log_Errors;
        logVal.innerText = DEFAULTS.Log_Errors;
        
        // Run diagnosis with default settings
        triggerPrediction();
    });
}


// Tab Switcher
function setupTabSwitcher() {
    navItems.forEach(item => {
        item.addEventListener('click', () => {
            const tabId = item.getAttribute('data-tab');
            
            // Toggle active menu class
            navItems.forEach(btn => btn.classList.remove('active'));
            item.classList.add('active');
            
            // Toggle active tab content
            tabContents.forEach(content => content.classList.remove('active'));
            document.getElementById(tabId).classList.add('active');
            
            // Update Title
            let text = item.querySelector('span').innerText;
            pageTitle.innerText = text;
        });
    });
}

// Stats Loader
async function loadNetworkStats() {
    try {
        const response = await fetch('/api/stats');
        const data = await response.json();
        
        if (data.success) {
            statTotalDevices.innerText = Number(data.total_devices).toLocaleString();
            statHealthRate.innerText = `${data.health_rate}%`;
            statFailedCount.innerText = Number(data.failed_count).toLocaleString();
            
            activeModelSidebar.innerText = data.active_model;
            
            // Highlight active row in metrics table
            highlightActiveModelRow(data.active_model);
        }
    } catch (error) {
        console.error('Error fetching statistics:', error);
        activeModelSidebar.innerText = 'Offline';
    }
}

// Highlight the model that is currently loaded in backend
function highlightActiveModelRow(modelName) {
    // Reset highlights
    [rowXGB, rowRF, rowLR].forEach(row => {
        if (row) {
            row.classList.remove('active-row-highlight');
            const badge = row.querySelector('.badge');
            if (badge) {
                badge.className = 'badge fallback-badge';
                badge.innerText = 'CANDIDATE';
            }
        }
    });

    let targetRow = null;
    if (modelName.toLowerCase().includes('xgb')) {
        targetRow = rowXGB;
    } else if (modelName.toLowerCase().includes('random')) {
        targetRow = rowRF;
    } else if (modelName.toLowerCase().includes('logistic')) {
        targetRow = rowLR;
    }

    if (targetRow) {
        targetRow.classList.add('active-row-highlight');
        const badge = targetRow.querySelector('.badge');
        if (badge) {
            badge.className = 'badge active-badge';
            badge.innerText = 'ACTIVE MODEL';
        }
    }
}

// Sliders Event Handlers
function setupSlidersListener() {
    const updateVal = (slider, span, suffix = '') => {
        span.innerText = slider.value;
        slider.addEventListener('input', () => {
            span.innerText = slider.value;
            debouncePrediction();
        });
    };

    updateVal(cpuSlider, cpuVal);
    updateVal(memorySlider, memoryVal);
    updateVal(temperatureSlider, temperatureVal);
    updateVal(uptimeSlider, uptimeVal);
    updateVal(errorsSlider, errorsVal);
    updateVal(lossSlider, lossVal);
    updateVal(bandwidthSlider, bandwidthVal);
    updateVal(logSlider, logVal);

    // Watch Radio switches
    document.querySelectorAll('input[name="Device_Type"]').forEach(radio => {
        radio.addEventListener('change', () => {
            // Remove 'selected' styling from presets when customized manually
            document.querySelectorAll('.preset-btn').forEach(btn => btn.classList.remove('selected'));
            debouncePrediction();
        });
    });
}

// Presets Selection
function setupPresetsListener() {
    document.querySelectorAll('.preset-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            // Toggle Selected State
            document.querySelectorAll('.preset-btn').forEach(b => b.classList.remove('selected'));
            btn.classList.add('selected');
            
            const presetKey = btn.getAttribute('data-preset');
            const presetData = PRESETS[presetKey];
            
            if (presetData) {
                // Populate Inputs
                if (presetData.Device_Type === "Router") {
                    document.getElementById('type-router').checked = true;
                } else {
                    document.getElementById('type-switch').checked = true;
                }
                
                cpuSlider.value = presetData.CPU_Usage;
                cpuVal.innerText = presetData.CPU_Usage;
                
                memorySlider.value = presetData.Memory_Usage;
                memoryVal.innerText = presetData.Memory_Usage;
                
                temperatureSlider.value = presetData.Temperature;
                temperatureVal.innerText = presetData.Temperature;
                
                uptimeSlider.value = presetData.Uptime;
                uptimeVal.innerText = presetData.Uptime;
                
                errorsSlider.value = presetData.Interface_Errors;
                errorsVal.innerText = presetData.Interface_Errors;
                
                lossSlider.value = presetData.Packet_Loss;
                lossVal.innerText = presetData.Packet_Loss;
                
                bandwidthSlider.value = presetData.Bandwidth_Usage;
                bandwidthVal.innerText = presetData.Bandwidth_Usage;
                
                logSlider.value = presetData.Log_Errors;
                logVal.innerText = presetData.Log_Errors;
                
                // Fire Inference immediately
                triggerPrediction();
            }
        });
    });
}

// Debouncing for smooth slider updates
function debouncePrediction() {
    // Unselect preset since user modified sliders manually
    document.querySelectorAll('.preset-btn').forEach(btn => btn.classList.remove('selected'));
    
    if (predictionTimeout) {
        clearTimeout(predictionTimeout);
    }
    predictionTimeout = setTimeout(triggerPrediction, 250);
}

// Perform AJAX Prediction
async function triggerPrediction() {
    const formData = new FormData(telemetryForm);
    const payload = {};
    
    // Add text fields
    payload["Device_Type"] = document.querySelector('input[name="Device_Type"]:checked').value;
    
    // Add numbers
    payload["CPU_Usage"] = parseFloat(cpuSlider.value);
    payload["Memory_Usage"] = parseFloat(memorySlider.value);
    payload["Temperature"] = parseFloat(temperatureSlider.value);
    payload["Uptime"] = parseFloat(uptimeSlider.value);
    payload["Interface_Errors"] = parseInt(errorsSlider.value);
    payload["Packet_Loss"] = parseFloat(lossSlider.value);
    payload["Bandwidth_Usage"] = parseFloat(bandwidthSlider.value);
    payload["Log_Errors"] = parseInt(logSlider.value);

    try {
        const response = await fetch('/api/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(payload)
        });
        
        const data = await response.json();
        
        if (data.success) {
            updateUIWithResults(data);
        } else if (data.error) {
            resultStatusSummary.innerText = `Error: ${data.error}`;
            resultPercent.innerText = "N/A";
        }
    } catch (err) {
        console.error('Diagnostics connection error:', err);
        resultStatusSummary.innerText = "Connection lost with NetGuard ML backend.";
    }
}

// Update DOM elements on prediction return
function updateUIWithResults(data) {
    const prob = data.probability;
    const pct = (prob * 100).toFixed(2);
    
    // Update percentage label
    resultPercent.innerText = `${pct}%`;
    
    // Update SVG Circular Gauge
    // Circumference C = 2 * PI * r = 251.2
    const circumference = 251.2;
    const offset = circumference - (prob * circumference);
    gaugeFillCircle.style.strokeDashoffset = offset;
    
    // Color mapping
    let color = data.risk_color;
    gaugeFillCircle.style.stroke = color;
    
    // Apply styling to percentage text
    resultPercent.style.color = color;
    
    // Update Badge
    riskText.innerText = `${data.risk} RISK`;
    riskBadge.className = 'risk-badge'; // Reset classes
    
    if (data.risk === 'LOW') {
        riskBadge.classList.add('low-risk');
        riskIcon.className = 'fa-solid fa-circle-check';
        riskBadge.style.color = 'var(--accent-green)';
    } else if (data.risk === 'MEDIUM') {
        riskBadge.classList.add('medium-risk');
        riskIcon.className = 'fa-solid fa-triangle-exclamation';
        riskBadge.style.color = 'var(--accent-warning)';
    } else {
        riskBadge.classList.add('high-risk');
        riskIcon.className = 'fa-solid fa-radiation';
        riskBadge.style.color = 'var(--accent-critical)';
    }
    
    // Update status paragraph description
    resultStatusSummary.innerText = data.status_text;
    
    // Update Recommendations list
    advisoryList.innerHTML = ''; // clear previous
    
    data.advisory.forEach(adv => {
        const li = document.createElement('li');
        li.className = 'advisory-item';
        
        let iconClass = 'fa-solid fa-circle-info advisory-icon font-indigo';
        if (data.risk === 'MEDIUM') {
            iconClass = 'fa-solid fa-circle-exclamation advisory-icon font-warning';
        } else if (data.risk === 'HIGH') {
            iconClass = 'fa-solid fa-triangle-exclamation advisory-icon font-critical';
        }
        
        li.innerHTML = `
            <i class="${iconClass}"></i>
            <span class="advisory-text">${adv}</span>
        `;
        advisoryList.appendChild(li);
    });
}

// Lightbox modal functions
function openLightbox(src, title) {
    lightbox.style.display = 'flex';
    lightboxImg.src = src;
    lightboxCaption.innerText = title;
}

function closeLightbox() {
    lightbox.style.display = 'none';
}
