/**
 * app_additions.js
 * ----------------
 * Intelligence Hub additions for NetGuard NOC dashboard.
 * Depends on app.js being loaded first (shares slider DOM references).
 *
 * Exports (assigned to window):
 *   initDashboardAdditions(deviceId)
 */

// ─── State ────────────────────────────────────────────────────────────────────
let _currentDeviceId = 'manual';
let _baselineResult   = null;   // last /api/predict result (for what-if diff)
let _whatifTimeout    = null;

// ─── Init ─────────────────────────────────────────────────────────────────────
window.initDashboardAdditions = function (deviceId) {
    _currentDeviceId = deviceId || 'manual';
    loadHistory(_currentDeviceId);
};

// Called by app.js's updateUIWithResults to hook into every real prediction
window._onPredictionResult = function (data) {
    _baselineResult = data;
    renderHealthScore(data);
    renderShapCauses(data.shap_causes || []);
    renderRiskWindow(data.risk_window || '');
    renderRecommendedActions(data.recommended_actions || []);
};

// ─── Health Score ─────────────────────────────────────────────────────────────
function renderHealthScore(data) {
    const scoreEl = document.getElementById('health-score-value');
    const arcEl   = document.getElementById('health-score-arc');
    const labelEl = document.getElementById('health-score-label');
    if (!scoreEl || !arcEl) return;

    const score = data.health_score ?? 0;
    scoreEl.textContent = score.toFixed(1);

    // SVG arc: full circle circumference = 2π×45 ≈ 282.7
    const circumference = 282.7;
    const fraction = Math.max(0, Math.min(score / 100, 1));
    arcEl.style.strokeDashoffset = circumference * (1 - fraction);

    // Colour gradient: green (100) → amber (50) → red (0)
    let color;
    if (score >= 70)      color = '#00e676';
    else if (score >= 40) color = '#ffb300';
    else                  color = '#ff1744';
    arcEl.style.stroke = color;
    scoreEl.style.color = color;

    if (labelEl) {
        if (score >= 70)      labelEl.textContent = 'HEALTHY';
        else if (score >= 40) labelEl.textContent = 'DEGRADED';
        else                  labelEl.textContent = 'CRITICAL';
        labelEl.style.color = color;
    }
}

// ─── Risk Window ──────────────────────────────────────────────────────────────
function renderRiskWindow(text) {
    const el = document.getElementById('risk-window-text');
    if (el) el.textContent = text;
}

// ─── SHAP / Cause Bars ────────────────────────────────────────────────────────
function renderShapCauses(causes) {
    const container = document.getElementById('shap-bars-container');
    if (!container) return;

    container.innerHTML = '';

    if (!causes || causes.length === 0) {
        container.innerHTML = '<p class="shap-empty">No cause data available.</p>';
        return;
    }

    // Max absolute contribution for relative bar widths
    const maxAbs = Math.max(...causes.map(c => Math.abs(c.contribution)), 0.001);

    causes.forEach((cause, idx) => {
        const pct = Math.round((Math.abs(cause.contribution) / maxAbs) * 100);
        const isRisk = cause.direction !== 'decreases_risk';
        const barColor = isRisk ? '#ff1744' : '#00e676';
        const dirIcon  = isRisk ? '↑' : '↓';

        const row = document.createElement('div');
        row.className = 'shap-row';
        row.style.animationDelay = `${idx * 80}ms`;
        row.innerHTML = `
            <div class="shap-label">
                <span class="shap-dir" style="color:${barColor}">${dirIcon}</span>
                <span>${cause.feature}</span>
            </div>
            <div class="shap-bar-track">
                <div class="shap-bar-fill"
                     style="width:0%; background:${barColor};"
                     data-target="${pct}">
                </div>
            </div>
            <span class="shap-pct" style="color:${barColor}">
                ${Math.abs(cause.contribution).toFixed(3)}
            </span>
        `;
        container.appendChild(row);
    });

    // Animate bars after paint
    requestAnimationFrame(() => {
        container.querySelectorAll('.shap-bar-fill').forEach(bar => {
            bar.style.transition = 'width 0.6s cubic-bezier(0.4,0,0.2,1)';
            bar.style.width = bar.dataset.target + '%';
        });
    });
}

// ─── Recommended Actions (Intelligence tab) ───────────────────────────────────
function renderRecommendedActions(actions) {
    const list = document.getElementById('intel-actions-list');
    if (!list) return;
    list.innerHTML = '';
    actions.forEach(action => {
        const li = document.createElement('li');
        li.className = 'intel-action-item';
        li.innerHTML = `<i class="fa-solid fa-arrow-right-long"></i><span>${action}</span>`;
        list.appendChild(li);
    });
}

// ─── Prediction History ───────────────────────────────────────────────────────
async function loadHistory(deviceId) {
    const container = document.getElementById('history-chart-container');
    const emptyEl   = document.getElementById('history-empty-msg');
    if (!container) return;

    try {
        const res  = await fetch(`/api/history/${encodeURIComponent(deviceId)}?limit=30`);
        const data = await res.json();

        if (!data.success || !data.records || data.records.length === 0) {
            if (emptyEl) emptyEl.style.display = 'block';
            container.innerHTML = '';
            return;
        }
        if (emptyEl) emptyEl.style.display = 'none';

        const records = [...data.records].reverse();  // oldest → newest
        renderHistorySparkline(container, records);
        renderHistoryTable(records);
    } catch (e) {
        console.warn('History load failed:', e);
    }
}

function renderHistorySparkline(container, records) {
    const W = container.clientWidth || 600;
    const H = 120;
    const PAD = { top: 12, right: 16, bottom: 28, left: 40 };
    const innerW = W - PAD.left - PAD.right;
    const innerH = H - PAD.top - PAD.bottom;

    const probs = records.map(r => r.probability);
    const n     = probs.length;
    const xStep = n > 1 ? innerW / (n - 1) : innerW;

    const toX = i => PAD.left + (n > 1 ? i * xStep : innerW / 2);
    const toY = v => PAD.top + innerH * (1 - v);   // v in [0,1]

    // Build SVG polyline points
    const points = probs.map((v, i) => `${toX(i)},${toY(v)}`).join(' ');

    // Area fill path
    const firstX = toX(0), lastX = toX(n - 1), baseY = PAD.top + innerH;
    const areaPath = `M${firstX},${baseY} L${firstX},${toY(probs[0])} `
        + probs.slice(1).map((v, i) => `L${toX(i + 1)},${toY(v)}`).join(' ')
        + ` L${lastX},${baseY} Z`;

    // Y-axis labels
    const yLabels = [0, 0.25, 0.5, 0.75, 1.0].map(v => `
        <text x="${PAD.left - 6}" y="${toY(v) + 4}" text-anchor="end"
              class="sparkline-axis-label">${Math.round(v * 100)}%</text>
        <line x1="${PAD.left}" y1="${toY(v)}" x2="${PAD.left + innerW}" y2="${toY(v)}"
              class="sparkline-grid-line"/>
    `).join('');

    // Dots for each point
    const dots = probs.map((v, i) => {
        const risk = records[i].risk || 'LOW';
        const dc   = risk === 'HIGH' ? '#ff1744' : risk === 'MEDIUM' ? '#ffb300' : '#00e676';
        return `<circle cx="${toX(i)}" cy="${toY(v)}" r="3.5" fill="${dc}" class="sparkline-dot"
                        data-idx="${i}" data-prob="${(v * 100).toFixed(1)}" data-risk="${risk}"/>`;
    }).join('');

    container.innerHTML = `
    <svg viewBox="0 0 ${W} ${H}" width="100%" height="${H}" class="sparkline-svg">
        <defs>
            <linearGradient id="spark-grad" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%"   stop-color="#6366f1" stop-opacity="0.45"/>
                <stop offset="100%" stop-color="#6366f1" stop-opacity="0"/>
            </linearGradient>
        </defs>
        ${yLabels}
        <path d="${areaPath}" fill="url(#spark-grad)"/>
        <polyline points="${points}" fill="none" stroke="#6366f1" stroke-width="2"
                  stroke-linejoin="round" stroke-linecap="round"/>
        ${dots}
    </svg>`;
}

function renderHistoryTable(records) {
    const tbody = document.getElementById('history-table-body');
    if (!tbody) return;
    const recent = [...records].reverse().slice(0, 10);
    tbody.innerHTML = recent.map(r => {
        const ts   = new Date(r.timestamp).toLocaleString();
        const prob = ((r.probability || 0) * 100).toFixed(1) + '%';
        const hs   = r.health_score != null ? r.health_score.toFixed(1) : '—';
        const riskClass = r.risk === 'HIGH' ? 'font-critical'
                        : r.risk === 'MEDIUM' ? 'font-warning' : 'font-accent';
        return `<tr>
            <td>${ts}</td>
            <td>${prob}</td>
            <td>${hs}</td>
            <td class="${riskClass}">${r.risk || '—'}</td>
        </tr>`;
    }).join('');
}

// Device ID lookup button
document.addEventListener('DOMContentLoaded', () => {
    const lookupBtn = document.getElementById('history-lookup-btn');
    const deviceInput = document.getElementById('history-device-input');

    if (lookupBtn && deviceInput) {
        lookupBtn.addEventListener('click', () => {
            const id = deviceInput.value.trim();
            if (!id) return;
            _currentDeviceId = id;
            loadHistory(id);
        });
        deviceInput.addEventListener('keydown', e => {
            if (e.key === 'Enter') lookupBtn.click();
        });
    }
});

// ─── What-If Simulator ────────────────────────────────────────────────────────
function getSliderPayload() {
    const deviceTypeEl = document.querySelector('input[name="Device_Type"]:checked');
    return {
        Device_Type:      deviceTypeEl ? deviceTypeEl.value : 'Router',
        CPU_Usage:        parseFloat(document.getElementById('cpu_usage')?.value    || 50),
        Memory_Usage:     parseFloat(document.getElementById('memory_usage')?.value || 60),
        Temperature:      parseFloat(document.getElementById('temperature')?.value  || 45),
        Uptime:           parseFloat(document.getElementById('uptime')?.value       || 100),
        Interface_Errors: parseInt(document.getElementById('interface_errors')?.value || 10),
        Packet_Loss:      parseFloat(document.getElementById('packet_loss')?.value  || 0.5),
        Bandwidth_Usage:  parseFloat(document.getElementById('bandwidth_usage')?.value || 40),
        Log_Errors:       parseInt(document.getElementById('log_errors')?.value     || 2),
    };
}

async function triggerWhatIf() {
    const payload = getSliderPayload();
    try {
        const res  = await fetch('/api/whatif', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload),
        });
        const data = await res.json();
        if (data.success) renderWhatIfPanel(data);
    } catch (e) {
        console.warn('What-if failed:', e);
    }
}

function renderWhatIfPanel(whatifData) {
    const panel = document.getElementById('whatif-panel');
    if (!panel) return;
    panel.style.display = 'block';

    const baseline = _baselineResult;
    const wProb    = (whatifData.probability  * 100).toFixed(1);
    const wHealth  = (whatifData.health_score ?? 0).toFixed(1);

    // Delta vs baseline
    let deltaProb = null, deltaHealth = null;
    if (baseline) {
        deltaProb   = ((whatifData.probability  - baseline.probability)  * 100).toFixed(1);
        deltaHealth = (whatifData.health_score  - (baseline.health_score ?? 0)).toFixed(1);
    }

    const probColor   = whatifData.risk === 'HIGH' ? '#ff1744'
                      : whatifData.risk === 'MEDIUM' ? '#ffb300' : '#00e676';

    const deltaSign   = v => v > 0 ? `+${v}` : `${v}`;
    const deltaClass  = (v, higherIsBad) => {
        if (v === null) return '';
        const n = parseFloat(v);
        const bad = higherIsBad ? n > 0 : n < 0;
        return bad ? 'delta-bad' : 'delta-good';
    };

    document.getElementById('wi-prob-val').textContent = wProb + '%';
    document.getElementById('wi-prob-val').style.color = probColor;
    document.getElementById('wi-health-val').textContent = wHealth;
    document.getElementById('wi-risk-val').textContent   = whatifData.risk + ' RISK';
    document.getElementById('wi-window-val').textContent = whatifData.risk_window || '—';

    const dpEl = document.getElementById('wi-delta-prob');
    const dhEl = document.getElementById('wi-delta-health');
    if (dpEl && deltaProb !== null) {
        dpEl.textContent  = deltaSign(deltaProb) + '%';
        dpEl.className    = 'wi-delta ' + deltaClass(deltaProb, true);
    }
    if (dhEl && deltaHealth !== null) {
        dhEl.textContent  = deltaSign(deltaHealth);
        dhEl.className    = 'wi-delta ' + deltaClass(deltaHealth, false);
    }

    renderShapCauses(whatifData.shap_causes || []);
}

// Wire What-If mode toggle
document.addEventListener('DOMContentLoaded', () => {
    const toggleBtn = document.getElementById('whatif-toggle-btn');
    const wiPanel   = document.getElementById('whatif-panel');
    let   wiActive  = false;

    if (toggleBtn) {
        toggleBtn.addEventListener('click', () => {
            wiActive = !wiActive;
            toggleBtn.classList.toggle('active', wiActive);
            toggleBtn.innerHTML = wiActive
                ? '<i class="fa-solid fa-flask-vial"></i> Exit What-If'
                : '<i class="fa-solid fa-flask-vial"></i> What-If Simulator';

            if (wiPanel) wiPanel.style.display = wiActive ? 'block' : 'none';

            if (wiActive) {
                // Intercept slider input events to call /api/whatif
                document.querySelectorAll('input[type="range"], input[name="Device_Type"]')
                    .forEach(el => el.addEventListener('input', scheduleWhatIf));
                triggerWhatIf();
            } else {
                document.querySelectorAll('input[type="range"], input[name="Device_Type"]')
                    .forEach(el => el.removeEventListener('input', scheduleWhatIf));
                if (wiPanel) wiPanel.style.display = 'none';
            }
        });
    }
});

function scheduleWhatIf() {
    clearTimeout(_whatifTimeout);
    _whatifTimeout = setTimeout(triggerWhatIf, 200);
}

// ─── Hook into app.js prediction pipeline ────────────────────────────────────
// Patch updateUIWithResults to also call our hook
(function patchUpdateUI() {
    const _original = window.updateUIWithResults;
    if (typeof _original === 'function') {
        window.updateUIWithResults = function (data) {
            _original(data);
            if (typeof window._onPredictionResult === 'function') {
                window._onPredictionResult(data);
            }
        };
    } else {
        // app.js not yet loaded — retry after a short delay
        setTimeout(patchUpdateUI, 200);
    }
})();

// Auto-init with 'manual' device on page load
document.addEventListener('DOMContentLoaded', () => {
    window.initDashboardAdditions('manual');
});
