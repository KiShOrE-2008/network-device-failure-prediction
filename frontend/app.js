/**
 * NetGuard NOC — Mission Control Command Center Application Controller
 */

let currentView = 'overview';
let cachedPredictions = [];
let cachedDevices = [];
let cachedTopology = null;

document.addEventListener('DOMContentLoaded', () => {
  initClock();
  setupCommandPaletteShortcuts();
  switchView('overview');
});

// Live UTC Header Clock
function initClock() {
  function updateTime() {
    const now = new Date();
    const timeStr = now.toISOString().replace('T', ' ').substring(11, 19) + ' UTC';
    const el = document.getElementById('header-time-display');
    if (el) el.innerText = timeStr;
  }
  updateTime();
  setInterval(updateTime, 1000);
}

// Single-Page View Router
function switchView(viewId) {
  currentView = viewId;

  // Update Sidebar Active Class
  document.querySelectorAll('.nav-item').forEach(item => {
    item.classList.remove('active');
    if (item.getAttribute('onclick') && item.getAttribute('onclick').includes(viewId)) {
      item.classList.add('active');
    }
  });

  // Toggle View Display
  document.querySelectorAll('.content-view').forEach(view => {
    view.classList.remove('active');
  });

  const targetView = document.getElementById(`view-${viewId}`);
  if (targetView) {
    targetView.classList.add('active');
  }

  // Load View Data
  switch (viewId) {
    case 'overview':
      loadOverviewData();
      break;
    case 'topology':
      loadTopologyData();
      break;
    case 'devices':
      loadDevicesData();
      break;
    case 'incidents':
      loadIncidentsData();
      break;
    case 'predictions':
      loadPredictionsData();
      break;
    case 'anomalies':
      loadAnomaliesData();
      break;
    case 'diagnostics':
      loadDiagnosticsData();
      break;
    case 'analytics':
      loadAnalyticsData();
      break;
    case 'discovery':
      loadDiscoveryData();
      break;
    case 'models':
      break;
  }
}

// -----------------------------------------------------------------------------
// 1. Overview Dashboard Loader
// -----------------------------------------------------------------------------
async function loadOverviewData() {
  try {
    const res = await fetch('/api/fleet/predictions');
    const data = await res.json();

    if (data.success) {
      cachedPredictions = data.predictions;
      const fleet = data.fleet;

      // Update KPIs
      document.getElementById('kpi-total').innerText = fleet.total_devices;
      document.getElementById('kpi-healthy').innerText = fleet.risk_summary.low;
      document.getElementById('kpi-healthy-pct').innerText = `${((fleet.risk_summary.low / fleet.total_devices) * 100).toFixed(1)}% operational`;
      document.getElementById('kpi-at-risk').innerText = fleet.risk_summary.medium + fleet.risk_summary.high;
      document.getElementById('kpi-critical').innerText = fleet.risk_summary.critical;

      // Update Network Health Index
      const healthScore = fleet.network_health_score.toFixed(1);
      document.getElementById('net-health-value').innerHTML = `${healthScore} <span style="font-size:16px; color:var(--text-muted)">/ 100</span>`;
      document.getElementById('net-health-fill').style.width = `${healthScore}%`;

      const healthPill = document.getElementById('net-health-status');
      if (healthScore >= 85) {
        healthPill.innerText = 'OPERATIONAL';
        healthPill.style.color = 'var(--color-low)';
        healthPill.style.borderColor = 'rgba(34,197,94,0.3)';
      } else if (healthScore >= 70) {
        healthPill.innerText = 'DEGRADED';
        healthPill.style.color = 'var(--color-medium)';
        healthPill.style.borderColor = 'rgba(234,179,8,0.3)';
      } else {
        healthPill.innerText = 'CRITICAL THREAT';
        healthPill.style.color = 'var(--color-critical)';
        healthPill.style.borderColor = 'rgba(239,68,68,0.3)';
      }

      // Update Stacked Risk Breakdown
      const lowPct = (fleet.risk_summary.low / fleet.total_devices) * 100;
      const medPct = (fleet.risk_summary.medium / fleet.total_devices) * 100;
      const highPct = (fleet.risk_summary.high / fleet.total_devices) * 100;
      const critPct = (fleet.risk_summary.critical / fleet.total_devices) * 100;

      const bar = document.getElementById('risk-stacked-bar');
      bar.innerHTML = `
        <div class="seg-low" style="width:${lowPct}%" title="Low Risk: ${fleet.risk_summary.low}"></div>
        <div class="seg-medium" style="width:${medPct}%" title="Medium Risk: ${fleet.risk_summary.medium}"></div>
        <div class="seg-high" style="width:${highPct}%" title="High Risk: ${fleet.risk_summary.high}"></div>
        <div class="seg-critical" style="width:${critPct}%" title="Critical Risk: ${fleet.risk_summary.critical}"></div>
      `;

      document.getElementById('count-low').innerText = fleet.risk_summary.low;
      document.getElementById('count-medium').innerText = fleet.risk_summary.medium;
      document.getElementById('count-high').innerText = fleet.risk_summary.high;
      document.getElementById('count-critical').innerText = fleet.risk_summary.critical;

      // Render Top Failure Predictions Table
      renderTopFailuresTable(data.predictions.slice(0, 10));
    }
  } catch (err) {
    console.error('Failed to load overview fleet data:', err);
  }
}

function renderTopFailuresTable(items) {
  const tbody = document.getElementById('top-failures-tbody');
  if (!tbody) return;

  tbody.innerHTML = items.map(p => `
    <tr onclick="openDeviceDrawer('${p.device_id}')">
      <td class="device-id mono-text" style="color:var(--accent-cyan); font-weight:600;">${p.device_id}</td>
      <td>${p.hostname}</td>
      <td class="ip-address mono-text" style="color:var(--text-muted);">${p.ip_address}</td>
      <td><span class="badge-risk ${p.risk}">${p.risk}</span></td>
      <td class="mono-text" style="font-weight:600;">${p.failure_probability_pct !== null && p.failure_probability_pct !== undefined ? p.failure_probability_pct + '%' : '<span style="color:var(--text-muted); font-size:11px;">N/A (ML Unavailable)</span>'}</td>
      <td><span style="font-weight:600; color:var(--text-main);">${p.predicted_failure}</span></td>

      <td class="mono-text">${p.health_score}</td>
      <td><button class="btn-noc" onclick="event.stopPropagation(); openDeviceDrawer('${p.device_id}')">Inspect →</button></td>
    </tr>
  `).join('');
}

async function triggerFleetPredict() {
  const btn = document.getElementById('btn-fleet-scan');
  const btnIcon = document.getElementById('btn-fleet-scan-icon');
  const btnText = document.getElementById('btn-fleet-scan-text');
  const banner = document.getElementById('fleet-scan-banner');

  if (btn) btn.disabled = true;
  if (btnIcon) btnIcon.innerHTML = '<span class="spinner"></span>';
  if (btnText) btnText.innerText = 'Scanning Fleet...';

  if (banner) {
    banner.style.display = 'block';
    banner.innerHTML = `
      <div style="width:100%;">
        <div style="display:flex; justify-content:space-between; align-items:center;">
          <span style="font-weight:600; font-size:12px; color:var(--accent-cyan);">⚡ FLEET PREDICTION SCAN IN PROGRESS</span>
          <span class="mono-text" style="font-size:11px; color:var(--text-muted);" id="fleet-scan-status-text">Analyzing 500 devices...</span>
        </div>
        <div class="scan-progress-bar">
          <div class="scan-progress-fill" id="fleet-scan-progress" style="width: 25%"></div>
        </div>
      </div>
    `;
  }

  try {
    const prog = document.getElementById('fleet-scan-progress');
    const txt = document.getElementById('fleet-scan-status-text');

    setTimeout(() => {
      if (prog) prog.style.width = '65%';
      if (txt) txt.innerText = 'Running RandomForest & Isolation Forest Inference...';
    }, 400);

    const res = await fetch('/api/fleet/predict', { method: 'POST' });
    const data = await res.json();

    if (prog) prog.style.width = '100%';
    if (txt) txt.innerText = 'Scan Complete! Updating NOC metrics...';

    setTimeout(() => {
      if (banner) banner.style.display = 'none';
      if (btn) btn.disabled = false;
      if (btnIcon) btnIcon.innerText = '⚡';
      if (btnText) btnText.innerText = 'Run Fleet Scan';
      if (data.success) {
        loadOverviewData();
      }
    }, 500);
  } catch (err) {
    console.error('Fleet prediction scan trigger failed:', err);
    if (banner) banner.style.display = 'none';
    if (btn) btn.disabled = false;
    if (btnIcon) btnIcon.innerText = '⚡';
    if (btnText) btnText.innerText = 'Run Fleet Scan';
  }
}


// -----------------------------------------------------------------------------
// 2. Topology View Loader & SVG Graph
// -----------------------------------------------------------------------------
async function loadTopologyData() {
  const container = document.getElementById('topology-svg-container');
  if (container) {
    container.innerHTML = `
      <div style="display:flex; flex-direction:column; align-items:center; justify-content:center; height:100%; color:var(--accent-cyan);">
        <span class="spinner" style="width:24px; height:24px; margin-bottom:12px;"></span>
        <div style="font-weight:600; font-size:13px;">Building Topology Hierarchy Graph...</div>
        <div style="font-size:11px; color:var(--text-muted); margin-top:4px;">Mapping 500 parent-child uplink nodes...</div>
      </div>
    `;
  }
  try {
    const res = await fetch('/api/topology');
    const data = await res.json();
    if (data.success) {
      if (container) container.innerHTML = '<svg id="topology-svg"></svg>';
      cachedTopology = data;
      renderTopologySVG(data.nodes, data.links);
    }
  } catch (err) {
    console.error('Failed to load topology:', err);
  }
}


function renderTopologySVG(nodes, links) {
  const svg = document.getElementById('topology-svg');
  if (!svg) return;

  const width = svg.clientWidth || 900;
  const height = 520;
  svg.setAttribute('viewBox', `0 0 ${width} ${height}`);
  svg.innerHTML = '';

  const nodeMap = {};
  const total = nodes.length;

  // Grid layout for 500 topology nodes
  const cols = Math.ceil(Math.sqrt(total * 1.8));
  const rows = Math.ceil(total / cols);
  const padding = 20;

  nodes.forEach((node, i) => {
    const r = Math.floor(i / cols);
    const c = i % cols;
    const x = padding + (c / cols) * (width - 2 * padding) + 20;
    const y = padding + (r / rows) * (height - 2 * padding) + 20;
    nodeMap[node.id] = { x, y, ...node };
  });

  // Render Links
  links.forEach(link => {
    const s = nodeMap[link.source];
    const t = nodeMap[link.target];
    if (s && t) {
      const line = document.createElementNS('http://www.w3.org/2000/svg', 'line');
      line.setAttribute('x1', s.x);
      line.setAttribute('y1', s.y);
      line.setAttribute('x2', t.x);
      line.setAttribute('y2', t.y);
      line.setAttribute('class', 'topo-link');
      svg.appendChild(line);
    }
  });

  // Render Nodes
  Object.values(nodeMap).forEach(node => {
    const g = document.createElementNS('http://www.w3.org/2000/svg', 'g');
    g.setAttribute('class', 'topo-node');
    g.setAttribute('onclick', `openDeviceDrawer('${node.id}')`);

    const circle = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
    circle.setAttribute('cx', node.x);
    circle.setAttribute('cy', node.y);
    circle.setAttribute('r', node.risk === 'CRITICAL' ? 7 : (node.risk === 'HIGH' ? 6 : 4));

    let fill = 'var(--color-low)';
    if (node.risk === 'CRITICAL') fill = 'var(--color-critical)';
    else if (node.risk === 'HIGH') fill = 'var(--color-high)';
    else if (node.risk === 'MEDIUM') fill = 'var(--color-medium)';

    circle.setAttribute('fill', fill);
    g.appendChild(circle);

    svg.appendChild(g);
  });
}

// -----------------------------------------------------------------------------
// 3. Devices Inventory View Loader
// -----------------------------------------------------------------------------
async function loadDevicesData() {
  try {
    const res = await fetch('/api/devices');
    const data = await res.json();
    if (data.success) {
      cachedDevices = data.devices;
      renderDevicesTable(cachedDevices);
    }
  } catch (err) {
    console.error('Failed to load devices list:', err);
  }
}

function renderDevicesTable(devices) {
  const tbody = document.getElementById('devices-list-tbody');
  if (!tbody) return;

  tbody.innerHTML = devices.map(d => `
    <tr onclick="openDeviceDrawer('${d.device_id}')">
      <td class="device-id mono-text" style="color:var(--accent-cyan); font-weight:600;">${d.device_id}</td>
      <td>${d.hostname}</td>
      <td class="ip-address mono-text">${d.ip_address}</td>
      <td>${d.device_type}</td>
      <td>${d.vendor}</td>
      <td>${d.location}</td>
      <td><span class="badge-risk ${d.risk}">${d.risk}</span></td>
      <td class="mono-text">${d.failure_probability !== null && d.failure_probability !== undefined ? (d.failure_probability * 100).toFixed(1) + '%' : '<span style="color:var(--text-muted); font-size:11px;">N/A (ML Unavailable)</span>'}</td>
      <td class="mono-text">${d.health_score}</td>

    </tr>
  `).join('');
}

function filterDevicesTable() {
  const q = document.getElementById('device-search-input').value.toLowerCase();
  const filtered = cachedDevices.filter(d =>
    d.device_id.toLowerCase().includes(q) ||
    d.hostname.toLowerCase().includes(q) ||
    d.ip_address.toLowerCase().includes(q) ||
    d.vendor.toLowerCase().includes(q) ||
    d.location.toLowerCase().includes(q)
  );
  renderDevicesTable(filtered);
}

// -----------------------------------------------------------------------------
// 4. Incidents Alert Management View Loader
// -----------------------------------------------------------------------------
async function loadIncidentsData() {
  try {
    const res = await fetch('/api/alerts');
    const data = await res.json();
    if (data.success) {
      renderIncidentsTable(data.alerts);
    }
  } catch (err) {
    console.error('Failed to load alerts:', err);
  }
}

function renderIncidentsTable(alerts) {
  const tbody = document.getElementById('incidents-tbody');
  if (!tbody) return;

  tbody.innerHTML = alerts.map(a => `
    <tr>
      <td class="mono-text" style="color:var(--accent-cyan)">INC-${a.id.toString().padStart(5, '0')}</td>
      <td class="timestamp mono-text" style="font-size:11px;">${a.timestamp.substring(0, 19).replace('T', ' ')}</td>
      <td class="device-id mono-text" onclick="openDeviceDrawer('${a.device_id}')" style="cursor:pointer; text-decoration:underline;">${a.device_id}</td>
      <td><span class="badge-risk ${a.severity}">${a.severity}</span></td>
      <td>
        <div style="font-weight:600; color:var(--text-main);">${a.title}</div>
        <div style="font-size:11px; color:var(--text-muted);">${a.message}</div>
      </td>
      <td><span class="mono-text" style="font-size:11px; font-weight:700; color:${a.status === 'ACTIVE' ? 'var(--color-critical)' : 'var(--color-medium)'}">${a.status}</span></td>
      <td>
        ${a.status === 'ACTIVE' ? `<button class="btn-noc" onclick="acknowledgeAlert(${a.id})">Acknowledge</button>` : ''}
        ${a.status !== 'RESOLVED' ? `<button class="btn-noc primary" onclick="resolveAlert(${a.id})">Resolve</button>` : ''}
      </td>
    </tr>
  `).join('');
}

async function acknowledgeAlert(alertId) {
  try {
    await fetch(`/api/alerts/${alertId}/acknowledge`, { method: 'POST' });
    loadIncidentsData();
  } catch (err) {
    console.error('Acknowledge alert error:', err);
  }
}

async function resolveAlert(alertId) {
  try {
    await fetch(`/api/alerts/${alertId}/resolve`, { method: 'POST' });
    loadIncidentsData();
  } catch (err) {
    console.error('Resolve alert error:', err);
  }
}

// -----------------------------------------------------------------------------
// 5. Predictions Forecast View Loader
// -----------------------------------------------------------------------------
function loadPredictionsData() {
  const tbody = document.getElementById('predictions-tbody');
  if (!tbody) return;
  tbody.innerHTML = cachedPredictions.map(p => `
    <tr onclick="openDeviceDrawer('${p.device_id}')">
      <td class="device-id mono-text" style="color:var(--accent-cyan);">${p.device_id}</td>
      <td>${p.hostname}</td>
      <td><span class="badge-risk ${p.risk}">${p.risk}</span></td>
      <td class="mono-text" style="font-weight:600;">${p.failure_probability_pct !== null && p.failure_probability_pct !== undefined ? p.failure_probability_pct + '%' : '<span style="color:var(--text-muted); font-size:11px;">N/A (ML Unavailable)</span>'}</td>
      <td>${p.predicted_failure}</td>
      <td class="mono-text">${p.diagnostic_confidence}%</td>
    </tr>
  `).join('');
}

// -----------------------------------------------------------------------------
// 6. Anomalies Detector View Loader
// -----------------------------------------------------------------------------
function loadAnomaliesData() {
  const tbody = document.getElementById('anomalies-tbody');
  if (!tbody) return;

  const anomalous = cachedPredictions.filter(p => p.anomaly_score > 40.0 || p.risk !== 'LOW');
  tbody.innerHTML = anomalous.map(p => `
    <tr onclick="openDeviceDrawer('${p.device_id}')">
      <td class="device-id mono-text" style="color:var(--accent-cyan);">${p.device_id}</td>
      <td>${p.hostname}</td>
      <td class="mono-text" style="font-weight:600; color:var(--accent-cyan);">${p.anomaly_score}%</td>
      <td class="mono-text">${p.failure_probability_pct !== null && p.failure_probability_pct !== undefined ? p.failure_probability_pct + '%' : '<span style="color:var(--text-muted); font-size:11px;">N/A</span>'}</td>
      <td><span class="badge-risk ${p.risk}">${p.risk}</span></td>
    </tr>
  `).join('');
}


// -----------------------------------------------------------------------------
// 7. Fleet Analytics View Loader
// -----------------------------------------------------------------------------

/** Renders a proportional horizontal-bar row list inside a container element. */
function _renderBarRows(container, entries, colorFn) {
  if (!container) return;
  const total = entries.reduce((s, [, v]) => s + (v.count || v), 0) || 1;
  container.innerHTML = entries.map(([label, v]) => {
    const count = v.count !== undefined ? v.count : v;
    const pct   = v.pct   !== undefined ? v.pct   : +((count / total) * 100).toFixed(1);
    const health = v.avg_health !== undefined
      ? `<span class="mono-text" style="font-size:11px; color:var(--text-muted); margin-left:8px;">⬆ ${v.avg_health}</span>`
      : '';
    const color = colorFn ? colorFn(label) : 'var(--accent-cyan)';
    return `
      <div style="margin-bottom:10px;">
        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:4px;">
          <span style="font-size:12px; font-weight:600; color:var(--text-main);">${label}${health}</span>
          <span class="mono-text" style="font-size:12px; color:${color}; font-weight:700;">${count} <span style="font-size:10px; color:var(--text-muted);">(${pct}%)</span></span>
        </div>
        <div style="background:var(--border-color); border-radius:4px; height:6px; overflow:hidden;">
          <div style="width:${pct}%; background:${color}; height:100%; border-radius:4px; transition:width 0.6s ease;"></div>
        </div>
      </div>`;
  }).join('');
}

const FAILURE_MODE_COLORS = {
  THERMAL:    '#ef4444',
  HARDWARE:   '#f97316',
  INTERFACE:  '#eab308',
  MEMORY:     '#a855f7',
  CONGESTION: '#38bdf8',
  NONE:       '#22c55e'
};

const VENDOR_COLORS = {
  Cisco:   '#3b82f6',
  Juniper: '#22c55e',
  Arista:  '#f97316',
};

const LOCATION_COLORS = ['#38bdf8', '#a855f7', '#22c55e', '#f97316', '#eab308'];

async function loadAnalyticsData() {
  try {
    const res = await fetch('/api/fleet/stats');
    const data = await res.json();
    if (!data.success) return;

    // ── KPI Strip ────────────────────────────────────────────────────────────
    const kpiTotal  = document.getElementById('analytics-kpi-total');
    const kpiHealth = document.getElementById('analytics-kpi-health');
    const kpiModes  = document.getElementById('analytics-kpi-modes');
    const kpiProb   = document.getElementById('analytics-kpi-prob');

    if (kpiTotal)  kpiTotal.textContent  = data.total_devices ?? '—';
    if (kpiHealth) kpiHealth.textContent = (data.network_health_score ?? '—') + (data.network_health_score != null ? '' : '');
    if (kpiModes)  kpiModes.textContent  = Object.keys(data.failure_mode_breakdown || {}).length;
    if (kpiProb)   kpiProb.textContent   = (data.average_failure_probability ?? 0).toFixed(1) + '%';

    // ── Failure Mode Distribution ─────────────────────────────────────────────
    const modeBox = document.getElementById('analytics-failure-types');
    if (modeBox) {
      const modeEntries = Object.entries(data.failure_mode_breakdown || {})
        .sort((a, b) => b[1] - a[1]);
      if (modeEntries.length === 0) {
        modeBox.innerHTML = '<p style="color:var(--text-muted); font-size:12px; padding:8px 0;">No active failure modes detected.</p>';
      } else {
        _renderBarRows(modeBox, modeEntries, (lbl) => FAILURE_MODE_COLORS[lbl] || 'var(--accent-cyan)');
      }
    }

    // ── Device Type Distribution ──────────────────────────────────────────────
    const devTypeBox = document.getElementById('analytics-device-types');
    if (devTypeBox) {
      const dtEntries = Object.entries(data.device_type_breakdown || {})
        .sort((a, b) => b[1].count - a[1].count);
      _renderBarRows(devTypeBox, dtEntries, () => 'var(--accent-blue)');
    }

    // ── Vendor Fleet Health ───────────────────────────────────────────────────
    const vendorBox = document.getElementById('analytics-vendor-health');
    if (vendorBox) {
      const vEntries = Object.entries(data.vendor_breakdown || {})
        .sort((a, b) => b[1].count - a[1].count);
      _renderBarRows(vendorBox, vEntries, (lbl) => VENDOR_COLORS[lbl] || 'var(--accent-cyan)');
    }

    // ── Location Overview ─────────────────────────────────────────────────────
    const locBox = document.getElementById('analytics-location-overview');
    if (locBox) {
      const locEntries = Object.entries(data.location_breakdown || {})
        .sort((a, b) => b[1].count - a[1].count);
      _renderBarRows(locBox, locEntries, (_, idx) => LOCATION_COLORS[idx % LOCATION_COLORS.length]);
    }

  } catch (err) {
    console.error('Failed to load analytics stats:', err);
  }
}


// -----------------------------------------------------------------------------
// 8. Discovery Subnet Scanner Loader
// -----------------------------------------------------------------------------
async function loadDiscoveryData() {
  try {
    const res = await fetch('/api/discovery/nodes');
    const data = await res.json();
    if (data.success) {
      renderDiscoveryTable(data.nodes);
    }
  } catch (err) {
    console.error('Failed to load discovered nodes:', err);
  }
}

async function runDiscoveryScan() {
  const cidr = document.getElementById('discovery-cidr-input').value;
  const mode = document.getElementById('discovery-mode-select').value;
  const btn = document.getElementById('btn-discovery-scan');

  if (btn) {
    btn.disabled = true;
    btn.innerHTML = '<span class="spinner"></span> Scanning Subnet...';
  }

  const tbody = document.getElementById('discovery-tbody');
  if (tbody) {
    tbody.innerHTML = `
      <tr>
        <td colspan="8" style="text-align:center; padding:30px;">
          <span class="spinner" style="width:20px; height:20px; margin-bottom:8px;"></span>
          <div style="font-weight:600; color:var(--accent-cyan); font-size:13px;">Probing CIDR Subnet ${cidr} (Mode: ${mode})...</div>
          <div style="font-size:11px; color:var(--text-muted); margin-top:4px;">Scanning reachability and querying SNMP/REST API object identifiers...</div>
        </td>
      </tr>
    `;
  }

  try {
    const res = await fetch('/api/discovery/scan', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ cidr, mode })
    });
    const data = await res.json();

    if (btn) {
      btn.disabled = false;
      btn.innerHTML = '🔍 Start Discovery Scan';
    }

    if (data.success) {
      renderDiscoveryTable(data.devices);
    }
  } catch (err) {
    console.error('Discovery scan error:', err);
    if (btn) {
      btn.disabled = false;
      btn.innerHTML = '🔍 Start Discovery Scan';
    }
  }
}


function renderDiscoveryTable(nodes) {
  const tbody = document.getElementById('discovery-tbody');
  if (!tbody) return;

  tbody.innerHTML = nodes.map(n => `
    <tr>
      <td class="device-id mono-text" style="color:var(--accent-cyan);">${n.device_id}</td>
      <td>${n.hostname}</td>
      <td class="ip-address mono-text">${n.ip_address}</td>
      <td>${n.device_type}</td>
      <td>${n.vendor}</td>
      <td>${n.model}</td>
      <td class="mono-text" style="font-size:11px;">${n.discovery_protocol}</td>
      <td><span class="mono-text" style="color:var(--color-low); font-weight:700;">${n.status}</span></td>
    </tr>
  `).join('');
}

// -----------------------------------------------------------------------------
// 9. Slide-out Device Investigation Drawer
// -----------------------------------------------------------------------------
async function openDeviceDrawer(deviceId) {
  const drawer = document.getElementById('device-drawer');
  const body = document.getElementById('drawer-body');
  if (!drawer || !body) return;

  document.getElementById('drawer-device-id').innerText = deviceId;
  drawer.classList.add('open');
  body.innerHTML = '<div style="padding:20px; text-align:center; color:var(--text-muted)">Loading device deep-dive telemetry...</div>';

  try {
    const res = await fetch(`/api/devices/${deviceId}`);
    const data = await res.json();
    if (data.success) {
      const dev = data.device;
      const t = dev.telemetry || {};

      body.innerHTML = `
        <div style="margin-bottom:20px;">
          <span class="badge-risk ${dev.risk}" style="font-size:13px; padding:4px 12px;">${dev.risk} RISK</span>
          <div style="font-size:28px; font-weight:700; color:#fff; margin-top:8px;" class="mono-text">${dev.failure_probability_pct !== null && dev.failure_probability_pct !== undefined ? dev.failure_probability_pct + '%' : '<span style="font-size:18px; color:var(--text-muted);">N/A (ML Model Unavailable)</span>'}</div>
          <div style="font-size:11px; color:var(--text-muted)">Failure Probability Next 12 Hours</div>
        </div>


        <div class="noc-card" style="margin-bottom:16px;">
          <div class="card-title">Device Telemetry Baseline</div>
          <div style="display:grid; grid-template-columns:1fr 1fr; gap:12px; font-size:12px;">
            <div>CPU Usage: <strong class="mono-text">${t.cpu}%</strong></div>
            <div>Memory Usage: <strong class="mono-text">${t.memory}%</strong></div>
            <div>Temperature: <strong class="mono-text">${t.temperature}°C</strong></div>
            <div>Interface Errors: <strong class="mono-text">${t.errors}</strong></div>
            <div>Packet Loss: <strong class="mono-text">${t.packet_loss}%</strong></div>
            <div>Bandwidth: <strong class="mono-text">${t.bandwidth}%</strong></div>
          </div>
        </div>

        <div class="noc-card" style="margin-bottom:16px;">
          <div class="card-title">Diagnostic Analysis</div>
          <div style="font-size:14px; font-weight:700; color:var(--accent-cyan); margin-bottom:6px;">MODE: ${dev.predicted_failure}</div>
          <div style="font-size:11px; color:var(--text-muted)">Diagnostic Confidence: ${dev.diagnostic_confidence}%</div>
        </div>

        <div class="noc-card">
          <div class="card-title">NOC Recommended Actions</div>
          <ul style="padding-left:16px; font-size:12px; color:var(--text-muted); display:flex; flex-direction:column; gap:6px;">
            ${(dev.recommended_actions || []).map(a => `<li>${a}</li>`).join('')}
          </ul>
        </div>
      `;
    }
  } catch (err) {
    console.error('Failed to load device details:', err);
  }
}

function closeDeviceDrawer() {
  const drawer = document.getElementById('device-drawer');
  if (drawer) drawer.classList.remove('open');
}

// -----------------------------------------------------------------------------
// 10. Command Palette Modal (Ctrl+K)
// -----------------------------------------------------------------------------
function setupCommandPaletteShortcuts() {
  document.addEventListener('keydown', e => {
    if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'k') {
      e.preventDefault();
      openCommandPalette();
    }
    if (e.key === 'Escape') {
      closeCommandPalette();
      closeDeviceDrawer();
    }
  });
}

function openCommandPalette() {
  const overlay = document.getElementById('cmd-modal-overlay');
  const input = document.getElementById('cmd-input');
  if (overlay) {
    overlay.classList.add('open');
    if (input) {
      input.value = '';
      input.focus();
    }
  }
}

function closeCommandPalette() {
  const overlay = document.getElementById('cmd-modal-overlay');
  if (overlay) overlay.classList.remove('open');
}

function handleCmdSearch() {
  const q = document.getElementById('cmd-input').value.toLowerCase().trim();
  const container = document.getElementById('cmd-results');
  if (!container) return;

  if (q.startsWith('dev-')) {
    container.innerHTML = `
      <div class="cmd-item" onclick="openDeviceDrawer('${q.toUpperCase()}'); closeCommandPalette();">
        <span>Inspect Device ${q.toUpperCase()}</span>
        <span class="kbd-badge">Inspect</span>
      </div>
    `;
  } else if (q.includes('diag') || q.includes('what') || q.includes('sim')) {
    container.innerHTML = `
      <div class="cmd-item" onclick="switchView('diagnostics'); closeCommandPalette();">
        <span>⚡ Device Diagnostics & What-If Simulator</span>
        <span class="kbd-badge">View</span>
      </div>
    `;
  }
}

// -----------------------------------------------------------------------------
// Diagnostics & What-If Simulator Controller
// -----------------------------------------------------------------------------
let whatifActive = false;
let whatifDebounceTimer = null;
let baselineDiagnosticResult = null;

function loadDiagnosticsData() {
  if (!baselineDiagnosticResult) {
    runDiagnosticsInference();
  }
}

function updateSliderVal(metric, val, unit) {
  const el = document.getElementById(`val-${metric}`);
  if (el) el.innerText = `${val}${unit}`;
  onSliderChange();
}

function onSliderChange() {
  if (whatifActive) {
    clearTimeout(whatifDebounceTimer);
    whatifDebounceTimer = setTimeout(runWhatIfInference, 200);
  }
}

function toggleWhatIfMode() {
  whatifActive = !whatifActive;
  const btn = document.getElementById('whatif-toggle-btn');
  const panel = document.getElementById('whatif-panel');

  if (btn) {
    btn.classList.toggle('active', whatifActive);
    btn.innerHTML = whatifActive
      ? '<span class="icon">✕</span> Exit Simulator'
      : '<span class="icon">🧪</span> What-If Simulator';
    btn.style.background = whatifActive ? 'rgba(56, 189, 248, 0.2)' : '';
    btn.style.borderColor = whatifActive ? 'var(--accent-cyan)' : '';
    btn.style.color = whatifActive ? 'var(--accent-cyan)' : '';
  }

  if (panel) {
    panel.style.display = whatifActive ? 'block' : 'none';
  }

  if (whatifActive) {
    runWhatIfInference();
  }
}

function getDiagnosticsPayload() {
  const typeEl = document.querySelector('input[name="Device_Type"]:checked');
  return {
    Device_Type: typeEl ? typeEl.value : 'Router',
    CPU_Usage: parseFloat(document.getElementById('cpu_usage')?.value || 92),
    Memory_Usage: parseFloat(document.getElementById('memory_usage')?.value || 94),
    Temperature: parseFloat(document.getElementById('temperature')?.value || 78),
    Interface_Errors: parseInt(document.getElementById('interface_errors')?.value || 156),
    Packet_Loss: parseFloat(document.getElementById('packet_loss')?.value || 8.2),
    Bandwidth_Usage: parseFloat(document.getElementById('bandwidth_usage')?.value || 95),
    Uptime: parseFloat(document.getElementById('uptime')?.value || 20),
    Log_Errors: parseInt(document.getElementById('log_errors')?.value || 20)
  };
}

async function runDiagnosticsInference() {
  const payload = getDiagnosticsPayload();
  const btn = document.getElementById('btn-run-diag');
  if (btn) {
    btn.disabled = true;
    btn.innerHTML = '<span>⚡ Running Diagnostics...</span>';
  }

  try {
    const res = await fetch('/api/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });
    const data = await res.json();
    baselineDiagnosticResult = data;
    updateUIWithResults(data);
  } catch (err) {
    console.error('Failed to run diagnostics inference:', err);
  } finally {
    if (btn) {
      btn.disabled = false;
      btn.innerHTML = '<span>⚡ Run Live Diagnostics</span>';
    }
  }
}

async function runWhatIfInference() {
  const payload = getDiagnosticsPayload();
  try {
    const res = await fetch('/api/whatif', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });
    const data = await res.json();
    renderWhatIfPanel(data);
  } catch (err) {
    console.error('Failed what-if simulation:', err);
  }
}

function updateUIWithResults(data) {
  // Update Health Score Arc & Text
  const score = data.health_score ?? 100;
  const arc = document.getElementById('health-score-arc');
  const val = document.getElementById('health-score-value');
  const label = document.getElementById('health-score-label');

  if (val) val.innerText = score.toFixed(1);
  if (arc) {
    const circumference = 282.7;
    const fraction = Math.max(0, Math.min(score / 100, 1));
    arc.style.strokeDashoffset = circumference * (1 - fraction);
    const color = score >= 70 ? 'var(--color-low)' : score >= 40 ? 'var(--color-medium)' : 'var(--color-critical)';
    arc.style.stroke = color;
    if (val) val.style.color = color;
    if (label) {
      label.innerText = score >= 70 ? 'HEALTHY' : score >= 40 ? 'DEGRADED' : 'CRITICAL';
      label.style.color = color;
    }
  }

  // Diagnosed Failure Type
  const failType = document.getElementById('failure-type-display');
  if (failType) {
    failType.innerText = data.predicted_failure || data.failure_type || 'NONE';
    failType.style.color = (data.predicted_failure && data.predicted_failure !== 'NONE') ? 'var(--color-critical)' : 'var(--color-low)';
  }

  // Narrative
  const narr = document.getElementById('failure-desc-display');
  if (narr) {
    narr.innerText = data.diagnosis_narrative || data.description || 'Nominal operational status.';
  }

  // Anomaly score & flag
  const anomVal = document.getElementById('anomaly-score-val');
  const anomBadge = document.getElementById('anomaly-flag-badge');
  if (anomVal) anomVal.innerText = `${(data.anomaly_score ?? 0).toFixed(1)}%`;
  if (anomBadge) {
    if (data.is_anomaly) {
      anomBadge.innerText = 'ANOMALY DETECTED';
      anomBadge.style.background = 'rgba(239, 68, 68, 0.2)';
      anomBadge.style.color = 'var(--color-critical)';
    } else {
      anomBadge.innerText = 'NORMAL';
      anomBadge.style.background = 'rgba(34, 197, 94, 0.15)';
      anomBadge.style.color = 'var(--color-low)';
    }
  }

  // Risk Window
  const winEl = document.getElementById('risk-window-text');
  if (winEl) winEl.innerText = data.risk_window || 'Immediate / Normal';

  // SHAP Attributions
  renderShapBars(data.shap_causes || data.top_contributing_causes || []);

  // Recommended Actions
  renderRecommendedRunbook(data.recommended_actions || []);
}

function renderShapBars(causes) {
  const container = document.getElementById('shap-bars-container');
  if (!container) return;
  if (!causes || causes.length === 0) {
    container.innerHTML = '<p class="mono-text" style="font-size:12px; color:var(--text-muted);">No significant risk drivers identified.</p>';
    return;
  }

  const maxAbs = Math.max(...causes.map(c => Math.abs(c.contribution)), 0.001);
  container.innerHTML = causes.map(c => {
    const isRisk = c.direction !== 'decreases_risk';
    const barColor = isRisk ? 'var(--color-critical)' : 'var(--color-low)';
    const pct = Math.round((Math.abs(c.contribution) / maxAbs) * 100);
    const dirIcon = isRisk ? '↑' : '↓';
    return `
      <div class="shap-row">
        <div class="shap-label">
          <span class="shap-dir" style="color:${barColor}; font-weight:700;">${dirIcon}</span>
          <span>${c.feature}</span>
        </div>
        <div class="shap-bar-track">
          <div class="shap-bar-fill" style="width:${pct}%; background:${barColor};"></div>
        </div>
        <span class="shap-pct mono-text" style="color:${barColor};">${Math.abs(c.contribution).toFixed(3)}</span>
      </div>
    `;
  }).join('');
}

function renderRecommendedRunbook(actions) {
  const list = document.getElementById('intel-actions-list');
  if (!list) return;
  if (!actions || actions.length === 0) {
    list.innerHTML = '<li>Device operating within nominal limits. Standard monitoring advised.</li>';
    return;
  }
  list.innerHTML = actions.map(act => `<li>${act}</li>`).join('');
}

function renderWhatIfPanel(whatifData) {
  const panel = document.getElementById('whatif-panel');
  if (!panel) return;
  panel.style.display = 'block';

  const prob = (whatifData.probability != null ? whatifData.probability : (whatifData.failure_probability || 0)) * 100;
  const health = whatifData.health_score ?? 100;

  const probEl = document.getElementById('wi-prob-val');
  const healthEl = document.getElementById('wi-health-val');
  const riskEl = document.getElementById('wi-risk-val');
  const winEl = document.getElementById('wi-window-val');

  if (probEl) {
    probEl.innerText = `${prob.toFixed(1)}%`;
    probEl.style.color = (whatifData.risk === 'CRITICAL' || whatifData.risk === 'HIGH') ? 'var(--color-critical)' : whatifData.risk === 'MEDIUM' ? 'var(--color-medium)' : 'var(--color-low)';
  }
  if (healthEl) {
    healthEl.innerText = health.toFixed(1);
    healthEl.style.color = health >= 70 ? 'var(--color-low)' : health >= 40 ? 'var(--color-medium)' : 'var(--color-critical)';
  }
  if (riskEl) {
    riskEl.innerText = `${whatifData.risk || 'LOW'} RISK`;
  }
  if (winEl) {
    winEl.innerText = whatifData.risk_window || 'Immediate';
  }

  // Deltas
  if (baselineDiagnosticResult) {
    const baseProb = (baselineDiagnosticResult.probability != null ? baselineDiagnosticResult.probability : (baselineDiagnosticResult.failure_probability || 0)) * 100;
    const baseHealth = baselineDiagnosticResult.health_score ?? 100;

    const deltaP = prob - baseProb;
    const deltaH = health - baseHealth;

    const dpEl = document.getElementById('wi-delta-prob');
    const dhEl = document.getElementById('wi-delta-health');

    if (dpEl) {
      dpEl.innerText = `${deltaP > 0 ? '+' : ''}${deltaP.toFixed(1)}%`;
      dpEl.className = `wi-delta ${deltaP > 0 ? 'delta-bad' : 'delta-good'}`;
    }
    if (dhEl) {
      dhEl.innerText = `${deltaH > 0 ? '+' : ''}${deltaH.toFixed(1)}`;
      dhEl.className = `wi-delta ${deltaH < 0 ? 'delta-bad' : 'delta-good'}`;
    }
  }

  // Also update right column details in real-time
  updateUIWithResults(whatifData);
}

window.updateUIWithResults = updateUIWithResults;
window.renderWhatIfPanel = renderWhatIfPanel;
window.toggleWhatIfMode = toggleWhatIfMode;
window.runDiagnosticsInference = runDiagnosticsInference;
window.updateSliderVal = updateSliderVal;
window.onSliderChange = onSliderChange;

