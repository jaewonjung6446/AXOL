/* Axol DSL Visual Debugger — Frontend Logic */

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------

let currentTrace = [];
let currentStep = -1;
let stateChart = null;
let playInterval = null;

// ---------------------------------------------------------------------------
// API helpers
// ---------------------------------------------------------------------------

async function apiCall(endpoint, body = null) {
  const opts = body !== null
    ? { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) }
    : { method: 'GET' };
  const res = await fetch(endpoint, opts);
  const data = await res.json();
  if (!res.ok && data.error) throw new Error(data.error);
  return data;
}

// ---------------------------------------------------------------------------
// Load examples
// ---------------------------------------------------------------------------

async function loadExamples() {
  try {
    const data = await apiCall('/api/examples');
    const sel = document.getElementById('examples');
    data.examples.forEach(ex => {
      const opt = document.createElement('option');
      opt.value = ex.source;
      opt.textContent = ex.name;
      sel.appendChild(opt);
    });
    sel.addEventListener('change', () => {
      if (sel.value) document.getElementById('editor').value = sel.value;
    });
  } catch (e) {
    console.error('Failed to load examples:', e);
  }
}

// ---------------------------------------------------------------------------
// Run program
// ---------------------------------------------------------------------------

async function runProgram() {
  const source = document.getElementById('editor').value;
  clearError();

  try {
    const data = await apiCall('/api/run', { source });
    showRunResult(data);
    currentTrace = data.trace || [];
    currentStep = currentTrace.length > 0 ? currentTrace.length - 1 : -1;
    renderTrace();
    buildChartKeySelector();
    updateChart();
  } catch (e) {
    showError(e.message);
  }
}

// ---------------------------------------------------------------------------
// Optimize
// ---------------------------------------------------------------------------

async function optimizeProgram() {
  const source = document.getElementById('editor').value;
  clearError();

  try {
    // Run with and without optimize
    const [orig, opt] = await Promise.all([
      apiCall('/api/run', { source, optimize: false }),
      apiCall('/api/run', { source, optimize: true }),
    ]);
    const optData = await apiCall('/api/optimize', { source });

    showRunResult(opt);
    currentTrace = opt.trace || [];
    currentStep = currentTrace.length > 0 ? currentTrace.length - 1 : -1;
    renderTrace();
    buildChartKeySelector();
    updateChart();

    // Show perf comparison
    showPerfResult(optData);
  } catch (e) {
    showError(e.message);
  }
}

// ---------------------------------------------------------------------------
// Encrypt
// ---------------------------------------------------------------------------

async function encryptProgram() {
  const source = document.getElementById('editor').value;
  clearError();

  try {
    const data = await apiCall('/api/encrypt', { source });
    showEncryptResult(data);
    showTab('exec-encrypt', document.querySelectorAll('.tab')[1]);
  } catch (e) {
    showError(e.message);
  }
}

// ---------------------------------------------------------------------------
// Display: Run result
// ---------------------------------------------------------------------------

function showRunResult(data) {
  const summaryEl = document.getElementById('result-summary');
  summaryEl.innerHTML = `
    <div class="result-box">
      <div class="label">Steps</div>
      <div class="value">${data.steps_executed}</div>
    </div>
    <div class="result-box">
      <div class="label">Terminated By</div>
      <div class="value">${data.terminated_by}</div>
    </div>
    <div class="result-box">
      <div class="label">Time</div>
      <div class="value">${data.elapsed_ms || '-'} ms</div>
    </div>
  `;

  const stateEl = document.getElementById('result-state');
  stateEl.innerHTML = '';
  if (data.final_state) {
    for (const [key, vals] of Object.entries(data.final_state)) {
      const formatted = Array.isArray(vals)
        ? vals.map(v => typeof v === 'number' ? v.toFixed(4) : v).join(', ')
        : vals;
      stateEl.innerHTML += `
        <div class="state-entry">
          <span class="state-key">${key}</span>
          <span class="state-val">[${formatted}]</span>
        </div>
      `;
    }
  }
}

// ---------------------------------------------------------------------------
// Display: Encrypt result
// ---------------------------------------------------------------------------

function showEncryptResult(data) {
  const el = document.getElementById('encrypt-result');
  let html = '<h4 style="color: var(--warning); margin-bottom: 12px;">Encryption Demo</h4>';

  // Original vs Encrypted vs Decrypted state
  html += '<div style="display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 8px; margin-bottom: 16px;">';

  html += '<div class="result-box"><div class="label">Original</div>';
  if (data.original_state) {
    for (const [k, v] of Object.entries(data.original_state)) {
      html += `<div class="state-entry"><span class="state-key">${k}</span><span class="state-val">[${formatArr(v)}]</span></div>`;
    }
  }
  html += '</div>';

  html += '<div class="result-box"><div class="label">Encrypted</div>';
  if (data.encrypted_state) {
    for (const [k, v] of Object.entries(data.encrypted_state)) {
      html += `<div class="state-entry"><span class="state-key">${k}</span><span class="state-val">[${formatArr(v)}]</span></div>`;
    }
  }
  html += '</div>';

  html += '<div class="result-box"><div class="label">Decrypted</div>';
  if (data.decrypted_state) {
    for (const [k, v] of Object.entries(data.decrypted_state)) {
      html += `<div class="state-entry"><span class="state-key">${k}</span><span class="state-val">[${formatArr(v)}]</span></div>`;
    }
  }
  html += '</div></div>';

  // Matrix heatmaps
  if (data.original_matrix && data.encrypted_matrix) {
    html += '<div class="heatmap-container">';
    html += '<div><div class="heatmap-label">Original Matrix</div><canvas class="heatmap" id="heatmap-orig" width="200" height="200"></canvas></div>';
    html += '<div><div class="heatmap-label">Encrypted Matrix</div><canvas class="heatmap" id="heatmap-enc" width="200" height="200"></canvas></div>';
    html += '</div>';
  }

  el.innerHTML = html;

  // Draw heatmaps
  if (data.original_matrix && data.encrypted_matrix) {
    setTimeout(() => {
      drawHeatmap('heatmap-orig', data.original_matrix);
      drawHeatmap('heatmap-enc', data.encrypted_matrix);
    }, 50);
  }
}

function formatArr(arr) {
  if (!Array.isArray(arr)) return String(arr);
  return arr.map(v => typeof v === 'number' ? v.toFixed(3) : v).join(', ');
}

function drawHeatmap(canvasId, matrix) {
  const canvas = document.getElementById(canvasId);
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  const rows = matrix.length;
  const cols = matrix[0].length;
  const cellW = canvas.width / cols;
  const cellH = canvas.height / rows;

  // Find min/max
  let mn = Infinity, mx = -Infinity;
  for (const row of matrix) {
    for (const v of row) {
      mn = Math.min(mn, v);
      mx = Math.max(mx, v);
    }
  }
  const range = mx - mn || 1;

  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      const t = (matrix[r][c] - mn) / range;
      const r_ = Math.round(247 * t + 26 * (1 - t));
      const g_ = Math.round(118 * (1 - Math.abs(t - 0.5) * 2));
      const b_ = Math.round(247 * (1 - t) + 26 * t);
      ctx.fillStyle = `rgb(${r_}, ${g_}, ${b_})`;
      ctx.fillRect(c * cellW, r * cellH, cellW, cellH);
    }
  }
}

// ---------------------------------------------------------------------------
// Display: Perf result
// ---------------------------------------------------------------------------

function showPerfResult(data) {
  const el = document.getElementById('perf-result');
  el.innerHTML = `
    <h4 style="color: var(--accent2); margin-bottom: 12px;">Optimizer Comparison</h4>
    <div class="perf-grid">
      <div class="perf-card">
        <h4>Original</h4>
        <div class="state-entry"><span class="state-key">Transitions</span><span class="state-val">${data.original.transition_count}</span></div>
        <div class="state-entry"><span class="state-key">Steps</span><span class="state-val">${data.original.steps_executed}</span></div>
        <div class="state-entry"><span class="state-key">Time</span><span class="state-val">${data.original.elapsed_ms} ms</span></div>
      </div>
      <div class="perf-card">
        <h4>Optimized</h4>
        <div class="state-entry"><span class="state-key">Transitions</span><span class="state-val">${data.optimized.transition_count}</span></div>
        <div class="state-entry"><span class="state-key">Steps</span><span class="state-val">${data.optimized.steps_executed}</span></div>
        <div class="state-entry"><span class="state-key">Time</span><span class="state-val">${data.optimized.elapsed_ms} ms</span></div>
      </div>
    </div>
  `;
  showTab('exec-perf', document.querySelectorAll('.tab')[2]);
}

// ---------------------------------------------------------------------------
// Trace viewer
// ---------------------------------------------------------------------------

function renderTrace() {
  const tbody = document.getElementById('trace-body');
  tbody.innerHTML = '';

  currentTrace.forEach((entry, i) => {
    const tr = document.createElement('tr');
    tr.className = i === currentStep ? 'active' : '';
    tr.onclick = () => { currentStep = i; renderTrace(); updateChart(); };

    const stateStr = Object.entries(entry.state)
      .map(([k, v]) => `${k}=[${formatArr(v)}]`)
      .join(' ');

    tr.innerHTML = `
      <td>${entry.step}</td>
      <td>${entry.transition}</td>
      <td style="font-size: 11px; color: var(--text-dim);">${stateStr}</td>
    `;
    tbody.appendChild(tr);
  });

  updateStepLabel();
}

function traceStep(delta) {
  if (currentTrace.length === 0) return;
  currentStep = Math.max(0, Math.min(currentTrace.length - 1, currentStep + delta));
  renderTrace();
  updateChart();
}

function tracePlay() {
  if (playInterval) {
    clearInterval(playInterval);
    playInterval = null;
    return;
  }
  currentStep = 0;
  playInterval = setInterval(() => {
    if (currentStep >= currentTrace.length - 1) {
      clearInterval(playInterval);
      playInterval = null;
      return;
    }
    currentStep++;
    renderTrace();
    updateChart();
  }, 300);
}

function updateStepLabel() {
  const label = document.getElementById('trace-step-label');
  label.textContent = currentStep >= 0 ? `Step ${currentStep + 1}/${currentTrace.length}` : '-';
}

// ---------------------------------------------------------------------------
// Chart
// ---------------------------------------------------------------------------

function buildChartKeySelector() {
  const sel = document.getElementById('chart-key');
  sel.innerHTML = '<option value="__all__">All Keys</option>';

  if (currentTrace.length === 0) return;
  const keys = Object.keys(currentTrace[0].state);
  keys.forEach(k => {
    const opt = document.createElement('option');
    opt.value = k;
    opt.textContent = k;
    sel.appendChild(opt);
  });
}

function updateChart() {
  if (currentTrace.length === 0) return;

  const sel = document.getElementById('chart-key');
  const selectedKey = sel.value;

  const labels = currentTrace.map(e => `Step ${e.step}`);
  const datasets = [];
  const colors = ['#7aa2f7', '#bb9af7', '#9ece6a', '#e0af68', '#f7768e', '#7dcfff', '#c0caf5'];

  if (selectedKey === '__all__') {
    const allKeys = Object.keys(currentTrace[0].state);
    allKeys.forEach((key, ki) => {
      const values = currentTrace.map(e => {
        const arr = e.state[key];
        return Array.isArray(arr) ? arr[0] : arr;
      });
      datasets.push({
        label: key,
        data: values,
        borderColor: colors[ki % colors.length],
        backgroundColor: colors[ki % colors.length] + '33',
        tension: 0.3,
        pointRadius: 3,
      });
    });
  } else {
    const arr0 = currentTrace[0].state[selectedKey];
    if (!arr0) return;
    const dim = Array.isArray(arr0) ? arr0.length : 1;

    for (let d = 0; d < dim; d++) {
      const values = currentTrace.map(e => {
        const arr = e.state[selectedKey];
        return Array.isArray(arr) ? arr[d] : arr;
      });
      datasets.push({
        label: `${selectedKey}[${d}]`,
        data: values,
        borderColor: colors[d % colors.length],
        backgroundColor: colors[d % colors.length] + '33',
        tension: 0.3,
        pointRadius: 3,
      });
    }
  }

  const canvas = document.getElementById('state-chart');
  if (stateChart) stateChart.destroy();
  stateChart = new Chart(canvas, {
    type: 'line',
    data: { labels, datasets },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        x: { ticks: { color: '#565f89' }, grid: { color: '#3b426133' } },
        y: { ticks: { color: '#565f89' }, grid: { color: '#3b426133' } },
      },
      plugins: {
        legend: { labels: { color: '#c0caf5', font: { family: "'JetBrains Mono', monospace", size: 11 } } },
      },
    },
  });
}

// ---------------------------------------------------------------------------
// Tabs
// ---------------------------------------------------------------------------

function showTab(tabId, btn) {
  document.querySelectorAll('.tab-content').forEach(el => el.classList.remove('active'));
  document.querySelectorAll('.tab').forEach(el => el.classList.remove('active'));
  document.getElementById(tabId).classList.add('active');
  if (btn) btn.classList.add('active');
}

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

function showError(msg) {
  document.getElementById('result-error').innerHTML = `<div class="error-msg">${msg}</div>`;
}

function clearError() {
  document.getElementById('result-error').innerHTML = '';
}

// ---------------------------------------------------------------------------
// Init
// ---------------------------------------------------------------------------

document.addEventListener('DOMContentLoaded', loadExamples);
