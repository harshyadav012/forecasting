/**
 * MetalForecasting - Institutional AI Gold & Silver Forecasting Terminal
 * Client-side Controller & Visual Engine
 */

// Global App State
const state = {
  horizon: 6,
  currency: 'DUAL', // 'USD', 'INR', 'DUAL'
  includeTax: false, // Domestic Indian import duty + GST simulation (~12%)
  isLoading: false,
  healthData: null,
  forecastData: null,
  activeTab: 'tab-trajectory',
  portfolio: {
    goldQty: 10,
    goldUnit: 'g', // 'g' or 'oz'
    silverQty: 1,
    silverUnit: 'kg' // 'kg' or 'oz'
  }
};

// Numerical Constants
const OZ_TO_G = 31.1034768; // 1 Troy Ounce = 31.1035 grams
const TAX_MULTIPLIER = 1.12; // Approx 9% custom duty + 3% GST in India

// Chart Instances Registry
const chartRegistry = {};

// SHAP Feature Importance Static Reference (Derived directly from trained backend GMM/GradientBoosting)
const shapData = [
  { feature: 'Spread (10Y - 2Y)', normal: 0.0138, crisis: 0.0027, delta: -0.0111, desc: 'Yield curve steepness & recession expectations' },
  { feature: 'DXY (US Dollar Index)', normal: 0.0126, crisis: 0.0067, delta: -0.0059, desc: 'Global currency reserve & greenback strength' },
  { feature: 'Inflation (CPI YoY %)', normal: 0.0076, crisis: 0.0027, delta: -0.0049, desc: 'Purchasing power loss & hard-asset hedge demand' },
  { feature: 'Real Yield (10Y - CPI)', normal: 0.0067, crisis: 0.0020, delta: -0.0047, desc: 'Opportunity cost of holding non-yielding metals' },
  { feature: 'VIX (Fear Index)', normal: 0.0057, crisis: 0.0046, delta: -0.0011, desc: 'Equity volatility & flight-to-safety regime' },
  { feature: 'USD/INR Exchange Rate', normal: 0.0032, crisis: 0.0059, delta: 0.0027, desc: 'Rupee depreciation & domestic parity pricing' },
  { feature: 'US 10-Year Treasury Yield', normal: 0.0034, crisis: 0.0025, delta: -0.0009, desc: 'Risk-free sovereign rate benchmark' },
  { feature: 'Fed Funds Effective Rate', normal: 0.0030, crisis: 0.0034, delta: 0.0004, desc: 'Monetary policy stance and liquidity cycles' },
  { feature: 'FSI (Financial Stress Index)', normal: 0.0012, crisis: 0.0042, delta: 0.0030, desc: 'Systemic banking & credit risk indicator' }
];

// Utility: Number Formatters
const fmt = {
  usd(val, decimals = 2) {
    if (val == null || isNaN(val)) return '—';
    return '$' + Number(val).toLocaleString('en-US', {
      minimumFractionDigits: decimals,
      maximumFractionDigits: decimals
    });
  },
  inr(val, decimals = 0) {
    if (val == null || isNaN(val)) return '—';
    return '₹' + Math.round(val).toLocaleString('en-IN');
  },
  pct(val, decimals = 2) {
    if (val == null || isNaN(val)) return '—';
    const n = Number(val);
    const sign = n > 0 ? '+' : '';
    return sign + n.toFixed(decimals) + '%';
  },
  ratio(val, decimals = 1) {
    if (val == null || isNaN(val)) return '—';
    return Number(val).toFixed(decimals) + 'x';
  },
  date(dStr) {
    if (!dStr) return '';
    const d = new Date(dStr);
    return d.toLocaleDateString('en-US', { month: 'short', year: 'numeric' });
  },
  dateShort(dStr) {
    if (!dStr) return '';
    const d = new Date(dStr);
    return d.toLocaleDateString('en-US', { month: 'short', year: '2-digit' });
  }
};

// UI Notification Toast
function showToast(message, type = 'success') {
  const container = document.getElementById('toast-container');
  if (!container) return;

  const toast = document.createElement('div');
  toast.className = `toast ${type}`;
  toast.innerHTML = `
    <span>${type === 'success' ? '✓' : 'ℹ'}</span>
    <span>${message}</span>
  `;

  container.appendChild(toast);

  requestAnimationFrame(() => {
    toast.classList.add('show');
  });

  setTimeout(() => {
    toast.classList.remove('show');
    setTimeout(() => toast.remove(), 350);
  }, 3200);
}

// Destroy all charts cleanly before re-rendering
function destroyCharts() {
  Object.keys(chartRegistry).forEach(k => {
    if (chartRegistry[k]) {
      chartRegistry[k].destroy();
      delete chartRegistry[k];
    }
  });
}

// Chart.js Shared Defaults
function getChartOptions(yTickCallback, isBar = false) {
  return {
    responsive: true,
    maintainAspectRatio: false,
    interaction: {
      mode: 'index',
      intersect: false
    },
    plugins: {
      legend: { display: false },
      tooltip: {
        backgroundColor: '#121620',
        borderColor: 'rgba(255, 255, 255, 0.15)',
        borderWidth: 1,
        titleColor: '#f8fafc',
        bodyColor: '#94a3b8',
        padding: 12,
        boxPadding: 6,
        usePointStyle: true,
        callbacks: {
          label: function(context) {
            let label = context.dataset.label || '';
            if (label) label += ': ';
            if (context.parsed.y !== null) {
              label += yTickCallback ? yTickCallback(context.parsed.y) : context.parsed.y;
            }
            return label;
          }
        }
      }
    },
    scales: {
      x: {
        grid: {
          color: 'rgba(255, 255, 255, 0.04)',
          drawBorder: false
        },
        ticks: {
          color: '#64748b',
          font: { family: "'JetBrains Mono', monospace", size: 10.5 },
          maxRotation: 0,
          autoSkip: true,
          maxTicksLimit: 12
        }
      },
      y: {
        grid: {
          color: 'rgba(255, 255, 255, 0.05)',
          drawBorder: false
        },
        ticks: {
          color: '#94a3b8',
          font: { family: "'JetBrains Mono', monospace", size: 10.5 },
          callback: yTickCallback || (v => v)
        }
      }
    }
  };
}

// Fetch Backend /health
async function fetchHealth() {
  try {
    const res = await fetch('/health');
    if (!res.ok) throw new Error(`Health check failed (${res.status})`);
    const data = await res.json();
    state.healthData = data;

    // Update Topbar Ticker
    document.getElementById('ticker-gold').textContent = fmt.usd(data.latest_gold);
    document.getElementById('ticker-silver').textContent = fmt.usd(data.latest_silver);
    document.getElementById('ticker-inr').textContent = '₹' + Number(data.latest_usdinr).toFixed(2);
    document.getElementById('ticker-date').textContent = fmt.date(data.latest_date);

    return data;
  } catch (err) {
    console.warn('Could not fetch /health directly:', err);
    return null;
  }
}

// Execute Forecast Request to Backend /forecast
async function runForecast(horizonMonths) {
  if (state.isLoading) return;

  const n = parseInt(horizonMonths || state.horizon, 10);
  if (isNaN(n) || n < 1 || n > 36) {
    showToast('Please select a horizon between 1 and 36 months.', 'error');
    return;
  }

  state.horizon = n;
  state.isLoading = true;

  // Update UI Loading State
  const btn = document.getElementById('run-btn');
  const btnText = document.getElementById('run-btn-text');
  const spinner = document.getElementById('run-btn-spinner');
  if (btn) btn.disabled = true;
  if (btnText) btnText.textContent = 'Simulating...';
  if (spinner) spinner.style.display = 'inline-block';

  try {
    const res = await fetch('/forecast', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ months: state.horizon })
    });

    if (!res.ok) {
      throw new Error(`Backend forecast error (HTTP ${res.status})`);
    }

    const data = await res.json();

    if (!Array.isArray(data) || data.length === 0) {
      throw new Error('Received empty forecast dataset.');
    }

    state.forecastData = data;

    // Render Everything
    renderAll();
    showToast(`Simulated ${state.horizon}-month Monte Carlo forecast.`, 'success');
  } catch (err) {
    console.error('Forecast failed:', err);
    showToast(err.message || 'Forecast calculation failed.', 'error');
  } finally {
    state.isLoading = false;
    if (btn) btn.disabled = false;
    if (btnText) btnText.textContent = 'Generate Forecast';
    if (spinner) spinner.style.display = 'none';
  }
}

// Render Dashboard Elements
function renderAll() {
  if (!state.forecastData || state.forecastData.length === 0) return;

  renderSummaryCards();
  renderTrajectoryCharts();
  renderDomesticCharts();
  renderReturnsChart();
  renderRelativeAndRatioCharts();
  renderDataTable();
  calculatePortfolio();
}

// Render Top Metric Cards
function renderSummaryCards() {
  const data = state.forecastData;
  const first = data[0];
  const last = data[data.length - 1];

  const goldNow = Number(first.gold_now);
  const silverNow = Number(first.silver_now);
  const usdinrNow = state.healthData?.latest_usdinr ? Number(state.healthData.latest_usdinr) : Number(first.usdinr);

  const goldEnd = Number(last.gold_usd);
  const silverEnd = Number(last.silver_usd);
  const inrEnd = Number(last.usdinr);

  const goldChgPct = ((goldEnd - goldNow) / goldNow) * 100;
  const silverChgPct = ((silverEnd - silverNow) / silverNow) * 100;
  const inrChgPct = ((inrEnd - usdinrNow) / usdinrNow) * 100;

  // Domestic Indian Calculations
  const tax = state.includeTax ? TAX_MULTIPLIER : 1.0;
  const goldInrNow = (goldNow / OZ_TO_G) * usdinrNow * 10 * tax;
  const goldInrEnd = (goldEnd / OZ_TO_G) * inrEnd * 10 * tax;
  const silverInrNow = (silverNow / OZ_TO_G) * usdinrNow * 1000 * tax;
  const silverInrEnd = (silverEnd / OZ_TO_G) * inrEnd * 1000 * tax;

  // Gold Card
  const elGoldVal = document.getElementById('card-gold-val');
  const elGoldSec = document.getElementById('card-gold-sec');
  const elGoldBadge = document.getElementById('card-gold-badge');
  const elGoldRange = document.getElementById('card-gold-range');

  if (state.currency === 'INR') {
    elGoldVal.textContent = fmt.inr(goldInrEnd);
    elGoldSec.textContent = `Today: ${fmt.inr(goldInrNow)} / 10g`;
  } else {
    elGoldVal.textContent = fmt.usd(goldEnd, 0);
    elGoldSec.textContent = `Today: ${fmt.usd(goldNow, 0)} (${fmt.inr(goldInrEnd)} / 10g)`;
  }
  elGoldBadge.textContent = fmt.pct(goldChgPct);
  elGoldBadge.className = `metric-badge ${goldChgPct >= 0 ? 'badge-up' : 'badge-down'}`;
  elGoldRange.textContent = `80% Band: ${fmt.usd(last.gold_usd_p10, 0)} – ${fmt.usd(last.gold_usd_p90, 0)}`;

  // Silver Card
  const elSilvVal = document.getElementById('card-silver-val');
  const elSilvSec = document.getElementById('card-silver-sec');
  const elSilvBadge = document.getElementById('card-silver-badge');
  const elSilvRange = document.getElementById('card-silver-range');

  if (state.currency === 'INR') {
    elSilvVal.textContent = fmt.inr(silverInrEnd);
    elSilvSec.textContent = `Today: ${fmt.inr(silverInrNow)} / kg`;
  } else {
    elSilvVal.textContent = fmt.usd(silverEnd, 2);
    elSilvSec.textContent = `Today: ${fmt.usd(silverNow, 2)} (${fmt.inr(silverInrEnd)} / kg)`;
  }
  elSilvBadge.textContent = fmt.pct(silverChgPct);
  elSilvBadge.className = `metric-badge ${silverChgPct >= 0 ? 'badge-up' : 'badge-down'}`;
  elSilvRange.textContent = `80% Band: ${fmt.usd(last.silver_usd_p10, 2)} – ${fmt.usd(last.silver_usd_p90, 2)}`;

  // Gold / Silver Ratio Card
  const ratioNow = goldNow / silverNow;
  const ratioEnd = goldEnd / silverEnd;
  const ratioChg = ratioEnd - ratioNow;

  document.getElementById('card-ratio-val').textContent = fmt.ratio(ratioEnd);
  document.getElementById('card-ratio-sec').textContent = `Benchmark: ${fmt.ratio(ratioNow)}`;
  const ratioBadge = document.getElementById('card-ratio-badge');
  ratioBadge.textContent = (ratioChg >= 0 ? '+' : '') + ratioChg.toFixed(1) + ' pts';
  ratioBadge.className = `metric-badge ${ratioChg > 0 ? 'badge-neutral' : 'badge-up'}`;
  document.getElementById('card-ratio-desc').textContent = ratioEnd < ratioNow 
    ? 'Silver outperforming Gold (Ratio compression)' 
    : 'Gold holding safe-haven premium';

  // USD/INR Card
  document.getElementById('card-inr-val').textContent = '₹' + inrEnd.toFixed(2);
  document.getElementById('card-inr-sec').textContent = `Anchor: ₹${usdinrNow.toFixed(2)}`;
  const inrBadge = document.getElementById('card-inr-badge');
  inrBadge.textContent = fmt.pct(inrChgPct);
  inrBadge.className = `metric-badge ${inrChgPct >= 0 ? 'badge-up' : 'badge-down'}`;
  document.getElementById('card-inr-range').textContent = `Annualized: ${(inrChgPct / (state.horizon / 12)).toFixed(1)}% YoY`;
}

// Chart 1: Gold & Silver Monte Carlo Trajectory & Confidence Bands
function renderTrajectoryCharts() {
  const data = state.forecastData;
  const first = data[0];
  const goldNow = Number(first.gold_now);
  const silverNow = Number(first.silver_now);

  const labels = ['Now', ...data.map(d => fmt.dateShort(d.date))];

  // Gold Trajectory
  const goldMean = [goldNow, ...data.map(d => Number(d.gold_usd))];
  const goldP10 = [goldNow, ...data.map(d => Number(d.gold_usd_p10))];
  const goldP90 = [goldNow, ...data.map(d => Number(d.gold_usd_p90))];

  const ctxGold = document.getElementById('chart-gold-trajectory');
  if (chartRegistry.goldTrajectory) chartRegistry.goldTrajectory.destroy();

  chartRegistry.goldTrajectory = new Chart(ctxGold, {
    type: 'line',
    data: {
      labels: labels,
      datasets: [
        {
          label: 'Bull Target (P90)',
          data: goldP90,
          borderColor: 'transparent',
          backgroundColor: 'rgba(245, 158, 11, 0.12)',
          fill: '+1',
          pointRadius: 0,
          tension: 0.35
        },
        {
          label: 'Bear Support (P10)',
          data: goldP10,
          borderColor: 'transparent',
          backgroundColor: 'rgba(245, 158, 11, 0.12)',
          fill: false,
          pointRadius: 0,
          tension: 0.35
        },
        {
          label: 'Base Case (Expected)',
          data: goldMean,
          borderColor: '#f59e0b',
          borderWidth: 2.75,
          backgroundColor: 'transparent',
          pointRadius: 3.5,
          pointBackgroundColor: '#fbbf24',
          pointBorderColor: '#07080c',
          pointBorderWidth: 1.5,
          tension: 0.35
        }
      ]
    },
    options: getChartOptions(v => fmt.usd(v, 0))
  });

  // Silver Trajectory
  const silverMean = [silverNow, ...data.map(d => Number(d.silver_usd))];
  const silverP10 = [silverNow, ...data.map(d => Number(d.silver_usd_p10))];
  const silverP90 = [silverNow, ...data.map(d => Number(d.silver_usd_p90))];

  const ctxSilver = document.getElementById('chart-silver-trajectory');
  if (chartRegistry.silverTrajectory) chartRegistry.silverTrajectory.destroy();

  chartRegistry.silverTrajectory = new Chart(ctxSilver, {
    type: 'line',
    data: {
      labels: labels,
      datasets: [
        {
          label: 'Bull Target (P90)',
          data: silverP90,
          borderColor: 'transparent',
          backgroundColor: 'rgba(203, 213, 225, 0.12)',
          fill: '+1',
          pointRadius: 0,
          tension: 0.35
        },
        {
          label: 'Bear Support (P10)',
          data: silverP10,
          borderColor: 'transparent',
          backgroundColor: 'rgba(203, 213, 225, 0.12)',
          fill: false,
          pointRadius: 0,
          tension: 0.35
        },
        {
          label: 'Base Case (Expected)',
          data: silverMean,
          borderColor: '#cbd5e1',
          borderWidth: 2.75,
          backgroundColor: 'transparent',
          pointRadius: 3.5,
          pointBackgroundColor: '#f1f5f9',
          pointBorderColor: '#07080c',
          pointBorderWidth: 1.5,
          tension: 0.35
        }
      ]
    },
    options: getChartOptions(v => fmt.usd(v, 2))
  });
}

// Chart 2: Domestic Indian Market (INR / 10g & INR / kg)
function renderDomesticCharts() {
  const data = state.forecastData;
  const tax = state.includeTax ? TAX_MULTIPLIER : 1.0;
  const labels = data.map(d => fmt.dateShort(d.date));

  const goldInrData = data.map(d => (Number(d.gold_usd) / OZ_TO_G) * Number(d.usdinr) * 10 * tax);
  const silverInrData = data.map(d => (Number(d.silver_usd) / OZ_TO_G) * Number(d.usdinr) * 1000 * tax);

  // Gold INR Bar Chart
  const ctxGoldInr = document.getElementById('chart-gold-inr');
  if (chartRegistry.goldInr) chartRegistry.goldInr.destroy();

  chartRegistry.goldInr = new Chart(ctxGoldInr, {
    type: 'bar',
    data: {
      labels: labels,
      datasets: [{
        label: 'Gold INR / 10g',
        data: goldInrData,
        backgroundColor: 'rgba(245, 158, 11, 0.65)',
        borderColor: '#f59e0b',
        borderWidth: 1.5,
        borderRadius: 6
      }]
    },
    options: getChartOptions(v => fmt.inr(v), true)
  });

  // Silver INR Bar Chart
  const ctxSilvInr = document.getElementById('chart-silver-inr');
  if (chartRegistry.silverInr) chartRegistry.silverInr.destroy();

  chartRegistry.silverInr = new Chart(ctxSilvInr, {
    type: 'bar',
    data: {
      labels: labels,
      datasets: [{
        label: 'Silver INR / kg',
        data: silverInrData,
        backgroundColor: 'rgba(203, 213, 225, 0.6)',
        borderColor: '#cbd5e1',
        borderWidth: 1.5,
        borderRadius: 6
      }]
    },
    options: getChartOptions(v => fmt.inr(v), true)
  });
}

// Chart 3: Month-on-Month Expected Returns
function renderReturnsChart() {
  const data = state.forecastData;
  const first = data[0];
  const goldNow = Number(first.gold_now);
  const silverNow = Number(first.silver_now);

  const labels = data.map(d => fmt.dateShort(d.date));
  const goldMoM = [];
  const silverMoM = [];

  for (let i = 0; i < data.length; i++) {
    const prevG = i === 0 ? goldNow : Number(data[i - 1].gold_usd);
    const prevS = i === 0 ? silverNow : Number(data[i - 1].silver_usd);

    goldMoM.push(((Number(data[i].gold_usd) - prevG) / prevG) * 100);
    silverMoM.push(((Number(data[i].silver_usd) - prevS) / prevS) * 100);
  }

  const ctxRet = document.getElementById('chart-returns');
  if (chartRegistry.returns) chartRegistry.returns.destroy();

  chartRegistry.returns = new Chart(ctxRet, {
    type: 'bar',
    data: {
      labels: labels,
      datasets: [
        {
          label: 'Gold MoM %',
          data: goldMoM,
          backgroundColor: goldMoM.map(v => v >= 0 ? 'rgba(245, 158, 11, 0.8)' : 'rgba(244, 63, 94, 0.75)'),
          borderRadius: 4
        },
        {
          label: 'Silver MoM %',
          data: silverMoM,
          backgroundColor: silverMoM.map(v => v >= 0 ? 'rgba(203, 213, 225, 0.8)' : 'rgba(244, 63, 94, 0.55)'),
          borderRadius: 4
        }
      ]
    },
    options: getChartOptions(v => fmt.pct(v, 1), true)
  });
}

// Chart 4: Relative Performance (Indexed = 100) & Gold/Silver Ratio Trend
function renderRelativeAndRatioCharts() {
  const data = state.forecastData;
  const first = data[0];
  const goldNow = Number(first.gold_now);
  const silverNow = Number(first.silver_now);

  const labels = ['Now', ...data.map(d => fmt.dateShort(d.date))];

  // Normalized Indexed to 100
  const goldNorm = [100, ...data.map(d => (Number(d.gold_usd) / goldNow) * 100)];
  const silverNorm = [100, ...data.map(d => (Number(d.silver_usd) / silverNow) * 100)];

  const ctxNorm = document.getElementById('chart-relative');
  if (chartRegistry.relative) chartRegistry.relative.destroy();

  chartRegistry.relative = new Chart(ctxNorm, {
    type: 'line',
    data: {
      labels: labels,
      datasets: [
        {
          label: 'Gold (Base=100)',
          data: goldNorm,
          borderColor: '#f59e0b',
          borderWidth: 2.5,
          tension: 0.35,
          pointRadius: 2.5,
          fill: false
        },
        {
          label: 'Silver (Base=100)',
          data: silverNorm,
          borderColor: '#cbd5e1',
          borderWidth: 2.5,
          borderDash: [5, 4],
          tension: 0.35,
          pointRadius: 2.5,
          fill: false
        }
      ]
    },
    options: getChartOptions(v => v.toFixed(0) + ' pts')
  });

  // Gold / Silver Ratio
  const ratioData = [goldNow / silverNow, ...data.map(d => Number(d.gold_usd) / Number(d.silver_usd))];
  const ctxRatio = document.getElementById('chart-ratio-trend');
  if (chartRegistry.ratioTrend) chartRegistry.ratioTrend.destroy();

  chartRegistry.ratioTrend = new Chart(ctxRatio, {
    type: 'line',
    data: {
      labels: labels,
      datasets: [{
        label: 'Gold / Silver Ratio',
        data: ratioData,
        borderColor: '#a855f7',
        backgroundColor: 'rgba(168, 85, 247, 0.08)',
        fill: true,
        borderWidth: 2.5,
        tension: 0.35,
        pointRadius: 3,
        pointBackgroundColor: '#c084fc'
      }]
    },
    options: getChartOptions(v => fmt.ratio(v))
  });
}

// Render Complete Forecast Data Table
function renderDataTable() {
  const tbody = document.getElementById('forecast-tbody');
  if (!tbody) return;

  const data = state.forecastData;
  const tax = state.includeTax ? TAX_MULTIPLIER : 1.0;
  let html = '';

  data.forEach((row, i) => {
    const goldUsd = Number(row.gold_usd);
    const silvUsd = Number(row.silver_usd);
    const fx = Number(row.usdinr);

    const goldInr = (goldUsd / OZ_TO_G) * fx * 10 * tax;
    const silvInr = (silvUsd / OZ_TO_G) * fx * 1000 * tax;
    const ratio = goldUsd / silvUsd;

    // Previous month comparison
    const prevG = i === 0 ? Number(row.gold_now) : Number(data[i - 1].gold_usd);
    const gMoM = ((goldUsd - prevG) / prevG) * 100;

    html += `
      <tr>
        <td class="td-date">${fmt.date(row.date)}</td>
        <td class="td-gold">${fmt.usd(goldUsd, 2)}</td>
        <td class="td-range">${fmt.usd(row.gold_usd_p10, 0)} – ${fmt.usd(row.gold_usd_p90, 0)}</td>
        <td class="td-mono">${fmt.inr(goldInr)}</td>
        <td class="td-silver">${fmt.usd(silvUsd, 2)}</td>
        <td class="td-range">${fmt.usd(row.silver_usd_p10, 2)} – ${fmt.usd(row.silver_usd_p90, 2)}</td>
        <td class="td-mono">${fmt.inr(silvInr)}</td>
        <td class="td-mono">₹${fx.toFixed(2)}</td>
        <td class="td-mono">${fmt.ratio(ratio)}</td>
        <td>
          <span class="metric-badge ${gMoM >= 0 ? 'badge-up' : 'badge-down'}">
            ${fmt.pct(gMoM, 1)}
          </span>
        </td>
      </tr>
    `;
  });

  tbody.innerHTML = html;
}

// Interactive Metal Portfolio Calculator
function calculatePortfolio() {
  if (!state.forecastData) return;

  const last = state.forecastData[state.forecastData.length - 1];
  const first = state.forecastData[0];
  const tax = state.includeTax ? TAX_MULTIPLIER : 1.0;

  const gQty = parseFloat(state.portfolio.goldQty) || 0;
  const sQty = parseFloat(state.portfolio.silverQty) || 0;

  // Convert inputs to Troy Ounces
  const goldOz = state.portfolio.goldUnit === 'g' ? gQty / OZ_TO_G : gQty;
  const silverOz = state.portfolio.silverUnit === 'kg' ? (sQty * 1000) / OZ_TO_G : sQty;

  const goldNowUsd = Number(first.gold_now);
  const silverNowUsd = Number(first.silver_now);
  const inrNow = state.healthData?.latest_usdinr ? Number(state.healthData.latest_usdinr) : Number(first.usdinr);

  const goldEndUsd = Number(last.gold_usd);
  const silverEndUsd = Number(last.silver_usd);
  const inrEnd = Number(last.usdinr);

  const goldEndP10 = Number(last.gold_usd_p10);
  const goldEndP90 = Number(last.gold_usd_p90);
  const silverEndP10 = Number(last.silver_usd_p10);
  const silverEndP90 = Number(last.silver_usd_p90);

  // Present & Projected Total Portfolio Values
  const nowTotalUsd = (goldOz * goldNowUsd) + (silverOz * silverNowUsd);
  const endTotalUsd = (goldOz * goldEndUsd) + (silverOz * silverEndUsd);

  const nowTotalInr = nowTotalUsd * inrNow * tax;
  const endTotalInr = endTotalUsd * inrEnd * tax;

  const endP10Inr = ((goldOz * goldEndP10) + (silverOz * silverEndP10)) * inrEnd * tax;
  const endP90Inr = ((goldOz * goldEndP90) + (silverOz * silverEndP90)) * inrEnd * tax;

  const endP10Usd = (goldOz * goldEndP10) + (silverOz * silverEndP10);
  const endP90Usd = (goldOz * goldEndP90) + (silverOz * silverEndP90);

  const plPct = nowTotalUsd > 0 ? ((endTotalUsd - nowTotalUsd) / nowTotalUsd) * 100 : 0;
  const plUsd = endTotalUsd - nowTotalUsd;
  const plInr = endTotalInr - nowTotalInr;

  // Output To DOM
  const elBigVal = document.getElementById('calc-projected-val');
  const elPl = document.getElementById('calc-pl-val');
  const elCurrent = document.getElementById('calc-curr-val');
  const elFloor = document.getElementById('calc-floor-val');
  const elCeil = document.getElementById('calc-ceil-val');

  if (state.currency === 'INR') {
    elBigVal.textContent = fmt.inr(endTotalInr);
    elPl.textContent = `${plInr >= 0 ? '+' : ''}${fmt.inr(plInr)} (${fmt.pct(plPct)})`;
    elPl.style.color = plInr >= 0 ? 'var(--accent-green)' : 'var(--accent-red)';
    elCurrent.textContent = fmt.inr(nowTotalInr);
    elFloor.textContent = fmt.inr(endP10Inr);
    elCeil.textContent = fmt.inr(endP90Inr);
  } else {
    elBigVal.textContent = fmt.usd(endTotalUsd, 0);
    elPl.textContent = `${plUsd >= 0 ? '+' : ''}${fmt.usd(plUsd, 0)} (${fmt.pct(plPct)})`;
    elPl.style.color = plUsd >= 0 ? 'var(--accent-green)' : 'var(--accent-red)';
    elCurrent.textContent = fmt.usd(nowTotalUsd, 0);
    elFloor.textContent = fmt.usd(endP10Usd, 0);
    elCeil.textContent = fmt.usd(endP90Usd, 0);
  }
}

// Export Full Forecast Data to CSV
function exportCSV() {
  if (!state.forecastData || state.forecastData.length === 0) {
    showToast('No forecast data to export.', 'error');
    return;
  }

  const tax = state.includeTax ? TAX_MULTIPLIER : 1.0;
  const headers = [
    'Date',
    'Gold_USD_Expected',
    'Gold_USD_P10_Bear',
    'Gold_USD_P90_Bull',
    'Gold_INR_10g',
    'Silver_USD_Expected',
    'Silver_USD_P10_Bear',
    'Silver_USD_P90_Bull',
    'Silver_INR_kg',
    'USD_INR_Exchange_Rate',
    'Gold_Silver_Ratio'
  ];

  const rows = state.forecastData.map(d => {
    const goldUsd = Number(d.gold_usd);
    const silvUsd = Number(d.silver_usd);
    const fx = Number(d.usdinr);
    const goldInr = Math.round((goldUsd / OZ_TO_G) * fx * 10 * tax);
    const silvInr = Math.round((silvUsd / OZ_TO_G) * fx * 1000 * tax);
    const ratio = (goldUsd / silvUsd).toFixed(2);

    return [
      `"${d.date}"`,
      goldUsd.toFixed(2),
      Number(d.gold_usd_p10).toFixed(2),
      Number(d.gold_usd_p90).toFixed(2),
      goldInr,
      silvUsd.toFixed(2),
      Number(d.silver_usd_p10).toFixed(2),
      Number(d.silver_usd_p90).toFixed(2),
      silvInr,
      fx.toFixed(4),
      ratio
    ].join(',');
  });

  const csvContent = 'data:text/csv;charset=utf-8,' + [headers.join(','), ...rows].join('\n');
  const encodedUri = encodeURI(csvContent);
  const link = document.createElement('a');
  link.setAttribute('href', encodedUri);
  link.setAttribute('download', `MetalForecasting_Forecast_${state.horizon}M_${new Date().toISOString().slice(0, 10)}.csv`);
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);

  showToast(`Exported ${state.horizon}-month forecast CSV successfully.`);
}

// Render SHAP Feature Importance Table in Modal
function renderShapTable() {
  const tbody = document.getElementById('shap-tbody');
  if (!tbody) return;

  const maxVal = Math.max(...shapData.map(s => Math.max(s.normal, s.crisis)));

  tbody.innerHTML = shapData.map(s => {
    const normalPct = (s.normal / maxVal) * 100;
    const crisisPct = (s.crisis / maxVal) * 100;
    const deltaSign = s.delta >= 0 ? '+' : '';

    return `
      <tr>
        <td>
          <div style="font-weight: 700; color: var(--text-main);">${s.feature}</div>
          <div style="font-size: 0.72rem; color: var(--text-subtle);">${s.desc}</div>
        </td>
        <td class="shap-bar-cell">
          <div style="display: flex; align-items: center; justify-content: space-between; font-size: 0.72rem; margin-bottom: 2px;">
            <span style="color: var(--gold-light);">Normal</span>
            <span class="td-mono">${(s.normal * 100).toFixed(2)}%</span>
          </div>
          <div class="shap-bar-wrap">
            <div class="shap-bar-fill normal" style="width: ${normalPct}%;"></div>
          </div>
        </td>
        <td class="shap-bar-cell">
          <div style="display: flex; align-items: center; justify-content: space-between; font-size: 0.72rem; margin-bottom: 2px;">
            <span style="color: var(--accent-red);">Crisis</span>
            <span class="td-mono">${(s.crisis * 100).toFixed(2)}%</span>
          </div>
          <div class="shap-bar-wrap">
            <div class="shap-bar-fill crisis" style="width: ${crisisPct}%;"></div>
          </div>
        </td>
        <td class="td-mono" style="color: ${s.delta >= 0 ? 'var(--accent-red)' : 'var(--accent-green)'};">
          ${deltaSign}${(s.delta * 100).toFixed(2)}%
        </td>
      </tr>
    `;
  }).join('');
}

// Event Listeners & Binding
function bindEvents() {
  // Horizon Chips
  document.querySelectorAll('.chip-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('.chip-btn').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      const val = parseInt(btn.dataset.months, 10);
      setHorizon(val);
      runForecast(val);
    });
  });

  // Horizon Stepper & Slider
  const slider = document.getElementById('horizon-slider');
  const numInput = document.getElementById('horizon-input');
  const decBtn = document.getElementById('btn-dec-month');
  const incBtn = document.getElementById('btn-inc-month');

  if (slider) {
    slider.addEventListener('input', (e) => {
      setHorizon(parseInt(e.target.value, 10));
    });
    slider.addEventListener('change', (e) => {
      runForecast(parseInt(e.target.value, 10));
    });
  }

  if (numInput) {
    numInput.addEventListener('change', (e) => {
      let val = parseInt(e.target.value, 10);
      if (isNaN(val)) val = 6;
      val = Math.max(1, Math.min(36, val));
      setHorizon(val);
      runForecast(val);
    });
    numInput.addEventListener('keydown', (e) => {
      if (e.key === 'Enter') {
        numInput.blur();
      }
    });
  }

  if (decBtn) {
    decBtn.addEventListener('click', () => {
      const val = Math.max(1, state.horizon - 1);
      setHorizon(val);
      runForecast(val);
    });
  }

  if (incBtn) {
    incBtn.addEventListener('click', () => {
      const val = Math.min(36, state.horizon + 1);
      setHorizon(val);
      runForecast(val);
    });
  }

  // Run Forecast Button
  const runBtn = document.getElementById('run-btn');
  if (runBtn) {
    runBtn.addEventListener('click', () => runForecast(state.horizon));
  }

  // Currency Mode Toggle
  document.querySelectorAll('.cur-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('.cur-btn').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      state.currency = btn.dataset.cur;
      renderSummaryCards();
      calculatePortfolio();
    });
  });

  // Domestic Indian GST / Import Duty Toggle
  const taxToggle = document.getElementById('tax-toggle');
  if (taxToggle) {
    taxToggle.addEventListener('change', (e) => {
      state.includeTax = e.target.checked;
      renderSummaryCards();
      renderDomesticCharts();
      renderDataTable();
      calculatePortfolio();
      showToast(state.includeTax 
        ? 'Applied ~12% Domestic Import Duty & GST to INR quotes.' 
        : 'Reverted to International Parity (ex-duty) INR quotes.');
    });
  }

  // Tab Switcher
  document.querySelectorAll('.tab-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
      document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));

      btn.classList.add('active');
      const targetId = btn.dataset.tab;
      const targetContent = document.getElementById(targetId);
      if (targetContent) targetContent.classList.add('active');

      state.activeTab = targetId;
      window.dispatchEvent(new Event('resize'));
    });
  });

  // Portfolio Calculator Inputs
  const calcGoldQty = document.getElementById('calc-gold-qty');
  const calcGoldUnit = document.getElementById('calc-gold-unit');
  const calcSilvQty = document.getElementById('calc-silv-qty');
  const calcSilvUnit = document.getElementById('calc-silv-unit');

  if (calcGoldQty) {
    calcGoldQty.addEventListener('input', (e) => {
      state.portfolio.goldQty = e.target.value;
      calculatePortfolio();
    });
  }
  if (calcGoldUnit) {
    calcGoldUnit.addEventListener('change', (e) => {
      state.portfolio.goldUnit = e.target.value;
      calculatePortfolio();
    });
  }
  if (calcSilvQty) {
    calcSilvQty.addEventListener('input', (e) => {
      state.portfolio.silverQty = e.target.value;
      calculatePortfolio();
    });
  }
  if (calcSilvUnit) {
    calcSilvUnit.addEventListener('change', (e) => {
      state.portfolio.silverUnit = e.target.value;
      calculatePortfolio();
    });
  }

  // Export CSV Button
  const exportBtn = document.getElementById('btn-export-csv');
  if (exportBtn) {
    exportBtn.addEventListener('click', exportCSV);
  }

  // Modal Controls
  const modalOpenBtn = document.getElementById('btn-methodology');
  const modalCloseBtn = document.getElementById('modal-close');
  const modal = document.getElementById('methodology-modal');

  if (modalOpenBtn && modal) {
    modalOpenBtn.addEventListener('click', () => {
      renderShapTable();
      modal.classList.add('open');
    });
  }

  if (modalCloseBtn && modal) {
    modalCloseBtn.addEventListener('click', () => {
      modal.classList.remove('open');
    });
  }

  if (modal) {
    modal.addEventListener('click', (e) => {
      if (e.target === modal) modal.classList.remove('open');
    });
  }

  // Regime Console Toggle Buttons
  const btnRegimeNormal = document.getElementById('btn-regime-normal');
  const btnRegimeCrisis = document.getElementById('btn-regime-crisis');
  const viewNormal = document.getElementById('state-normal-view');
  const viewCrisis = document.getElementById('state-crisis-view');
  const regimeBadge = document.getElementById('regime-status-badge');
  const regimeLabel = document.getElementById('current-regime-status');
  const stressLabel = document.getElementById('stress-level-label');
  const stressMarker = document.getElementById('stress-marker');

  if (btnRegimeNormal && btnRegimeCrisis) {
    btnRegimeNormal.addEventListener('click', () => {
      btnRegimeNormal.classList.add('active');
      btnRegimeCrisis.classList.remove('active');
      if (viewNormal) viewNormal.style.display = 'flex';
      if (viewCrisis) viewCrisis.style.display = 'none';
      if (regimeBadge) regimeBadge.classList.remove('crisis-mode');
      if (regimeLabel) regimeLabel.textContent = 'NORMAL REGIME ACTIVE';
      if (stressLabel) {
        stressLabel.textContent = 'Normal (VIX ~18.4)';
        stressLabel.style.color = 'var(--accent-green)';
      }
      if (stressMarker) stressMarker.style.left = '28%';
      showToast('Model switched to State 0: Normal Market Expansion profile.');
    });

    btnRegimeCrisis.addEventListener('click', () => {
      btnRegimeCrisis.classList.add('active');
      btnRegimeNormal.classList.remove('active');
      if (viewNormal) viewNormal.style.display = 'none';
      if (viewCrisis) viewCrisis.style.display = 'flex';
      if (regimeBadge) regimeBadge.classList.add('crisis-mode');
      if (regimeLabel) regimeLabel.textContent = 'CRISIS REGIME ACTIVE';
      if (stressLabel) {
        stressLabel.textContent = 'Crisis Alert (VIX > 35)';
        stressLabel.style.color = 'var(--accent-red)';
      }
      if (stressMarker) stressMarker.style.left = '82%';
      showToast('Model switched to State 1: Crisis Turbulence profile.', 'error');
    });
  }

  window.addEventListener('keydown', (e) => {
    if (e.key === 'Escape' && modal && modal.classList.contains('open')) {
      modal.classList.remove('open');
    }
  });
}

// Synchronize Horizon Inputs
function setHorizon(val) {
  state.horizon = val;
  const slider = document.getElementById('horizon-slider');
  const numInput = document.getElementById('horizon-input');
  if (slider) slider.value = val;
  if (numInput) numInput.value = val;

  document.querySelectorAll('.chip-btn').forEach(btn => {
    btn.classList.toggle('active', parseInt(btn.dataset.months, 10) === val);
  });
}

// Initial App Boot Sequence
window.addEventListener('DOMContentLoaded', async () => {
  bindEvents();
  renderShapTable();

  // 1. Fetch benchmark health info
  await fetchHealth();

  // 2. Automatically launch default 6-month forecast
  await runForecast(6);
});
