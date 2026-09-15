/**
 * GenWealth AI — Vanilla JS SPA
 * File: frontend/app.js
 *
 * Architecture:
 *   - API_BASE: auto-detects host (same-origin in production, localhost:8000 in dev)
 *   - Modules:
 *       TabManager      — tab switching
 *       StatusPoller    — /api/v1/health polling on load
 *       SignalGauge     — animated Canvas arc gauge
 *       PriceChart      — TradingView Lightweight Charts (live yfinance via Yahoo)
 *       AdvisoryStream  — SSE consumer for /api/v1/advisor/analyze
 *       PortfolioUI     — /api/v1/portfolio/allocate + Chart.js doughnut
 *       SimulationUI    — /api/v1/simulate/dynamic + walkforward + equity curve
 *       ChatDock        — SSE consumer for /api/v1/chat/stream
 *
 * All modules are IIFE-isolated to prevent namespace pollution.
 * ES6+: async/await, fetch, EventSource-style manual SSE, arrow functions.
 */

'use strict';

// ============================================================
// Config
// ============================================================
const API_BASE = (() => {
  if (window.GENWEALTH_API_URL) return window.GENWEALTH_API_URL.replace(/\/+$/, '');
  try {
    const saved = localStorage.getItem('genwealth_api_url');
    if (saved) return saved.replace(/\/+$/, '');
  } catch (_) {}
  const { protocol, hostname, port } = window.location;
  // If served via FastAPI on any port other than 80/443, use same origin
  return `${protocol}//${hostname}${port ? ':' + port : ''}`;
})();

// ============================================================
// Utilities
// ============================================================
const $ = (sel, ctx = document) => ctx.querySelector(sel);
const $$ = (sel, ctx = document) => [...ctx.querySelectorAll(sel)];
const el = (id) => document.getElementById(id);

const fmt = {
  pct:   (v) => v == null ? '—' : (v >= 0 ? '+' : '') + v.toFixed(2) + '%',
  num:   (v, d = 4) => v == null ? '—' : Number(v).toFixed(d),
  price: (v, sym = '$') => v == null ? '—' : `${sym}${Number(v).toLocaleString(undefined, {minimumFractionDigits: 2, maximumFractionDigits: 4})}`,
  sig:   (v) => v == null ? '—' : Number(v).toFixed(4),
};

const colourFor = (v) => v >= 0.65 ? '#00ff88' : v <= 0.35 ? '#ff4466' : '#ffaa00';
const labelFor  = (v) => v >= 0.65 ? 'BULLISH'  : v <= 0.35 ? 'BEARISH'  : 'NEUTRAL';

function setClass(el, cls, condition) {
  el.classList.toggle(cls, condition);
}

// SSE Manual Reader — fetch + ReadableStream to handle text/event-stream
async function* sseReader(response) {
  const reader  = response.body.getReader();
  const decoder = new TextDecoder('utf-8');
  let   buffer  = '';

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split('\n');
    buffer = lines.pop(); // last incomplete line stays in buffer

    let event = null;
    let data  = null;
    for (const line of lines) {
      if (line.startsWith('event: ')) event = line.slice(7).trim();
      else if (line.startsWith('data: ')) {
        data = line.slice(6).trim();
      } else if (line === '') {
        if (event && data !== null) {
          try { yield { event, data: JSON.parse(data) }; }
          catch { yield { event, data }; }
          event = null; data = null;
        }
      }
    }
  }
}

// ============================================================
// 0. Hero Stock Canvas Animation (Pure HTML5 Canvas / Math)
// ============================================================
const HeroStockAnimation = (() => {
  let canvas = null;
  let ctx = null;
  let animId = null;
  let isRunning = false;
  let width = 0;
  let height = 0;
  let step = 0;

  // Candlesticks data
  const candles = [];
  const numCandles = 22;

  function initCandles() {
    candles.length = 0;
    for (let i = 0; i < numCandles; i++) {
      candles.push({
        xRatio: i / (numCandles - 1),
        open: 0.4 + Math.random() * 0.3,
        close: 0.4 + Math.random() * 0.3,
        width: 10 + Math.random() * 6,
        pulseSpeed: 0.02 + Math.random() * 0.03,
        phase: Math.random() * Math.PI * 2,
      });
    }
  }

  function resize() {
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    width = rect.width;
    height = rect.height;
    if (width === 0 || height === 0) return;
    canvas.width = width * dpr;
    canvas.height = height * dpr;
    ctx.scale(dpr, dpr);
  }

  function draw() {
    if (!isRunning || !ctx) return;
    step += 0.015;

    ctx.clearRect(0, 0, width, height);

    // 1. Subtle Financial Grid Lines
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.03)';
    ctx.lineWidth = 1;
    const gridRows = 6;
    for (let r = 1; r < gridRows; r++) {
      const y = (height / gridRows) * r;
      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.lineTo(width, y);
      ctx.stroke();
    }
    const gridCols = 10;
    for (let c = 1; c < gridCols; c++) {
      const x = (width / gridCols) * c;
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x, height);
      ctx.stroke();
    }

    // 2. Render Floating Glowing Candlesticks
    candles.forEach((c) => {
      const cx = c.xRatio * width;
      const wave = Math.sin(step * 1.5 + c.phase) * 12;
      const baseH = height * 0.68 + wave;
      
      const openY = baseH + (c.open - 0.5) * 60;
      const closeY = baseH + (c.close - 0.5) * 60;
      const highY = Math.min(openY, closeY) - 14 - Math.sin(step * 2 + c.phase) * 5;
      const lowY = Math.max(openY, closeY) + 14 + Math.cos(step * 2 + c.phase) * 5;

      const isBull = closeY < openY;
      const candleColor = isBull ? '#00ffa6' : '#ff4d6d';

      // Wick
      ctx.strokeStyle = candleColor;
      ctx.lineWidth = 1.5;
      ctx.globalAlpha = 0.35;
      ctx.beginPath();
      ctx.moveTo(cx, highY);
      ctx.lineTo(cx, lowY);
      ctx.stroke();

      // Body
      ctx.globalAlpha = 0.25;
      ctx.fillStyle = candleColor;
      const topY = Math.min(openY, closeY);
      const bodyH = Math.max(6, Math.abs(closeY - openY));
      ctx.fillRect(cx - c.width / 2, topY, c.width, bodyH);

      // Border with neon glow
      ctx.globalAlpha = 0.6;
      ctx.strokeRect(cx - c.width / 2, topY, c.width, bodyH);
    });

    ctx.globalAlpha = 1.0;

    // 3. Multi-Layer Animated Trend Curves (Lower half)
    // Trend 1: Cyan Bull Wave
    drawTrendWave({
      color: '#00f0ff',
      alpha: 0.75,
      fillAlpha: 0.08,
      freq: 0.0035,
      speed: 1.8,
      offsetY: height * 0.70,
      amp: 36,
      noiseFreq: 0.012,
      noiseAmp: 16,
    });

    // Trend 2: Emerald Momentum Wave
    drawTrendWave({
      color: '#00ffa6',
      alpha: 0.6,
      fillAlpha: 0.04,
      freq: 0.0042,
      speed: 1.2,
      offsetY: height * 0.64,
      amp: 28,
      noiseFreq: 0.018,
      noiseAmp: 12,
    });

    // Trend 3: Violet Deep Cycle Wave
    drawTrendWave({
      color: '#b57edc',
      alpha: 0.4,
      fillAlpha: 0.02,
      freq: 0.0028,
      speed: 0.8,
      offsetY: height * 0.76,
      amp: 42,
      noiseFreq: 0.008,
      noiseAmp: 14,
    });

    animId = requestAnimationFrame(draw);
  }

  function drawTrendWave(opts) {
    const points = [];
    const numPoints = 50;
    const dx = width / (numPoints - 1);

    for (let i = 0; i < numPoints; i++) {
      const x = i * dx;
      const mainWave = Math.sin(x * opts.freq + step * opts.speed) * opts.amp;
      const noise = Math.cos(x * opts.noiseFreq + step * (opts.speed * 0.6)) * opts.noiseAmp;
      const y = opts.offsetY + mainWave + noise;
      points.push({ x, y });
    }

    // Path
    ctx.beginPath();
    ctx.moveTo(points[0].x, points[0].y);
    for (let i = 1; i < points.length - 1; i++) {
      const xc = (points[i].x + points[i + 1].x) / 2;
      const yc = (points[i].y + points[i + 1].y) / 2;
      ctx.quadraticCurveTo(points[i].x, points[i].y, xc, yc);
    }
    const last = points[points.length - 1];
    ctx.lineTo(last.x, last.y);

    // Stroke
    ctx.strokeStyle = opts.color;
    ctx.globalAlpha = opts.alpha;
    ctx.lineWidth = 2.5;
    ctx.stroke();

    // Area Fill
    ctx.lineTo(width, height);
    ctx.lineTo(0, height);
    ctx.closePath();
    ctx.fillStyle = opts.color;
    ctx.globalAlpha = opts.fillAlpha;
    ctx.fill();

    // Occasional glowing node on peaks
    ctx.globalAlpha = opts.alpha;
    for (let i = 5; i < points.length; i += 12) {
      ctx.beginPath();
      ctx.arc(points[i].x, points[i].y, 3.5, 0, Math.PI * 2);
      ctx.fillStyle = '#ffffff';
      ctx.fill();
      ctx.beginPath();
      ctx.arc(points[i].x, points[i].y, 7, 0, Math.PI * 2);
      ctx.strokeStyle = opts.color;
      ctx.lineWidth = 1.5;
      ctx.stroke();
    }

    ctx.globalAlpha = 1.0;
  }

  function start() {
    if (!canvas) {
      canvas = el('hero-stock-canvas');
      if (!canvas) return;
      ctx = canvas.getContext('2d');
      initCandles();
      window.addEventListener('resize', resize);
    }
    resize();
    if (!isRunning) {
      isRunning = true;
      animId = requestAnimationFrame(draw);
    }
  }

  function stop() {
    isRunning = false;
    if (animId) {
      cancelAnimationFrame(animId);
      animId = null;
    }
  }

  return { start, stop, resize };
})();

// ============================================================
// 1. Tab Manager
// ============================================================
const TabManager = (() => {
  const panels = {
    home:       'panel-home',
    research:   'panel-research',
    portfolio:  'panel-portfolio',
    simulation: 'panel-simulation',
    about:      'panel-about',
  };

  function activate(tabId) {
    $$('.tab-btn').forEach(b => {
      const active = b.dataset.tab === tabId;
      b.classList.toggle('active', active);
      b.setAttribute('aria-selected', active);
    });
    Object.entries(panels).forEach(([key, panelId]) => {
      const p = el(panelId);
      if (p) p.classList.toggle('hidden', key !== tabId);
    });

    // Control Hero Stock Canvas Animation
    if (tabId === 'home') {
      HeroStockAnimation.start();
    } else {
      HeroStockAnimation.stop();
    }

    // Auto-close floating chatbot when switching tabs to prevent screen obstruction
    if (typeof ChatDock !== 'undefined' && ChatDock.close) {
      ChatDock.close();
    }

    // Scroll smoothly to top
    window.scrollTo({ top: 0, behavior: 'smooth' });
  }

  function init() {
    $$('.tab-btn').forEach(btn => {
      btn.addEventListener('click', () => activate(btn.dataset.tab));
    });

    // Header Logo clicks return to Home
    const logo = el('header-logo');
    if (logo) logo.addEventListener('click', () => activate('home'));

    // Wire Home CTA buttons
    const btnLaunch = el('btn-launch-workshop');
    if (btnLaunch) btnLaunch.addEventListener('click', () => activate('research'));

    const btnHeroAbout = el('btn-hero-about');
    if (btnHeroAbout) btnHeroAbout.addEventListener('click', () => activate('about'));

    // Wire Workshop tool cards
    $$('.workshop-tool-card').forEach(card => {
      card.addEventListener('click', () => {
        const target = card.dataset.launch;
        if (target) activate(target);
      });
    });

    // Wire About page workshop button
    const btnAboutWorkshop = el('btn-about-to-workshop');
    if (btnAboutWorkshop) btnAboutWorkshop.addEventListener('click', () => activate('research'));

    // Initial activation
    activate('home');
    HeroStockAnimation.start();

    // Translucent Glass Theme Cycler: Cyber-Glass -> Aurora Matrix -> Pastel Sky White
    const themes = [
      { cls: 'theme-cyber',  icon: '✨', name: 'Cyber-Glass Obsidian' },
      { cls: 'theme-aurora', icon: '🌌', name: 'Aurora Matrix Glass' },
      { cls: 'theme-pastel', icon: '🌤️', name: 'Pastel Sky White' },
    ];
    let curThemeIdx = 0;
    document.body.classList.add(themes[0].cls);
    el('theme-toggle').textContent = themes[0].icon;
    el('theme-toggle').title = `Active Theme: ${themes[0].name} (Click to switch)`;

    el('theme-toggle').addEventListener('click', () => {
      document.body.classList.remove(themes[curThemeIdx].cls);
      curThemeIdx = (curThemeIdx + 1) % themes.length;
      document.body.classList.add(themes[curThemeIdx].cls);
      el('theme-toggle').textContent = themes[curThemeIdx].icon;
      el('theme-toggle').title = `Active Theme: ${themes[curThemeIdx].name} (Click to switch)`;
    });
  }

  return { init, activate };
})();


// ============================================================
// 2. Status Poller
// ============================================================
const StatusPoller = (() => {
  async function poll() {
    try {
      const res = await fetch(`${API_BASE}/api/v1/health`);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const h = await res.json();

      const dot   = el('status-dot');
      const label = el('status-label');
      const uptimeBadge = el('uptime-badge');

      if (dot) dot.className = 'status-dot ' + (h.status === 'ok' ? 'ok' : h.status === 'degraded' ? 'degraded' : 'error');
      if (label) label.textContent = `API ${h.status.toUpperCase()} · LSTM ${h.lstm_ok ? '✓' : '✗'} · RF ${h.rf_ok ? '✓' : '✗'} · MongoDB ${h.mongodb}`;
      if (uptimeBadge) uptimeBadge.textContent = `↑ ${Math.floor(h.uptime_s / 60)}m ${Math.floor(h.uptime_s % 60)}s`;
    } catch (e) {
      if (el('status-dot')) el('status-dot').className = 'status-dot error';
      if (el('status-label')) el('status-label').textContent = 'API Offline';
    }
  }

  return { poll };
})();


// ============================================================
// 3. Signal Gauge (Canvas Arc)
// ============================================================
const SignalGauge = (() => {
  let canvas, ctx, lastValue = null;

  function draw(value) {
    if (!canvas) { canvas = el('signal-gauge-canvas'); ctx = canvas.getContext('2d'); }
    const W = canvas.width, H = canvas.height;
    ctx.clearRect(0, 0, W, H);

    const cx = W / 2, cy = H - 20;
    const r  = Math.min(W, H * 2) / 2 - 16;
    const startAngle = Math.PI;
    const endAngle   = 2 * Math.PI;
    const valAngle   = startAngle + (value ?? 0.5) * Math.PI;
    const colour     = colourFor(value ?? 0.5);

    // Background arc
    ctx.beginPath();
    ctx.arc(cx, cy, r, startAngle, endAngle);
    ctx.lineWidth = 14;
    ctx.strokeStyle = 'rgba(255,255,255,0.06)';
    ctx.lineCap = 'round';
    ctx.stroke();

    // Coloured value arc
    ctx.beginPath();
    ctx.arc(cx, cy, r, startAngle, valAngle);
    ctx.lineWidth = 14;
    ctx.strokeStyle = colour;
    ctx.lineCap = 'round';
    ctx.shadowColor = colour;
    ctx.shadowBlur = 12;
    ctx.stroke();
    ctx.shadowBlur = 0;

    // Tick marks at BEARISH / NEUTRAL / BULLISH boundaries
    [[0.35, '#ff4466'], [0.5, '#ffaa00'], [0.65, '#00ff88']].forEach(([v, c]) => {
      const a = startAngle + v * Math.PI;
      const x1 = cx + (r - 10) * Math.cos(a), y1 = cy + (r - 10) * Math.sin(a);
      const x2 = cx + (r + 2)  * Math.cos(a), y2 = cy + (r + 2)  * Math.sin(a);
      ctx.beginPath();
      ctx.moveTo(x1, y1); ctx.lineTo(x2, y2);
      ctx.strokeStyle = c; ctx.lineWidth = 2; ctx.stroke();
    });

    // Needle
    const nx = cx + (r - 4) * Math.cos(valAngle);
    const ny = cy + (r - 4) * Math.sin(valAngle);
    ctx.beginPath();
    ctx.moveTo(cx, cy); ctx.lineTo(nx, ny);
    ctx.strokeStyle = 'white'; ctx.lineWidth = 2.5;
    ctx.shadowColor = 'white'; ctx.shadowBlur = 6;
    ctx.stroke();
    ctx.shadowBlur = 0;

    // Animate label
    lastValue = value;
  }

  function update(value, label) {
    draw(value);
    el('gauge-value').textContent = value != null ? (value * 100).toFixed(1) : '—';
    const lbl = el('gauge-label');
    lbl.textContent = label ?? labelFor(value ?? 0.5);
    lbl.className = `gauge-label ${labelFor(value ?? 0.5).toLowerCase()}`;
  }

  return { update, draw };
})();


// ============================================================
// 4. Price Chart (TradingView Lightweight Charts via Backend API)
// ============================================================
const PriceChart = (() => {
  let chart = null;
  let series = null;
  let currentTicker = 'NVDA';
  let currentPeriod = '30d';
  let currentChartType = 'candlestick'; // 'candlestick' | 'line'
  let cachedCandles = [];
  let cachedPayload = null;

  async function load(ticker, period = currentPeriod, chartType = currentChartType) {
    if (ticker) currentTicker = ticker.trim().toUpperCase();
    currentPeriod = period || currentPeriod;
    currentChartType = chartType || currentChartType;

    const container = el('price-chart-container');
    const placeholder = el('chart-placeholder');
    placeholder.classList.remove('hidden');
    placeholder.textContent = `Fetching market data for ${currentTicker} (${currentPeriod.toUpperCase()})…`;

    try {
      const resp = await fetch(`${API_BASE}/api/v1/market/history?ticker=${encodeURIComponent(currentTicker)}&period=${currentPeriod}`);
      if (!resp.ok) {
        const errJson = await resp.json().catch(() => ({}));
        throw new Error(errJson.detail || `HTTP ${resp.status}`);
      }
      const data = await resp.json();
      cachedPayload = data;
      cachedCandles = data.candles || [];

      if (cachedCandles.length === 0) {
        placeholder.textContent = `No historical price bars available for ${currentTicker}`;
        return;
      }

      // Hide placeholder
      placeholder.classList.add('hidden');

      // Update Header Labels & Badges
      el('chart-ticker-label').textContent = data.ticker;
      if (el('currency-badge')) el('currency-badge').textContent = data.currency || 'USD';
      if (el('exchange-badge')) el('exchange-badge').textContent = data.exchange || 'Market';

      const sym = data.symbol || '$';
      const delta = data.change;
      const deltaPct = data.change_pct;
      const sign = delta >= 0 ? '+' : '';
      el('chart-price-label').innerHTML =
        `<span class="${delta >= 0 ? 'positive' : 'negative'}">`
        + `${sym}${Number(data.current_price).toLocaleString(undefined, {minimumFractionDigits:2, maximumFractionDigits:2})} `
        + `${sign}${sym}${Math.abs(delta).toFixed(2)} (${sign}${deltaPct}%)</span>`;

      render();
    } catch (err) {
      console.warn('[PriceChart] Load error:', err);
      placeholder.classList.remove('hidden');
      placeholder.textContent = `Unable to load market data for ${currentTicker}: ${err.message}`;
    }
  }

  function render() {
    const container = el('price-chart-container');
    if (!container || cachedCandles.length === 0) return;

    if (chart) {
      chart.remove();
      chart = null;
      series = null;
    }

    chart = LightweightCharts.createChart(container, {
      width:  container.clientWidth,
      height: 280,
      layout: {
        background: { type: 'solid', color: 'transparent' },
        textColor: '#94a9cc',
        fontFamily: "'Inter', system-ui, sans-serif",
      },
      grid: {
        vertLines: { color: 'rgba(255, 255, 255, 0.04)' },
        horzLines: { color: 'rgba(255, 255, 255, 0.04)' },
      },
      crosshair: {
        mode: LightweightCharts.CrosshairMode.Normal,
        vertLine: { color: 'rgba(0, 240, 255, 0.4)', width: 1, style: 2 },
        horzLine: { color: 'rgba(0, 240, 255, 0.4)', width: 1, style: 2 },
      },
      rightPriceScale: {
        borderColor: 'rgba(255, 255, 255, 0.1)',
        scaleMargins: { top: 0.12, bottom: 0.12 },
      },
      timeScale: {
        borderColor: 'rgba(255, 255, 255, 0.1)',
        timeVisible: true,
      },
    });

    if (currentChartType === 'candlestick') {
      series = chart.addCandlestickSeries({
        upColor:         '#00ff88',
        downColor:       '#ff3366',
        borderUpColor:   '#00ff88',
        borderDownColor: '#ff3366',
        wickUpColor:     '#00ff88',
        wickDownColor:   '#ff3366',
      });
      series.setData(cachedCandles.map(c => ({
        time:  c.time,
        open:  c.open,
        high:  c.high,
        low:   c.low,
        close: c.close,
      })));
    } else {
      series = chart.addAreaSeries({
        topColor:    'rgba(0, 240, 255, 0.32)',
        bottomColor: 'rgba(0, 240, 255, 0.01)',
        lineColor:   '#00f0ff',
        lineWidth:   2,
      });
      series.setData(cachedCandles.map(c => ({
        time:  c.time,
        value: c.close,
      })));
    }

    chart.timeScale().fitContent();

    new ResizeObserver(() => {
      if (chart && container) chart.applyOptions({ width: container.clientWidth });
    }).observe(container);
  }

  function setType(type) {
    if (type === currentChartType) return;
    currentChartType = type;
    $$('.chart-btn').forEach(btn => btn.classList.toggle('active', btn.dataset.type === type));
    render();
  }

  function setPeriod(period) {
    if (period === currentPeriod) return;
    currentPeriod = period;
    $$('.period-btn').forEach(btn => btn.classList.toggle('active', btn.dataset.period === period));
    load(currentTicker, currentPeriod, currentChartType);
  }

  function init() {
    // Chart type buttons (Candles vs Line)
    $$('.chart-btn').forEach(btn => {
      btn.addEventListener('click', () => setType(btn.dataset.type));
    });

    // Lookback period buttons (7D, 30D, 90D, 1Y)
    $$('.period-btn').forEach(btn => {
      btn.addEventListener('click', () => setPeriod(btn.dataset.period));
    });
  }

  return { init, load, setType, setPeriod };
})();


// ============================================================
// 5. Advisory Stream (SSE consumer)
// ============================================================
const AdvisoryStream = (() => {
  let currencySymbol = '$';

  function setStage(id, state) {
    const el_s = el(id);
    if (!el_s) return;
    ['active', 'done', 'error'].forEach(c => el_s.classList.remove(c));
    if (typeof state === 'string') {
      const s = state.trim();
      if (s && (s === 'active' || s === 'done' || s === 'error')) {
        el_s.classList.add(s);
      }
    }
  }

  function renderPhase1(p1) {
    // Resolve currency symbol from currency code
    const currencyMap = {
      'INR': '₹', 'EUR': '€', 'JPY': '¥', 'GBP': '£',
      'HKD': 'HK$', 'AUD': 'A$', 'CAD': 'C$', 'SGD': 'S$',
      'KRW': '₩', 'CNY': '¥', 'CHF': 'Fr', 'SEK': 'kr',
      'NOK': 'kr', 'NZD': 'NZ$', 'BRL': 'R$', 'ILS': '₪',
      'ZAR': 'R', 'MXN': 'M$', 'USD': '$',
    };
    const currency = p1.currency || 'USD';
    currencySymbol = currencyMap[currency] || '$';

    el('currency-badge').textContent  = currency;
    el('exchange-badge').textContent  = p1.exchange || el('exchange-badge').textContent || 'Markets';
    el('inference-badge').textContent = p1.inference_mode || '—';

    SignalGauge.update(p1.phase1_signal, p1.signal_label);

    // Metric chips
    el('rf-prob').textContent   = p1.rf_prob   != null ? (p1.rf_prob   * 100).toFixed(1) + '%' : '—';
    el('lstm-prob').textContent = p1.lstm_prob  != null ? (p1.lstm_prob * 100).toFixed(1) + '%' : '—';
    el('sent-val').textContent  = p1.sentiment  != null ? p1.sentiment.toFixed(3) : '—';
    el('vol-ratio').textContent = p1.vol_ratio  != null ? p1.vol_ratio.toFixed(2)  : '—';

    // Quant feature stats
    el('as-of-date').textContent = p1.as_of_date || '—';
    el('s-close').textContent  = p1.close_price != null ? `${currencySymbol}${Number(p1.close_price).toLocaleString(undefined, {minimumFractionDigits: 2, maximumFractionDigits: 4})}` : '—';
    el('s-logret').textContent = fmt.pct(p1.log_ret != null ? p1.log_ret * 100 : null);
    el('s-vol20').textContent  = p1.vol_20 != null ? (p1.vol_20 * 100).toFixed(2) + '%' : '—';
    el('s-ret1m').textContent  = fmt.pct(p1.ret_1m != null ? p1.ret_1m * 100 : null);
    el('s-eff').textContent    = p1.efficiency != null ? p1.efficiency.toFixed(4) : '—';
    el('s-shock').textContent  = p1.vol_ratio  != null ? p1.vol_ratio.toFixed(3) : '—';

    // Colour sign for returns
    ['s-logret', 's-ret1m'].forEach(id => {
      const val_el = el(id);
      const v = parseFloat(val_el.textContent);
      val_el.className = `stat-val ${v >= 0 ? 'positive' : 'negative'}`;
    });

    // News headlines list (if provided)
    if (p1.news_headlines?.length) {
      const ragSec = el('rag-section');
      const ragDocs = el('rag-docs');
      ragSec.classList.remove('hidden');
      ragDocs.innerHTML = p1.news_headlines.slice(0, 3).map(h =>
        `<div class="rag-doc">
          <div>
            <div class="rag-doc-title">📰 ${h}</div>
            <div class="rag-doc-source">Live News</div>
          </div>
        </div>`
      ).join('');
    }
  }

  function renderRAGDocs(docs) {
    if (!Array.isArray(docs) || !docs.length) return;
    const ragSec = el('rag-section');
    const ragDocs = el('rag-docs');
    ragSec.classList.remove('hidden');
    ragDocs.innerHTML = docs.map(d =>
      `<div class="rag-doc">
        <div style="flex:1">
          <div class="rag-doc-title">${d.title || 'Document'}</div>
          <div class="rag-doc-source">${d.source || 'Knowledge Base'}</div>
          <div style="font-size:0.76rem; color:var(--text-muted); margin-top:3px;">${(d.snippet || '').slice(0, 200)}…</div>
        </div>
        <div class="rag-doc-score">Sim: ${(d.similarity_score || 0).toFixed(3)}</div>
      </div>`
    ).join('');
  }

  async function run(ticker, useLiveEngine) {
    // Reset UI
    el('report-placeholder').classList.add('hidden');
    const reportContent = el('report-content');
    reportContent.classList.remove('hidden');
    reportContent.innerHTML = '';
    el('compliance-block').classList.add('hidden');
    el('rag-section').classList.add('hidden');
    el('stream-status').textContent = '⟳ Streaming…';
    el('stream-status').className = 'stream-status streaming';
    el('latency-badge').textContent = '';

    ['stage-phase1','stage-phase2','stage-rag','stage-report'].forEach(id => setStage(id, null));

    // Load price chart concurrently
    PriceChart.load(ticker).catch(console.warn);

    try {
      const resp = await fetch(`${API_BASE}/api/v1/advisor/analyze`, {
        method:  'POST',
        headers: { 'Content-Type': 'application/json' },
        body:    JSON.stringify({ ticker, use_live_engine: useLiveEngine }),
      });

      if (!resp.ok) {
        throw new Error(`HTTP ${resp.status}: ${await resp.text()}`);
      }

      for await (const { event, data } of sseReader(resp)) {
        switch (event) {
          case 'phase1':
            setStage('stage-phase1', 'active');
            renderPhase1(data);
            setStage('stage-phase1', 'done');
            break;

          case 'phase2':
            setStage('stage-phase2', 'active');
            // Phase2 summary in stats could go here if needed
            setStage('stage-phase2', 'done');
            break;

          case 'rag':
            setStage('stage-rag', 'active');
            let ragList = data;
            if (typeof ragList === 'string') {
              try { ragList = JSON.parse(ragList); } catch {}
            }
            if (Array.isArray(ragList)) {
              renderRAGDocs(ragList);
            }
            setStage('stage-rag', 'done');
            break;

          case 'report':
            setStage('stage-report', 'active');
            reportContent.innerHTML = marked.parse(data.content || '');
            break;

          case 'complete':
            setStage('stage-report', 'done');
            el('stream-status').textContent = '✓ Report complete';
            el('stream-status').className = 'stream-status';
            el('latency-badge').textContent = `${data.latency_s}s`;
            // Show compliance block
            if (data.compliance) {
              const cb = el('compliance-block');
              cb.classList.remove('hidden');
              el('compliance-text').textContent =
                `${data.compliance.flags_detected} compliance flag(s) detected. Disclaimer appended. `
                + (data.compliance.flag_details?.join(' ') || '');
            }
            break;

          case 'error':
            setStage('stage-report', 'error');
            reportContent.innerHTML = `<div style="color:var(--accent-red)">⚠ ${data.error}</div>`;
            el('stream-status').textContent = '✗ Error';
            el('stream-status').className = 'stream-status';
            break;
        }
      }
    } catch (err) {
      reportContent.innerHTML = `<div style="color:var(--accent-red)">⚠ Connection error: ${err.message}</div>`;
      el('stream-status').textContent = '✗ Disconnected';
      el('stream-status').className = 'stream-status';
    }
  }

  function init() {
    el('analyze-btn').addEventListener('click', () => {
      const ticker = el('ticker-input').value.trim().toUpperCase();
      if (!ticker) { el('ticker-input').focus(); return; }
      const useLive = el('live-engine-toggle').checked;

      // Update exchange/currency badge before loading
      el('exchange-badge').textContent = '…';
      el('currency-badge').textContent = '…';

      el('analyze-btn').disabled = true;
      el('analyze-btn-text').classList.add('hidden');
      el('analyze-loader').classList.remove('hidden');

      run(ticker, useLive).finally(() => {
        el('analyze-btn').disabled = false;
        el('analyze-btn-text').classList.remove('hidden');
        el('analyze-loader').classList.add('hidden');
      });
    });

    el('ticker-input').addEventListener('keydown', (e) => {
      if (e.key === 'Enter') el('analyze-btn').click();
    });
  }

  return { init };
})();


// ============================================================
// 6. Portfolio Allocator UI
// ============================================================
const PortfolioUI = (() => {
  let pieChart = null;

  const PALETTE = [
    '#00d4ff','#00ff88','#ffaa00','#ff4466','#9b6eff',
    '#00aaff','#66ff44','#ff8800','#ff44aa','#44aaff',
  ];

  function renderPie(allocations) {
    const canvas = el('allocation-pie-canvas');
    const placeholder = el('alloc-chart-placeholder');

    if (allocations.length === 0) { placeholder.classList.remove('hidden'); return; }
    placeholder.classList.add('hidden');

    if (pieChart) { pieChart.destroy(); pieChart = null; }

    pieChart = new Chart(canvas, {
      type: 'doughnut',
      data: {
        labels: allocations.map(a => a.ticker),
        datasets: [{
          data:             allocations.map(a => parseFloat((a.weight * 100).toFixed(2))),
          backgroundColor:  allocations.map((_, i) => PALETTE[i % PALETTE.length] + '99'),
          borderColor:      allocations.map((_, i) => PALETTE[i % PALETTE.length]),
          borderWidth: 2,
        }],
      },
      options: {
        responsive: true,
        cutout: '65%',
        plugins: {
          legend: { labels: { color: '#8fa3c7', font: { family: 'Inter', size: 12 } } },
          tooltip: {
            callbacks: {
              label: ctx => ` ${ctx.label}: ${ctx.parsed.toFixed(1)}%`,
            },
          },
        },
      },
    });
  }

  function renderTable(allocations, currency, cashReserved) {
    const STANCE_COLOUR = { BUY: 'positive', HOLD: 'neutral', SELL: 'negative', REDUCE: 'negative' };
    const sym = currency === 'INR' ? '₹' : currency === 'EUR' ? '€' : '$';

    const tbody = el('allocation-tbody');
    tbody.innerHTML = allocations.map((a, i) => `
      <tr>
        <td style="color:${PALETTE[i % PALETTE.length]};font-weight:700">${a.ticker}</td>
        <td class="${colourFor(a.signal) === '#00ff88' ? 'positive' : colourFor(a.signal) === '#ff4466' ? 'negative' : 'neutral'}">${(a.signal * 100).toFixed(1)}%</td>
        <td class="${STANCE_COLOUR[a.stance] || ''}">${a.stance}</td>
        <td style="font-weight:600">${(a.weight * 100).toFixed(2)}%</td>
        <td>${sym}${Number(a.capital).toLocaleString(undefined, {minimumFractionDigits: 0, maximumFractionDigits: 0})}</td>
      </tr>
    `).join('');

    // Cash row
    const cashPct = allocations.reduce((s, a) => s + a.weight, 0);
    tbody.innerHTML += `
      <tr style="border-top:1px solid rgba(255,255,255,0.1)">
        <td style="color:var(--text-muted)">💵 CASH</td>
        <td>—</td><td>—</td>
        <td style="color:var(--text-muted)">${((1 - cashPct) * 100).toFixed(1)}%</td>
        <td style="color:var(--text-muted)">${sym}${Number(cashReserved).toLocaleString()}</td>
      </tr>`;
  }

  async function run() {
    const raw = el('portfolio-tickers').value;
    const tickers = raw.split(/[,\s]+/).map(t => t.trim().toUpperCase()).filter(Boolean);
    const capital  = parseFloat(el('portfolio-capital').value) || 100000;
    const currency = el('portfolio-currency').value;

    if (!tickers.length) { alert('Enter at least one ticker.'); return; }

    el('allocate-btn').disabled = true;
    el('allocate-btn-text').classList.add('hidden');
    el('allocate-loader').classList.remove('hidden');

    try {
      const resp = await fetch(`${API_BASE}/api/v1/portfolio/allocate`, {
        method:  'POST',
        headers: { 'Content-Type': 'application/json' },
        body:    JSON.stringify({ tickers, capital, currency }),
      });
      if (!resp.ok) throw new Error(await resp.text());
      const data = await resp.json();

      // Render pie
      renderPie(data.allocations);

      // Render table
      renderTable(data.allocations, data.currency, data.cash_reserved);

      // Summary badges
      el('alloc-method-badge').textContent = data.allocation_method;
      const sym = currency === 'INR' ? '₹' : currency === 'EUR' ? '€' : '$';
      el('alloc-capital-summary').textContent =
        `${sym}${Number(data.total_capital).toLocaleString()} total · Cash ${data.cash_pct.toFixed(1)}% reserved`;

      // PPO metrics
      if (data.ppo_metrics) {
        const pm = data.ppo_metrics;
        el('ppo-metrics').classList.remove('hidden');
        el('ppo-strategy').textContent  = pm.strategy;
        el('ppo-return').textContent    = fmt.pct(pm.total_return_pct);
        el('ppo-sharpe').textContent    = pm.sharpe_ratio.toFixed(3);
        el('ppo-drawdown').textContent  = fmt.pct(pm.max_drawdown_pct);
        el('ppo-method').textContent    = pm.allocation_method;
      }

    } catch (err) {
      alert(`Portfolio allocation failed: ${err.message}`);
    } finally {
      el('allocate-btn').disabled = false;
      el('allocate-btn-text').classList.remove('hidden');
      el('allocate-loader').classList.add('hidden');
    }
  }

  function init() {
    el('allocate-btn').addEventListener('click', run);
  }

  return { init };
})();


// ============================================================
// 7. Simulation UI (30-Day Dynamic & Custom Date Range Backtest)
// ============================================================
const SimulationUI = (() => {
  let equityChart = null;
  let currentSimType = 'dynamic'; // 'dynamic' | 'custom'
  const cachedResults = { dynamic: null, custom: null };

  const fmtDate = (d) => d.toISOString().split('T')[0];

  // Hidden background date constraint manager & presets
  function initDateConstraints() {
    const today = new Date();
    const maxDateStr = fmtDate(today);

    const startInput = el('custom-start');
    const endInput   = el('custom-end');
    if (!startInput || !endInput) return;

    // Constraint 1: Maximum date is today (cannot backtest future days)
    startInput.max = maxDateStr;
    endInput.max   = maxDateStr;

    // Default: 30 calendar days lookback ending yesterday/today
    setPresetDays(30);

    // Constraint 2: End date must be strictly after start date (min 1-2 days)
    const handleStartChange = () => {
      const sVal = startInput.value;
      if (!sVal) return;
      const sDate = new Date(sVal);
      const minEnd = new Date(sDate);
      minEnd.setDate(minEnd.getDate() + 1);
      endInput.min = fmtDate(minEnd);

      // Auto-correct if end <= start
      if (endInput.value && new Date(endInput.value) <= sDate) {
        const nextDay = new Date(sDate);
        nextDay.setDate(nextDay.getDate() + 15);
        if (nextDay > today) nextDay.setTime(today.getTime());
        endInput.value = fmtDate(nextDay);
      }
      syncPresetHighlightWithDates();
    };

    const handleEndChange = () => {
      const sVal = startInput.value;
      const eVal = endInput.value;
      if (sVal && eVal && new Date(eVal) <= new Date(sVal)) {
        // Auto-correct start date backwards
        const prevDate = new Date(eVal);
        prevDate.setDate(prevDate.getDate() - 15);
        startInput.value = fmtDate(prevDate);
      }
      syncPresetHighlightWithDates();
    };

    startInput.addEventListener('change', handleStartChange);
    startInput.addEventListener('input', handleStartChange);
    endInput.addEventListener('change', handleEndChange);
    endInput.addEventListener('input', handleEndChange);

    // Preset pills handler
    $$('.preset-pill').forEach(pill => {
      pill.addEventListener('click', () => {
        $$('.preset-pill').forEach(p => p.classList.remove('active'));
        pill.classList.add('active');
        if (pill.dataset.days === 'custom') {
          startInput.focus();
          return;
        }
        const days = parseInt(pill.dataset.days, 10) || 30;
        setPresetDays(days);
      });
    });
  }

  function setPresetDays(numDays) {
    const today = new Date();
    const end = new Date(today);
    // If weekend, snap to Friday
    if (end.getDay() === 0) end.setDate(end.getDate() - 2);
    else if (end.getDay() === 6) end.setDate(end.getDate() - 1);

    const start = new Date(end);
    start.setDate(start.getDate() - numDays);

    el('custom-start').value = fmtDate(start);
    el('custom-end').value   = fmtDate(end);
    el('custom-end').min     = fmtDate(new Date(start.getTime() + 86400000));

    // Highlight matching preset pill
    $$('.preset-pill').forEach(p => {
      if (p.dataset.days !== 'custom' && parseInt(p.dataset.days, 10) === numDays) {
        p.classList.add('active');
      } else {
        p.classList.remove('active');
      }
    });

    updateDaysBadge();
  }

  function syncPresetHighlightWithDates() {
    const sVal = el('custom-start').value;
    const eVal = el('custom-end').value;
    if (!sVal || !eVal) return;

    const s = new Date(sVal);
    const e = new Date(eVal);
    const diffTime = e.getTime() - s.getTime();
    const diffDays = Math.max(1, Math.round(diffTime / (1000 * 60 * 60 * 24)));

    // Check if diff matches any fixed preset
    let matchedPill = null;
    $$('.preset-pill').forEach(pill => {
      pill.classList.remove('active');
      if (pill.dataset.days !== 'custom' && parseInt(pill.dataset.days, 10) === diffDays) {
        matchedPill = pill;
      }
    });

    if (matchedPill) {
      matchedPill.classList.add('active');
    } else {
      // Dates are custom - activate Custom pill
      const customPill = el('preset-custom');
      if (customPill) customPill.classList.add('active');
    }

    updateDaysBadge();
  }

  function updateDaysBadge() {
    const sVal = el('custom-start').value;
    const eVal = el('custom-end').value;
    const badge = el('range-days-count');
    if (!sVal || !eVal || !badge) return;

    const s = new Date(sVal);
    const e = new Date(eVal);
    const diffTime = e.getTime() - s.getTime();
    const diffDays = Math.max(1, Math.round(diffTime / (1000 * 60 * 60 * 24)));
    const estTradingDays = Math.max(1, Math.round(diffDays * 5 / 7));
    badge.textContent = `🗓️ ${diffDays} Calendar Days (~${estTradingDays} Trading Days)`;
  }

  function renderEquityCurve(curve, simType, sym = '$') {
    const canvas      = el('equity-curve-canvas');
    const placeholder = el('equity-placeholder');

    // Always clean up previous chart first to prevent chart ghosting
    if (equityChart) {
      equityChart.destroy();
      equityChart = null;
    }

    if (!curve?.length) {
      placeholder.classList.remove('hidden');
      return;
    }
    placeholder.classList.add('hidden');

    const isDynamic = simType === 'DYNAMIC';

    // Format labels: Dynamic -> 'Day 0', 'Day 5'.. | Custom -> Date / Day
    const labels = curve.map(([step]) => {
      if (typeof step === 'number') {
        return isDynamic ? `Day ${step * 5}` : `Day ${step}`;
      }
      // If it's a date string like '2024-08-15', format to 'Aug 15'
      const str = String(step).slice(0, 10);
      const parts = str.split('-');
      if (parts.length === 3) {
        const d = new Date(str + 'T00:00:00');
        if (!isNaN(d.getTime())) {
          return d.toLocaleDateString(undefined, { month: 'short', day: 'numeric' });
        }
      }
      return str;
    });

    const values = curve.map(([_, v]) => v);

    // Color theme separation: Dynamic = Cyan | Custom = Emerald/Mint
    const strokeColor = isDynamic ? '#00f0ff' : '#00ffa6';
    const fillColor   = isDynamic ? 'rgba(0, 240, 255, 0.09)' : 'rgba(0, 255, 166, 0.09)';

    equityChart = new Chart(canvas, {
      type: 'line',
      data: {
        labels,
        datasets: [{
          label:           isDynamic ? '30-Day Dynamic Portfolio' : 'Custom Regime Portfolio',
          data:            values,
          borderColor:     strokeColor,
          backgroundColor: fillColor,
          borderWidth:     2.5,
          fill:            true,
          tension:         0.35,
          pointRadius:     labels.length > 40 ? 2 : 4,
          pointHoverRadius:6,
          pointBackgroundColor: strokeColor,
        }],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        animation: { duration: 600, easing: 'easeOutQuart' },
        plugins: {
          legend: { display: false },
          tooltip: {
            callbacks: {
              label: ctx => ` ${sym}${Number(ctx.parsed.y).toLocaleString(undefined, {minimumFractionDigits: 2, maximumFractionDigits: 2})}`,
            },
          },
        },
        scales: {
          x: {
            ticks: { color: '#8fa3c7', maxRotation: 45, autoSkip: true, maxTicksLimit: 12 },
            grid:  { color: 'rgba(255,255,255,0.05)' },
          },
          y: {
            ticks: { color: '#8fa3c7', callback: v => sym + Number(v).toLocaleString() },
            grid:  { color: 'rgba(255,255,255,0.05)' },
          },
        },
      },
    });
  }

  const ACTION_CLASS = {
    BUY: 'action-buy', TRIM_PROFIT: 'action-trim', STOP_LOSS_EXIT: 'action-stop',
    STOP_LOSS: 'action-stop', VOL_EXIT: 'action-stop', CLOSE_ALL: 'action-close', REINVEST: 'action-reinvest',
  };

  function renderLedger(ledger) {
    const tbody = el('ledger-tbody');
    if (!ledger?.length) {
      tbody.innerHTML = '<tr><td colspan="7" class="table-empty">No trades recorded</td></tr>';
      return;
    }
    tbody.innerHTML = ledger.map(row => {
      const pnl = parseFloat(row.realized_pnl) || 0;
      return `<tr>
        <td>${row.step}</td>
        <td>${row.date?.slice(0, 10) || '—'}</td>
        <td style="font-weight:600">${row.ticker}</td>
        <td class="${ACTION_CLASS[row.action] || ''}">${row.action}</td>
        <td>${Number(row.exec_price).toFixed(2)}</td>
        <td class="${pnl >= 0 ? 'pnl-positive' : 'pnl-negative'}">${pnl >= 0 ? '+' : ''}${pnl.toFixed(2)}</td>
        <td>${(Number(row.signal || 0.5) * 100).toFixed(1)}%</td>
      </tr>`;
    }).join('');
  }

  function applySimulationResult(data) {
    if (!data) return;

    // Render equity curve with distinct styling
    renderEquityCurve(data.portfolio_curve, data.simulation_type, data.symbol || '$');

    // KPI cards
    el('kpi-grid').classList.remove('hidden');
    const roiEl = el('kpi-roi');
    roiEl.textContent = fmt.pct(data.total_roi_pct);
    roiEl.className   = `kpi-value ${data.total_roi_pct >= 0 ? 'positive' : 'danger'}`;
    el('kpi-winrate').textContent  = fmt.pct(data.win_rate_pct);
    el('kpi-drawdown').textContent = fmt.pct(data.max_drawdown_pct);
    el('kpi-trades').textContent   = data.n_trades;

    // Equity chart caption
    el('equity-caption').textContent =
      `Final: ${data.symbol || '$'}${Number(data.final_cash + data.final_positions_value).toLocaleString(undefined, {minimumFractionDigits: 2, maximumFractionDigits: 2})} `
      + `(PnL: ${data.total_realized_pnl >= 0 ? '+' : ''}${Number(data.total_realized_pnl).toLocaleString(undefined, {minimumFractionDigits: 2, maximumFractionDigits: 2})})`;

    // Ledger
    renderLedger(data.ledger);
    el('ledger-summary').textContent = `${data.n_trades} trades · Latency: ${data.latency_s}s`;
  }

  function clearViewForMode(mode) {
    if (cachedResults[mode]) {
      applySimulationResult(cachedResults[mode]);
    } else {
      if (equityChart) { equityChart.destroy(); equityChart = null; }
      el('equity-placeholder').classList.remove('hidden');
      el('equity-placeholder').textContent = mode === 'dynamic'
        ? "Run a 30-day dynamic simulation to generate the equity curve"
        : "Select custom dates and click 'Run Simulation' to generate the equity curve";
      el('kpi-grid').classList.add('hidden');
      el('equity-caption').textContent = '';
      el('ledger-summary').textContent = '';
      el('ledger-tbody').innerHTML = '<tr><td colspan="7" class="table-empty">No simulation run yet for this mode</td></tr>';
    }
  }

  async function run() {
    const tickers = el('sim-tickers').value.split(/[,\s]+/).map(t => t.trim().toUpperCase()).filter(Boolean);
    const capital = parseFloat(el('sim-capital').value) || 100000;

    if (!tickers.length) { alert('Enter at least one ticker.'); return; }

    el('run-sim-btn').disabled = true;
    el('sim-btn-text').classList.add('hidden');
    el('sim-loader').classList.remove('hidden');

    try {
      let data;
      if (currentSimType === 'dynamic') {
        const resp = await fetch(`${API_BASE}/api/v1/simulate/dynamic`, {
          method:  'POST',
          headers: { 'Content-Type': 'application/json' },
          body:    JSON.stringify({ tickers, capital }),
        });
        if (!resp.ok) throw new Error(await resp.text());
        data = await resp.json();
      } else {
        // Custom Date Range Backtest
        const sVal = el('custom-start').value;
        const eVal = el('custom-end').value;
        if (!sVal || !eVal) throw new Error('Please select both start and end dates.');

        const resp = await fetch(`${API_BASE}/api/v1/simulate/walkforward`, {
          method:  'POST',
          headers: { 'Content-Type': 'application/json' },
          body:    JSON.stringify({
            tickers, capital,
            regime_start: sVal,
            regime_end:   eVal,
          }),
        });
        if (!resp.ok) throw new Error(await resp.text());
        data = await resp.json();
      }

      // Cache result separately for this mode
      cachedResults[currentSimType] = data;
      applySimulationResult(data);

    } catch (err) {
      alert(`Simulation failed: ${err.message}`);
    } finally {
      el('run-sim-btn').disabled = false;
      el('sim-btn-text').classList.remove('hidden');
      el('sim-loader').classList.add('hidden');
    }
  }

  function init() {
    initDateConstraints();

    // Sim type toggle: 30-Day Dynamic vs Custom Date Range
    $$('.sim-type-btn').forEach(btn => {
      btn.addEventListener('click', () => {
        $$('.sim-type-btn').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        currentSimType = btn.dataset.simtype;

        // Toggle custom date range panel
        el('custom-dates-box').classList.toggle('hidden', currentSimType !== 'custom');

        // Restore or clear chart specifically for this mode
        clearViewForMode(currentSimType);
      });
    });

    el('run-sim-btn').addEventListener('click', run);
  }

  return { init };
})();


// ============================================================
// 8. Chat Dock (SSE & Translucent Glass Workstation)
// ============================================================
const ChatDock = (() => {
  let isOpen = false;
  let isExpanded = false;
  let conversationHistory = [];
  let activeTicker = '';

  function open()  {
    el('chat-dock').classList.remove('hidden');
    isOpen = true;
    el('chat-dot').classList.remove('visible');
  }

  function close() {
    el('chat-dock').classList.add('hidden');
    isOpen = false;
  }

  function toggleExpand() {
    isExpanded = !isExpanded;
    el('chat-dock').classList.toggle('expanded', isExpanded);
    const expandBtn = el('chat-expand-btn');
    if (expandBtn) {
      expandBtn.textContent = isExpanded ? '⤡' : '⤢';
      expandBtn.title = isExpanded ? 'Contract Chat' : 'Expand Chat Workstation';
    }
  }

  function appendMessage(role, content, isTyping = false) {
    const msgs = el('chat-messages');
    const div  = document.createElement('div');
    div.className = `chat-msg ${role}${isTyping ? ' chat-typing' : ''}`;

    const bubble = document.createElement('div');
    bubble.className = 'msg-bubble';
    if (!isTyping) bubble.innerHTML = marked.parse(content);
    div.appendChild(bubble);

    msgs.appendChild(div);
    msgs.scrollTop = msgs.scrollHeight;
    return { div, bubble };
  }

  function updateActiveTicker() {
    // Sync ticker from research tab
    const t = el('ticker-input').value.trim().toUpperCase();
    if (t) activeTicker = t;
  }

  async function sendMessage(message) {
    if (!message.trim()) return;

    updateActiveTicker();

    // User message in UI
    appendMessage('user', message);
    conversationHistory.push({ role: 'user', content: message });

    // Typing indicator
    const { div: typingDiv, bubble: typingBubble } = appendMessage('assistant', '', true);

    try {
      const resp = await fetch(`${API_BASE}/api/v1/chat/stream`, {
        method:  'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          message,
          ticker: activeTicker || null,
          conversation_history: conversationHistory.slice(-10),
        }),
      });

      if (!resp.ok) throw new Error(`HTTP ${resp.status}`);

      // Remove typing indicator and start streaming
      typingDiv.classList.remove('chat-typing');
      typingBubble.innerHTML = '';
      let fullReply = '';

      for await (const { event, data } of sseReader(resp)) {
        if (event === 'token') {
          fullReply += data.content;
          typingBubble.innerHTML = marked.parse(fullReply);
          el('chat-messages').scrollTop = el('chat-messages').scrollHeight;
        } else if (event === 'done') {
          conversationHistory.push({ role: 'assistant', content: fullReply });
          // Cap history at 20 turns
          if (conversationHistory.length > 20) conversationHistory = conversationHistory.slice(-20);
          break;
        } else if (event === 'error') {
          typingBubble.innerHTML = `<span style="color:var(--accent-red)">⚠ ${data.content}</span>`;
          break;
        }
      }
    } catch (err) {
      typingDiv.classList.remove('chat-typing');
      typingBubble.innerHTML = `<span style="color:var(--accent-red)">⚠ Connection error: ${err.message}</span>`;
    }
  }

  function init() {
    el('chat-fab').addEventListener('click', () => isOpen ? close() : open());
    el('chat-close-btn').addEventListener('click', close);
    if (el('chat-expand-btn')) {
      el('chat-expand-btn').addEventListener('click', toggleExpand);
    }

    el('chat-send-btn').addEventListener('click', () => {
      const msg = el('chat-input').value.trim();
      el('chat-input').value = '';
      if (msg) sendMessage(msg);
    });

    el('chat-input').addEventListener('keydown', (e) => {
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        el('chat-send-btn').click();
      }
    });

    // Show notification dot when chat is closed and a new analysis completes
    document.addEventListener('advisory-complete', () => {
      if (!isOpen) el('chat-dot').classList.add('visible');
    });
  }

  return { init, open, close, toggleExpand };
})();


// ============================================================
// 9. Info / Architecture Reference Modal
// ============================================================
const InfoModal = (() => {
  function open(defaultSection = 'section-phase1') {
    el('info-modal-backdrop').classList.remove('hidden');
    switchSection(defaultSection);
  }

  function close() {
    el('info-modal-backdrop').classList.add('hidden');
  }

  function switchSection(sectionId) {
    const isPhase1 = sectionId === 'section-phase1';
    el('modal-tab-phase1').classList.toggle('active', isPhase1);
    el('modal-tab-engine').classList.toggle('active', !isPhase1);
    el('section-phase1').classList.toggle('hidden', !isPhase1);
    el('section-engine').classList.toggle('hidden', isPhase1);
  }

  function init() {
    // Wire up inline popover click and hover toggles + parent card stacking elevation
    const popoverWrappers = document.querySelectorAll('.info-popover-wrapper');

    function setParentElevation(wrapper, elevate) {
      const parentCard = wrapper.closest('.glass-card');
      const searchRow = wrapper.closest('.search-row');
      if (parentCard) parentCard.classList.toggle('popover-open', elevate);
      if (searchRow) searchRow.classList.toggle('popover-open', elevate);
    }

    popoverWrappers.forEach((wrapper) => {
      // Elevate parent stacking order on mouse hover so popover floats in front of all boxes
      wrapper.addEventListener('mouseenter', () => setParentElevation(wrapper, true));
      wrapper.addEventListener('mouseleave', () => {
        if (!wrapper.classList.contains('active')) {
          setParentElevation(wrapper, false);
        }
      });

      const btn = wrapper.querySelector('.info-help-btn');
      if (btn) {
        btn.addEventListener('click', (e) => {
          e.stopPropagation();
          const wasActive = wrapper.classList.contains('active');
          // Close all open popovers first
          popoverWrappers.forEach(w => {
            w.classList.remove('active');
            setParentElevation(w, false);
          });
          // Toggle this one
          if (!wasActive) {
            wrapper.classList.add('active');
            setParentElevation(wrapper, true);
          }
        });
      }
    });

    // Close any active inline popovers when clicking elsewhere
    document.addEventListener('click', (e) => {
      if (!e.target.closest('.info-popover-wrapper')) {
        popoverWrappers.forEach(w => {
          w.classList.remove('active');
          setParentElevation(w, false);
        });
      }
    });

    if (el('info-modal-close')) {
      el('info-modal-close').addEventListener('click', close);
    }
    if (el('info-modal-backdrop')) {
      el('info-modal-backdrop').addEventListener('click', (e) => {
        if (e.target === el('info-modal-backdrop')) close();
      });
    }
    if (el('modal-tab-phase1')) {
      el('modal-tab-phase1').addEventListener('click', () => switchSection('section-phase1'));
    }
    if (el('modal-tab-engine')) {
      el('modal-tab-engine').addEventListener('click', () => switchSection('section-engine'));
    }
    // Update live engine badge when checkbox changes
    if (el('live-engine-toggle')) {
      el('live-engine-toggle').addEventListener('change', (e) => {
        const isOpen = e.target.checked;
        const badge = el('engine-state-badge');
        if (badge) {
          badge.textContent = isOpen ? 'OPEN' : 'CLOSED';
          badge.className = `engine-state-badge${isOpen ? '' : ' closed'}`;
        }
      });
    }
  }

  return { init, open, close, switchSection };
})();

window.InfoModal = InfoModal;
window.ChatDock = ChatDock;
window.PriceChart = PriceChart;
window.TabManager = TabManager;


// ============================================================
// BOOT
// ============================================================
document.addEventListener('DOMContentLoaded', async () => {
  // Initial gauge render (neutral)
  SignalGauge.draw(0.5);

  // Initialise all modules
  TabManager.init();
  PriceChart.init();
  AdvisoryStream.init();
  PortfolioUI.init();
  SimulationUI.init();
  ChatDock.init();
  InfoModal.init();

  // Health check
  await StatusPoller.poll();

  // Auto-load example ticker on first visit
  const savedTicker = sessionStorage.getItem('gw_ticker') || 'NVDA';
  el('ticker-input').value = savedTicker;

  // Persist ticker across tab switches
  el('ticker-input').addEventListener('change', () => {
    sessionStorage.setItem('gw_ticker', el('ticker-input').value.trim().toUpperCase());
  });

  // Automatically load initial 30-day candlestick chart for default ticker
  PriceChart.load(savedTicker, '30d', 'candlestick').catch(console.warn);
});
