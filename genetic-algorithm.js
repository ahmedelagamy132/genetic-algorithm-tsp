/* ============================================================================
   TSP · Evolutionary Router
   A genetic algorithm solving the Travelling Salesman Problem.

   This file has two halves:
     1. A pure GA core (RNG, operators, the GA class) with no DOM access, so it
        can be unit-tested in Node.
     2. A browser app (rendering + control wiring) booted on DOMContentLoaded.
   ============================================================================ */

/* ----------------------------------------------------------------------------
   Geography
   Cities live in unit space [0,1] x [0,1] and are scaled to kilometres only
   for the readouts, so the map is resolution independent and distances stay
   human-readable.
---------------------------------------------------------------------------- */
const MAP_SPAN_KM = 3200;

// Deterministic RNG (mulberry32) so a given seed always yields the same map.
// A fair tune-and-compare needs the cities held fixed between runs.
function makeRng(seed) {
  let a = seed >>> 0;
  return function () {
    a |= 0; a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function makeCities(n, seed) {
  const rng = makeRng(seed);
  const pad = 0.06;
  const cities = [];
  for (let i = 0; i < n; i++) {
    cities.push({ x: pad + rng() * (1 - 2 * pad), y: pad + rng() * (1 - 2 * pad) });
  }
  return cities;
}

function buildDistanceMatrix(cities) {
  const n = cities.length;
  const d = Array.from({ length: n }, () => new Float64Array(n));
  for (let i = 0; i < n; i++) {
    for (let j = i + 1; j < n; j++) {
      const dx = cities[i].x - cities[j].x;
      const dy = cities[i].y - cities[j].y;
      const km = Math.hypot(dx, dy) * MAP_SPAN_KM;
      d[i][j] = km; d[j][i] = km;
    }
  }
  return d;
}

function tourLength(tour, dist) {
  let total = 0;
  for (let i = 0; i < tour.length; i++) {
    total += dist[tour[i]][tour[(i + 1) % tour.length]];
  }
  return total;
}

/* ----------------------------------------------------------------------------
   Operators
---------------------------------------------------------------------------- */
function shuffledRange(n) {
  const a = Array.from({ length: n }, (_, i) => i);
  for (let i = n - 1; i > 0; i--) {
    const j = (Math.random() * (i + 1)) | 0;
    const t = a[i]; a[i] = a[j]; a[j] = t;
  }
  return a;
}

// Order Crossover (OX): keep a contiguous slice of p1, fill the gaps with p2's
// order. Preserves relative ordering, the trait that matters for a tour.
function orderCrossover(p1, p2) {
  const n = p1.length;
  const a = (Math.random() * n) | 0;
  const b = (Math.random() * n) | 0;
  const lo = Math.min(a, b), hi = Math.max(a, b);
  const child = new Array(n).fill(-1);
  const taken = new Uint8Array(n);
  for (let i = lo; i <= hi; i++) { child[i] = p1[i]; taken[p1[i]] = 1; }
  let idx = (hi + 1) % n;
  for (let k = 0; k < n; k++) {
    const gene = p2[(hi + 1 + k) % n];
    if (!taken[gene]) { child[idx] = gene; taken[gene] = 1; idx = (idx + 1) % n; }
  }
  return child;
}

// Partially Mapped Crossover (PMX): copy a slice of p1, then repair conflicts
// from p2 through the position mapping induced by the slice.
function pmx(p1, p2) {
  const n = p1.length;
  let a = (Math.random() * n) | 0;
  let b = (Math.random() * n) | 0;
  if (a > b) { const t = a; a = b; b = t; }
  const child = new Array(n).fill(-1);
  const placed = new Uint8Array(n);
  for (let i = a; i <= b; i++) { child[i] = p1[i]; placed[p1[i]] = 1; }
  const posInP2 = new Int32Array(n);
  for (let i = 0; i < n; i++) posInP2[p2[i]] = i;
  for (let i = a; i <= b; i++) {
    const g = p2[i];
    if (placed[g]) continue;
    let idx = i;
    while (true) {
      const j = posInP2[p1[idx]];
      if (j < a || j > b) { child[j] = g; placed[g] = 1; break; }
      idx = j;
    }
  }
  for (let i = 0; i < n; i++) { if (child[i] === -1) { child[i] = p2[i]; } }
  return child;
}

// Swap mutation: each city may trade places with another. Per-gene, so its
// disruption scales with the rate.
function swapMutate(tour, rate) {
  const n = tour.length;
  for (let i = 0; i < n; i++) {
    if (Math.random() < rate) {
      const j = (Math.random() * n) | 0;
      const t = tour[i]; tour[i] = tour[j]; tour[j] = t;
    }
  }
  return tour;
}

// Inversion mutation: reverse a sub-segment. Equivalent to an untangling
// 2-opt move, the single most effective local edit for a tour.
function inversionMutate(tour, rate) {
  if (Math.random() < rate) {
    const n = tour.length;
    let a = (Math.random() * n) | 0;
    let b = (Math.random() * n) | 0;
    if (a > b) { const t = a; a = b; b = t; }
    while (a < b) { const t = tour[a]; tour[a] = tour[b]; tour[b] = t; a++; b--; }
  }
  return tour;
}

/* ----------------------------------------------------------------------------
   The genetic algorithm
---------------------------------------------------------------------------- */
class GA {
  constructor(cities, params) {
    this.cities = cities;
    this.dist = buildDistanceMatrix(cities);
    this.params = params; // { pop, mut, tourn, elite, crossover, mutation, crossoverRate }
    this.reset();
  }

  reset() {
    const n = this.cities.length;
    this.generation = 0;
    this.population = [];
    for (let i = 0; i < this.params.pop; i++) this.population.push(shuffledRange(n));
    this.lengths = this.population.map((t) => tourLength(t, this.dist));
    let bi = 0;
    for (let i = 1; i < this.lengths.length; i++) if (this.lengths[i] < this.lengths[bi]) bi = i;
    this.best = { tour: this.population[bi].slice(), length: this.lengths[bi], gen: 0 };
    this.initialBest = this.best.length;
    this.improvedThisStep = false;
    this.stagnant = 0;
    this.history = [{ best: this.best.length, avg: this._avg() }];
  }

  _avg() {
    let s = 0;
    for (let i = 0; i < this.lengths.length; i++) s += this.lengths[i];
    return s / this.lengths.length;
  }

  _tournament() {
    const k = this.params.tourn;
    let best = (Math.random() * this.population.length) | 0;
    let bestLen = this.lengths[best];
    for (let i = 1; i < k; i++) {
      const c = (Math.random() * this.population.length) | 0;
      if (this.lengths[c] < bestLen) { best = c; bestLen = this.lengths[c]; }
    }
    return best;
  }

  step() {
    const p = this.params;
    const newPop = [];

    // Elitism: the fittest individuals cross into the next generation intact.
    if (p.elite > 0) {
      const order = Array.from(this.lengths.keys()).sort((i, j) => this.lengths[i] - this.lengths[j]);
      for (let e = 0; e < p.elite && e < order.length; e++) newPop.push(this.population[order[e]].slice());
    }

    while (newPop.length < p.pop) {
      const a = this._tournament();
      let child;
      if (Math.random() < p.crossoverRate) {
        const b = this._tournament();
        child = p.crossover === 'pmx'
          ? pmx(this.population[a], this.population[b])
          : orderCrossover(this.population[a], this.population[b]);
      } else {
        child = this.population[a].slice();
      }
      if (p.mutation === 'swap') swapMutate(child, p.mut);
      else inversionMutate(child, p.mut);
      newPop.push(child);
    }

    this.population = newPop;
    this.lengths = this.population.map((t) => tourLength(t, this.dist));
    this.generation++;

    let bi = 0;
    for (let i = 1; i < this.lengths.length; i++) if (this.lengths[i] < this.lengths[bi]) bi = i;
    if (this.lengths[bi] < this.best.length) {
      this.best = { tour: this.population[bi].slice(), length: this.lengths[bi], gen: this.generation };
      this.improvedThisStep = true;
      this.stagnant = 0;
    } else {
      this.improvedThisStep = false;
      this.stagnant++;
    }
    this.history.push({ best: this.best.length, avg: this._avg() });
  }
}

/* ============================================================================
   Node test hook. Browser ignores this; `require()` uses it.
   ============================================================================ */
if (typeof module !== 'undefined' && module.exports) {
  module.exports = {
    makeCities, buildDistanceMatrix, tourLength,
    orderCrossover, pmx, swapMutate, inversionMutate, shuffledRange, GA,
  };
}

/* ============================================================================
   Browser application
   ============================================================================ */
if (typeof document !== 'undefined') {
  const THIN = ' ';
  const reduceMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;

  const fmtKm = (x) => Math.round(x).toString().replace(/\B(?=(\d{3})+(?!\d))/g, THIN) + ' km';
  const fmtInt = (x) => Math.round(x).toString().replace(/\B(?=(\d{3})+(?!\d))/g, THIN);

  // ---- DOM ----
  const $ = (id) => document.getElementById(id);
  const stage = $('stage');
  const chart = $('chart');
  const sctx = stage.getContext('2d');
  const cctx = chart.getContext('2d');

  // ---- State ----
  const state = {
    citySeed: 7,
    cities: [],
    ga: null,
    running: false,
    gensPerSec: 12,
    pinned: null, // { best:[], length, label }
    flash: 0,     // 0..1 improvement glow envelope
  };

  const params = {
    n: 22,
    pop: 220,
    mut: 0.22,
    tourn: 5,
    elite: 2,
    crossover: 'ox',
    mutation: 'inversion',
    crossoverRate: 0.95,
  };

  function rebuild(newSeed) {
    if (newSeed !== undefined) state.citySeed = newSeed;
    state.cities = makeCities(params.n, state.citySeed);
    state.ga = new GA(state.cities, {
      pop: params.pop, mut: params.mut, tourn: params.tourn, elite: params.elite,
      crossover: params.crossover, mutation: params.mutation, crossoverRate: params.crossoverRate,
    });
    state.flash = 0;
  }

  // ---- Hi-DPI canvas sizing ----
  function fit(canvas, ctx) {
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const r = canvas.getBoundingClientRect();
    canvas.width = Math.max(1, Math.round(r.width * dpr));
    canvas.height = Math.max(1, Math.round(r.height * dpr));
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    return r;
  }

  // ---- Colour tokens (read from CSS so the JS never hard-codes the palette) ----
  const css = getComputedStyle(document.documentElement);
  const tok = (name, fallback) => (css.getPropertyValue(name).trim() || fallback);
  const COL = {
    line: tok('--line', 'rgba(120,140,170,.5)'),
    lineSoft: tok('--line-soft', 'rgba(120,140,170,.3)'),
    ink: tok('--ink', '#eef'),
    inkDim: tok('--ink-dim', '#9ab'),
    inkFaint: tok('--ink-faint', '#789'),
    champ: tok('--champ', '#d8a24a'),
    champHi: tok('--champ-hi', '#f0c070'),
    accent: tok('--accent', '#5fb8c8'),
    ghost: tok('--ghost', 'rgba(150,170,200,.10)'),
    field: tok('--bg', '#10141c'),
  };

  // ---- Map projection ----
  function project(rect) {
    const m = Math.min(rect.width, rect.height);
    const inset = Math.max(34, m * 0.07);
    const w = rect.width - inset * 2;
    const h = rect.height - inset * 2;
    const side = Math.min(w, h);
    const ox = inset + (w - side) / 2;
    const oy = inset + (h - side) / 2;
    return { ox, oy, side, inset, rect };
  }

  function px(c, P) { return { x: P.ox + c.x * P.side, y: P.oy + c.y * P.side }; }

  // ---- Render the map ----
  function renderMap() {
    const rect = fit(stage, sctx);
    sctx.clearRect(0, 0, rect.width, rect.height);
    const P = project(rect);
    const ga = state.ga;

    drawFrameAndTicks(P);

    // Faint sample of the searching population behind the champion.
    if (!reduceMotion || true) {
      const pop = ga.population;
      const sample = Math.min(20, pop.length);
      sctx.lineWidth = 1;
      sctx.strokeStyle = COL.ghost;
      for (let s = 1; s <= sample; s++) {
        const tour = pop[((s * 9973) % pop.length)];
        drawTour(tour, P, false);
      }
    }

    // Champion tour.
    const champ = ga.best.tour;
    const glow = state.flash;
    sctx.save();
    sctx.lineJoin = 'round';
    sctx.lineCap = 'round';
    sctx.shadowColor = COL.champ;
    sctx.shadowBlur = 10 + glow * 16;
    sctx.strokeStyle = glow > 0.01 ? COL.champHi : COL.champ;
    sctx.lineWidth = 2.4 + glow * 1.4;
    drawTour(champ, P, true);
    sctx.restore();

    drawCities(P);
    drawCompass(P);
  }

  function drawTour(tour, P, closeMarker) {
    const cities = state.cities;
    sctx.beginPath();
    for (let i = 0; i < tour.length; i++) {
      const p = px(cities[tour[i]], P);
      if (i === 0) sctx.moveTo(p.x, p.y); else sctx.lineTo(p.x, p.y);
    }
    sctx.closePath();
    sctx.stroke();
  }

  function drawCities(P) {
    const cities = state.cities;
    const small = cities.length <= 32;
    for (let i = 0; i < cities.length; i++) {
      const p = px(cities[i], P);
      const isStart = state.ga.best.tour[0] === i;
      sctx.beginPath();
      sctx.arc(p.x, p.y, isStart ? 5 : 3.2, 0, Math.PI * 2);
      sctx.fillStyle = isStart ? COL.champ : COL.field;
      sctx.fill();
      sctx.lineWidth = 1.4;
      sctx.strokeStyle = isStart ? COL.champHi : COL.inkDim;
      sctx.stroke();
      if (small && !isStart) {
        sctx.fillStyle = COL.inkFaint;
        sctx.font = '10px ui-monospace, monospace';
        sctx.fillText(String(i), p.x + 6, p.y - 5);
      }
    }
  }

  function drawFrameAndTicks(P) {
    const r = P.rect;
    // border frame
    sctx.strokeStyle = COL.lineSoft;
    sctx.lineWidth = 1;
    sctx.strokeRect(P.inset, P.inset, r.width - P.inset * 2, r.height - P.inset * 2);

    // corner brackets
    const b = 16;
    sctx.strokeStyle = COL.line;
    sctx.lineWidth = 1.5;
    const corners = [
      [P.inset, P.inset, 1, 1], [r.width - P.inset, P.inset, -1, 1],
      [P.inset, r.height - P.inset, 1, -1], [r.width - P.inset, r.height - P.inset, -1, -1],
    ];
    for (const [x, y, sx, sy] of corners) {
      sctx.beginPath();
      sctx.moveTo(x + sx * b, y); sctx.lineTo(x, y); sctx.lineTo(x, y + sy * b);
      sctx.stroke();
    }

    // edge ticks
    sctx.strokeStyle = COL.lineSoft;
    sctx.fillStyle = COL.inkFaint;
    sctx.lineWidth = 1;
    sctx.font = '9px ui-monospace, monospace';
    const divs = 8;
    for (let i = 0; i <= divs; i++) {
      const t = i / divs;
      const x = P.ox + t * P.side;
      const y = P.oy + t * P.side;
      const len = i % 2 === 0 ? 7 : 4;
      // top + bottom
      sctx.beginPath(); sctx.moveTo(x, P.inset); sctx.lineTo(x, P.inset + len); sctx.stroke();
      sctx.beginPath(); sctx.moveTo(x, P.rect.height - P.inset); sctx.lineTo(x, P.rect.height - P.inset - len); sctx.stroke();
      // left + right
      sctx.beginPath(); sctx.moveTo(P.inset, y); sctx.lineTo(P.inset + len, y); sctx.stroke();
      sctx.beginPath(); sctx.moveTo(P.rect.width - P.inset, y); sctx.lineTo(P.rect.width - P.inset - len, y); sctx.stroke();
    }
  }

  function drawCompass(P) {
    const r = P.rect;
    const cx = r.width - P.inset - 30;
    const cy = r.height - P.inset - 30;
    const rad = 18;
    sctx.save();
    sctx.translate(cx, cy);
    sctx.strokeStyle = COL.line;
    sctx.fillStyle = COL.inkFaint;
    sctx.lineWidth = 1;
    sctx.beginPath(); sctx.arc(0, 0, rad, 0, Math.PI * 2); sctx.stroke();
    // needle
    sctx.beginPath();
    sctx.moveTo(0, -rad + 2); sctx.lineTo(4, 0); sctx.lineTo(0, rad - 2); sctx.lineTo(-4, 0); sctx.closePath();
    sctx.fillStyle = COL.champ; sctx.globalAlpha = 0.85; sctx.fill(); sctx.globalAlpha = 1;
    sctx.fillStyle = COL.inkFaint;
    sctx.font = '8px ui-monospace, monospace';
    sctx.textAlign = 'center';
    sctx.fillText('N', 0, -rad - 4);
    sctx.textAlign = 'start';
    sctx.restore();
  }

  // ---- Render the convergence chart ----
  function renderChart() {
    const rect = fit(chart, cctx);
    cctx.clearRect(0, 0, rect.width, rect.height);
    const hist = state.ga.history;
    const padL = 8, padR = 8, padT = 12, padB = 8;
    const w = rect.width - padL - padR;
    const h = rect.height - padT - padB;

    // y range across best + avg + pinned
    let lo = Infinity, hi = -Infinity;
    for (const p of hist) { lo = Math.min(lo, p.best); hi = Math.max(hi, p.avg); }
    if (state.pinned) for (const v of state.pinned.best) { lo = Math.min(lo, v); hi = Math.max(hi, v); }
    if (!isFinite(lo) || lo === hi) { hi = lo + 1; }
    const span = hi - lo;
    const nx = Math.max(hist.length, 2);

    const X = (i) => padL + (i / (nx - 1)) * w;
    const Y = (v) => padT + (1 - (v - lo) / span) * h;

    // baseline grid
    cctx.strokeStyle = COL.lineSoft;
    cctx.lineWidth = 1;
    for (let g = 0; g <= 3; g++) {
      const yy = padT + (g / 3) * h;
      cctx.beginPath(); cctx.moveTo(padL, yy); cctx.lineTo(padL + w, yy); cctx.stroke();
    }

    // pinned reference (dashed)
    if (state.pinned) {
      cctx.save();
      cctx.setLineDash([4, 4]);
      cctx.strokeStyle = COL.inkFaint;
      cctx.lineWidth = 1.25;
      cctx.beginPath();
      const pb = state.pinned.best;
      const pnx = Math.max(pb.length, 2);
      for (let i = 0; i < pb.length; i++) {
        const xx = padL + (i / (pnx - 1)) * w;
        const yy = Y(pb[i]);
        if (i === 0) cctx.moveTo(xx, yy); else cctx.lineTo(xx, yy);
      }
      cctx.stroke();
      cctx.restore();
    }

    // average (cool)
    cctx.strokeStyle = COL.accent;
    cctx.globalAlpha = 0.55;
    cctx.lineWidth = 1.25;
    cctx.beginPath();
    for (let i = 0; i < hist.length; i++) {
      const xx = X(i), yy = Y(hist[i].avg);
      if (i === 0) cctx.moveTo(xx, yy); else cctx.lineTo(xx, yy);
    }
    cctx.stroke();
    cctx.globalAlpha = 1;

    // best (brass)
    cctx.strokeStyle = COL.champ;
    cctx.lineWidth = 2;
    cctx.beginPath();
    for (let i = 0; i < hist.length; i++) {
      const xx = X(i), yy = Y(hist[i].best);
      if (i === 0) cctx.moveTo(xx, yy); else cctx.lineTo(xx, yy);
    }
    cctx.stroke();
  }

  // ---- Readouts ----
  function renderReadouts() {
    const ga = state.ga;
    $('r-gen').textContent = fmtInt(ga.generation);
    $('r-best').textContent = fmtKm(ga.best.length);
    $('r-avg').textContent = fmtKm(ga._avg());
    const improve = (1 - ga.best.length / ga.initialBest) * 100;
    $('r-improve').textContent = (improve >= 0 ? '-' : '+') + Math.abs(improve).toFixed(1) + '%';
    $('r-cities').textContent = String(params.n);
    $('r-stagnant').textContent = fmtInt(ga.stagnant);

    const pinRow = $('pin-row');
    if (state.pinned) {
      pinRow.hidden = false;
      $('r-pinned').textContent = fmtKm(state.pinned.length);
      const delta = ga.best.length - state.pinned.length;
      const dl = $('r-delta');
      const pct = (delta / state.pinned.length) * 100;
      dl.textContent = (delta <= 0 ? '-' : '+') + fmtKm(Math.abs(delta)).replace(' km', '') +
        '  (' + (delta <= 0 ? '' : '+') + pct.toFixed(1) + '%)';
      dl.dataset.sign = delta <= 0 ? 'good' : 'warn';
    } else {
      pinRow.hidden = true;
    }
  }

  // ---- Main loop ----
  let last = performance.now();
  let acc = 0;
  function frame(now) {
    const dt = Math.min((now - last) / 1000, 0.1);
    last = now;

    if (state.running) {
      acc += dt * state.gensPerSec;
      let steps = Math.floor(acc);
      if (steps > 0) {
        acc -= steps;
        steps = Math.min(steps, 500);
        let improved = false;
        for (let s = 0; s < steps; s++) { state.ga.step(); improved = improved || state.ga.improvedThisStep; }
        if (improved && !reduceMotion) state.flash = 1;
      }
    }
    if (state.flash > 0) state.flash = Math.max(0, state.flash - dt * 2.6);

    renderMap();
    renderChart();
    renderReadouts();
    requestAnimationFrame(frame);
  }

  /* ----------------------------------------------------------------------------
     Controls
  ---------------------------------------------------------------------------- */
  function setRunning(on) {
    state.running = on;
    const btn = $('btn-play');
    btn.dataset.on = on ? '1' : '0';
    btn.querySelector('.label').textContent = on ? 'Pause' : 'Play';
    $('run-state').textContent = on ? 'RUNNING' : 'PAUSED';
    $('run-state').dataset.on = on ? '1' : '0';
  }

  function bindSlider(id, valId, fmt, onInput, structural) {
    const el = $(id);
    const out = $(valId);
    const render = () => { out.textContent = fmt(parseFloat(el.value)); };
    el.addEventListener('input', () => {
      render();
      if (!structural) onInput(parseFloat(el.value));
    });
    if (structural) {
      el.addEventListener('change', () => onInput(parseFloat(el.value)));
    }
    render();
  }

  function bindSegment(groupId, key, apply) {
    const group = $(groupId);
    group.addEventListener('click', (e) => {
      const btn = e.target.closest('[data-val]');
      if (!btn) return;
      for (const b of group.querySelectorAll('[data-val]')) b.dataset.on = '0';
      btn.dataset.on = '1';
      params[key] = btn.dataset.val;
      apply();
    });
  }

  function syncGaParams() {
    if (!state.ga) return;
    state.ga.params.mut = params.mut;
    state.ga.params.tourn = params.tourn;
    state.ga.params.elite = params.elite;
    state.ga.params.crossover = params.crossover;
    state.ga.params.mutation = params.mutation;
  }

  function init() {
    rebuild();

    bindSlider('c-cities', 'v-cities', (v) => `${v} cities`, (v) => { params.n = v | 0; rebuild(); }, true);
    bindSlider('c-pop', 'v-pop', (v) => `${v}`, (v) => { params.pop = v | 0; rebuild(); }, true);
    bindSlider('c-mut', 'v-mut', (v) => `${Math.round(v * 100)}%`, (v) => { params.mut = v; syncGaParams(); });
    bindSlider('c-tourn', 'v-tourn', (v) => `k = ${v}`, (v) => { params.tourn = v | 0; syncGaParams(); });
    bindSlider('c-elite', 'v-elite', (v) => `${v}`, (v) => { params.elite = v | 0; syncGaParams(); });
    bindSlider('c-speed', 'v-speed', (v) => `${v}/s`, (v) => { state.gensPerSec = v; });

    bindSegment('seg-cx', 'crossover', syncGaParams);
    bindSegment('seg-mut', 'mutation', syncGaParams);

    $('btn-play').addEventListener('click', () => setRunning(!state.running));
    $('btn-step').addEventListener('click', () => { state.ga.step(); if (state.ga.improvedThisStep && !reduceMotion) state.flash = 1; });
    $('btn-reset').addEventListener('click', () => { state.ga.reset(); state.flash = 0; });
    $('btn-regen').addEventListener('click', () => { rebuild((Math.random() * 1e9) | 0); });

    $('btn-pin').addEventListener('click', () => {
      state.pinned = {
        best: state.ga.history.map((h) => h.best),
        length: state.ga.best.length,
        label: `pop ${params.pop} · mut ${Math.round(params.mut * 100)}% · k${params.tourn}`,
      };
      $('pin-label').textContent = state.pinned.label;
      $('btn-clearpin').hidden = false;
    });
    $('btn-clearpin').addEventListener('click', () => {
      state.pinned = null;
      $('pin-label').textContent = '';
      $('btn-clearpin').hidden = true;
    });

    // keyboard: space toggles run, s steps, r resets
    window.addEventListener('keydown', (e) => {
      if (e.target.tagName === 'INPUT') return;
      if (e.code === 'Space') { e.preventDefault(); setRunning(!state.running); }
      else if (e.key === 's') { state.ga.step(); }
      else if (e.key === 'r') { state.ga.reset(); }
    });

    setRunning(true);
    requestAnimationFrame((t) => { last = t; frame(t); });
  }

  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init);
  else init();
}
