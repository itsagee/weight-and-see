"""
build_gallery.py
────────────────
This script scans the 'output/' folder and writes a single self-contained gallery.html.
There are no servers involved, no dependencies. The user can just open the HTML file in any browser! (yay)

Run this from the project root:
  python build_gallery.py                        # uses ./output, writes ./gallery.html
  python build_gallery.py --output ./my_output   # custom output dir
  python build_gallery.py --dest ./report.html   # custom destination
"""

import argparse
import base64
import json
import re
import sys
from pathlib import Path

# Scanner  ()
# ────────────
# pretty self-explanatory 

ALGO_PREFIXES = {"fs": "Floyd-Steinberg",
                 "jjn": "Jarvis-Judice-Ninke",
                 "st": "Stucki"}

CS_LABELS = {"rgb": "RGB", "cielab": "CIELAB", "ciexyy": "CIExyY"}

STRATEGY_ORDER = [
    "nearest", "weights", "combined", "softmax",
    "error_scaled_scale", "error_scaled_weighted_target", "error_scaled_confidence",
]
STRATEGY_LABELS = {
    "nearest":                      "Nearest-colour",
    "weights":                      "Weight-driven",
    "combined":                     "Weighted-nearest",
    "softmax":                      "Softmax",
    "error_scaled_scale":           "Error-scaled (scale)",
    "error_scaled_weighted_target": "Error-scaled (weighted_target)",
    "error_scaled_confidence":      "Error-scaled (confidence)",
}

# this function parses the filename stem and returns a dict with the parsed info, or None if it doesn't match
def parse_filename(stem: str, image_name: str, prefix: str):
    
    tag = stem.removeprefix(f"{image_name}_{prefix}_")
    
    if tag == stem:
        return None
    
    tag = re.sub(r"_α=[0-9.]+", "", tag)
    tag = re.sub(r"_p=[0-9.]+",  "", tag)
    
    parts = tag.rsplit("_", 1)
    
    if len(parts) != 2:
        return None
    body, cs = parts
    if cs not in CS_LABELS:
        return None
    sk = body.replace("-", "_")
    if sk not in STRATEGY_LABELS:
        return None
    
    return {"strategy_key": sk, "cs": cs}

# this function scans the output directory and returns a flat list of records, where each record is a dict with the following keys:
def scan(output_dir: str) -> list[dict]:
    
    records = []
    base = Path(output_dir)
    if not base.exists():
        return records

    for image_dir in sorted(base.iterdir()):
        if not image_dir.is_dir():
            continue
        image_name = image_dir.name
        
        for algo_dir in sorted(image_dir.iterdir()):
            if not algo_dir.is_dir():
                continue
            prefix = algo_dir.name
            if prefix not in ALGO_PREFIXES:
                continue
            
            for png in sorted(algo_dir.glob("*.png")):
                parsed = parse_filename(png.stem, image_name, prefix)
                if parsed is None:
                    continue
                  
                # some black magic to embed the PNG as a base64 string in the HTML, so we don't have to worry about relative paths or copying files around
                b64 = base64.b64encode(png.read_bytes()).decode()
                records.append({
                    "image":    image_name,
                    "algo":     ALGO_PREFIXES[prefix],
                    "prefix":   prefix,
                    "strategy": STRATEGY_LABELS[parsed["strategy_key"]],
                    "sk":       parsed["strategy_key"],
                    "cs":       CS_LABELS[parsed["cs"]],
                    "cs_key":   parsed["cs"],
                    "src":      f"data:image/png;base64,{b64}",
                    "order":    STRATEGY_ORDER.index(parsed["strategy_key"]),
                })
                
    # sort as: image → algo → strategy order → cs
    records.sort(key=lambda r: (r["image"], r["prefix"], r["order"], r["cs_key"]))
    
    return records

# HTML template
# ───────────────

HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Dithering Results Gallery</title>
<style>
  /* ── tokens ── */
  :root {
    --bg:       #0f1117;
    --surface:  #181c27;
    --panel:    #1e2333;
    --border:   #2a3045;
    --accent:   #5b8dee;
    --accent2:  #a78bfa;
    --text:     #e2e6f0;
    --muted:    #7b85a0;
    --mono:     'JetBrains Mono', 'Fira Mono', 'Cascadia Code', monospace;
    --sans:     'Inter', system-ui, sans-serif;
    --radius:   6px;
    --sidebar:  260px;
  }

  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

  body {
    font-family: var(--sans);
    background: var(--bg);
    color: var(--text);
    display: flex;
    height: 100vh;
    overflow: hidden;
  }

  /* ── sidebar ── */
  #sidebar {
    width: var(--sidebar);
    min-width: var(--sidebar);
    background: var(--surface);
    border-right: 1px solid var(--border);
    display: flex;
    flex-direction: column;
    overflow: hidden;
  }

  #sidebar-header {
    padding: 18px 16px 12px;
    border-bottom: 1px solid var(--border);
  }

  #sidebar-header h1 {
    font-size: 13px;
    font-weight: 600;
    letter-spacing: .08em;
    text-transform: uppercase;
    color: var(--accent);
  }

  #sidebar-header p {
    font-size: 11px;
    color: var(--muted);
    margin-top: 3px;
    font-family: var(--mono);
  }

  #filters {
    flex: 1;
    overflow-y: auto;
    padding: 12px 0;
  }

  .filter-group { padding: 0 16px 14px; }

  .filter-group label.group-label {
    display: block;
    font-size: 10px;
    font-weight: 700;
    letter-spacing: .1em;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 7px;
  }

  .chip-row { display: flex; flex-wrap: wrap; gap: 5px; }

  .chip {
    font-family: var(--mono);
    font-size: 10px;
    padding: 3px 8px;
    border-radius: 99px;
    border: 1px solid var(--border);
    background: var(--panel);
    color: var(--muted);
    cursor: pointer;
    user-select: none;
    transition: background .12s, color .12s, border-color .12s;
    white-space: nowrap;
  }
  .chip.on {
    background: var(--accent);
    border-color: var(--accent);
    color: #fff;
  }
  .chip:hover:not(.on) { border-color: var(--accent); color: var(--text); }

  #sidebar-footer {
    padding: 10px 16px;
    border-top: 1px solid var(--border);
    display: flex;
    gap: 8px;
  }

  .btn {
    flex: 1;
    font-size: 11px;
    font-family: var(--mono);
    padding: 6px 0;
    border-radius: var(--radius);
    border: 1px solid var(--border);
    background: var(--panel);
    color: var(--text);
    cursor: pointer;
    transition: background .12s;
  }
  .btn:hover { background: var(--border); }
  .btn.primary { background: var(--accent); border-color: var(--accent); color: #fff; }
  .btn.primary:hover { background: #4a7cdc; }

  /* ── main area ── */
  #main {
    flex: 1;
    display: flex;
    flex-direction: column;
    overflow: hidden;
  }

  /* ── toolbar ── */
  #toolbar {
    background: var(--surface);
    border-bottom: 1px solid var(--border);
    padding: 9px 16px;
    display: flex;
    align-items: center;
    gap: 14px;
    flex-shrink: 0;
  }

  #count {
    font-family: var(--mono);
    font-size: 11px;
    color: var(--muted);
    min-width: 120px;
  }

  .toolbar-sep { width: 1px; height: 18px; background: var(--border); }

  .toolbar-label {
    font-size: 11px;
    color: var(--muted);
    white-space: nowrap;
  }

  #cols-range {
    width: 80px;
    accent-color: var(--accent);
  }

  #view-tabs { display: flex; gap: 4px; margin-left: auto; }

  .tab-btn {
    font-size: 11px;
    font-family: var(--mono);
    padding: 4px 11px;
    border-radius: var(--radius);
    border: 1px solid var(--border);
    background: transparent;
    color: var(--muted);
    cursor: pointer;
    transition: all .12s;
  }
  .tab-btn.active {
    background: var(--accent2);
    border-color: var(--accent2);
    color: #fff;
  }

  /* ── gallery ── */
  #gallery-wrap {
    flex: 1;
    overflow-y: auto;
    padding: 16px;
  }

  #gallery {
    display: grid;
    gap: 12px;
    grid-template-columns: repeat(var(--cols, 3), 1fr);
  }

  .card {
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    overflow: hidden;
    cursor: zoom-in;
    transition: border-color .15s, transform .15s;
    position: relative;
  }
  .card:hover { border-color: var(--accent); transform: translateY(-2px); }

  .card img {
    width: 100%;
    display: block;
    image-rendering: pixelated;
  }

  .card-meta {
    padding: 6px 8px;
    border-top: 1px solid var(--border);
  }

  .card-meta .line1 {
    font-family: var(--mono);
    font-size: 10px;
    color: var(--text);
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }

  .card-meta .line2 {
    font-family: var(--mono);
    font-size: 9px;
    color: var(--muted);
    margin-top: 2px;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }

  /* cs badge */
  .cs-badge {
    position: absolute;
    top: 6px; right: 6px;
    font-family: var(--mono);
    font-size: 9px;
    font-weight: 700;
    padding: 2px 6px;
    border-radius: 99px;
    letter-spacing: .06em;
  }
  .cs-rgb    { background: #1a3a5c; color: #7ec8ff; }
  .cs-cielab { background: #1a3a28; color: #7effc0; }
  .cs-ciexyy { background: #3a2a1a; color: #ffcc7e; }

  /* ── lightbox ── */
  #lightbox {
    display: none;
    position: fixed;
    inset: 0;
    background: rgba(0,0,0,.88);
    z-index: 999;
    flex-direction: column;
    align-items: center;
    justify-content: center;
  }
  #lightbox.open { display: flex; }

  #lb-img {
    max-width: 90vw;
    max-height: 80vh;
    image-rendering: pixelated;
    border-radius: var(--radius);
    border: 1px solid var(--border);
  }

  #lb-meta {
    margin-top: 14px;
    font-family: var(--mono);
    font-size: 12px;
    color: var(--text);
    text-align: center;
    line-height: 1.7;
  }

  #lb-nav {
    display: flex;
    gap: 12px;
    margin-top: 14px;
  }

  #lb-close {
    position: absolute;
    top: 16px; right: 20px;
    font-size: 24px;
    background: none;
    border: none;
    color: var(--muted);
    cursor: pointer;
  }
  #lb-close:hover { color: var(--text); }

  /* ── compare view ── */
  #compare-bar {
    display: none;
    background: var(--surface);
    border-bottom: 1px solid var(--border);
    padding: 8px 16px;
    gap: 10px;
    align-items: center;
    flex-shrink: 0;
    flex-wrap: wrap;
  }
  #compare-bar.visible { display: flex; }

  .compare-select {
    font-family: var(--mono);
    font-size: 11px;
    background: var(--panel);
    border: 1px solid var(--border);
    color: var(--text);
    border-radius: var(--radius);
    padding: 4px 8px;
  }

  /* ── empty state ── */
  #empty {
    display: none;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    height: 100%;
    color: var(--muted);
    font-family: var(--mono);
    font-size: 13px;
    gap: 8px;
  }
  #empty .emoji { font-size: 36px; }

  /* scrollbar */
  ::-webkit-scrollbar { width: 6px; }
  ::-webkit-scrollbar-track { background: transparent; }
  ::-webkit-scrollbar-thumb { background: var(--border); border-radius: 99px; }

  /* ── pixel inspector tooltip ── */
  #px-tooltip {
    display: none;
    position: fixed;
    z-index: 9000;
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 8px;
    pointer-events: none;
    box-shadow: 0 4px 20px rgba(0,0,0,.5);
    width: max-content;
    max-width: calc(100vw - 16px);
    max-height: calc(100vh - 16px);
    overflow-x: auto;
    overflow-y: auto;
  }
  #px-tooltip.visible { display: block; }
  #px-panels { gap: 6px; }
  .px-group {
    display: inline-block;
    vertical-align: top;
    margin-right: 8px;
  }
  .px-group:last-child { margin-right: 0; }
  .px-label {
    font-family: var(--mono);
    font-size: 9px;
    color: var(--muted);
    text-align: center;
    margin-bottom: 3px;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    max-width: 80px;
  }
  .px-canvas {
    display: block;
    border: 1px solid var(--border);
    border-radius: 3px;
    image-rendering: pixelated;
  }
  #px-coords {
    font-family: var(--mono);
    font-size: 9px;
    color: var(--muted);
    text-align: center;
    margin-top: 5px;
  }
</style>
</head>
<body>

<!-- ── sidebar ── -->
<nav id="sidebar">
  <div id="sidebar-header">
    <h1>Dithering Results</h1>
    <p id="dir-label">__DIR__</p>
  </div>
  <div id="filters">
    <div class="filter-group">
      <label class="group-label">Image</label>
      <div class="chip-row" id="f-image"></div>
    </div>
    <div class="filter-group">
      <label class="group-label">Algorithm</label>
      <div class="chip-row" id="f-algo"></div>
    </div>
    <div class="filter-group">
      <label class="group-label">Strategy</label>
      <div class="chip-row" id="f-strategy"></div>
    </div>
    <div class="filter-group">
      <label class="group-label">Colour space</label>
      <div class="chip-row" id="f-cs"></div>
    </div>
  </div>
  <!-- inspector controls -->
  <div id="inspector-section" style="border-top:1px solid var(--border); padding:12px 16px;">
    <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:10px;">
      <label class="group-label" style="margin-bottom:0;">Pixel Inspector</label>
      <button id="inspector-toggle" onclick="toggleInspector()" style="font-size:10px;font-family:var(--mono);padding:3px 10px;border-radius:99px;border:1px solid var(--border);background:var(--panel);color:var(--muted);cursor:pointer;transition:all .12s;">OFF</button>
    </div>
    <div id="inspector-options" style="display:none;">
      <label class="group-label" style="margin-bottom:6px;">Grid size</label>
      <div class="chip-row" style="margin-bottom:10px;">
        <span class="chip px-size-chip" data-size="3" onclick="setPxGrid(3,this)">3x3</span>
        <span class="chip px-size-chip on" data-size="5" onclick="setPxGrid(5,this)">5x5</span>
        <span class="chip px-size-chip" data-size="7" onclick="setPxGrid(7,this)">7x7</span>
        <span class="chip px-size-chip" data-size="9" onclick="setPxGrid(9,this)">9x9</span>
      </div>
      <label class="group-label" style="margin-bottom:6px;">Algorithms to compare</label>
      <div class="chip-row" style="margin-bottom:10px;" id="px-algo-chips"></div>
      <label class="group-label" style="margin-bottom:6px;">Strategies to compare</label>
      <div class="chip-row" id="px-strategy-chips"></div>
    </div>
  </div>
  <div id="sidebar-footer">
    <button class="btn" onclick="selectAll()">All</button>
    <button class="btn" onclick="clearAll()">None</button>
  </div>
</nav>

<!-- ── main ── -->
<div id="main">
  <div id="toolbar">
    <span id="count">— images</span>
    <div class="toolbar-sep"></div>
    <span class="toolbar-label">Columns</span>
    <input type="range" id="cols-range" min="1" max="8" value="3"
           oninput="setCols(this.value)">
    <div class="toolbar-sep"></div>
    <div id="view-tabs">
      <button class="tab-btn active" onclick="setView('grid', this)">Grid</button>
      <button class="tab-btn" onclick="setView('compare', this)">Compare</button>
    </div>
  </div>

  <!-- compare controls (hidden in grid mode) -->
  <div id="compare-bar">
    <span class="toolbar-label">Fix:</span>
    <select class="compare-select" id="cmp-image" onchange="renderCompare()">
      <option value="">— image —</option>
    </select>
    <select class="compare-select" id="cmp-cs" onchange="renderCompare()">
      <option value="">— colour space —</option>
    </select>
    <select class="compare-select" id="cmp-axis" onchange="renderCompare()">
      <option value="strategy">Rows = strategies, cols = algorithms</option>
      <option value="algo">Rows = algorithms, cols = strategies</option>
    </select>
  </div>

  <div id="gallery-wrap">
    <div id="gallery"></div>
    <div id="empty"><span class="emoji">◻</span><span>No images match the current filters.</span></div>
  </div>
</div>

<!-- ── pixel inspector tooltip ── -->
<div id="px-tooltip">
  <div id="px-panels"></div>
  <div id="px-coords"></div>
</div>

<!-- ── lightbox ── -->
<div id="lightbox" onclick="closeLb(event)">
  <button id="lb-close" onclick="closeLightbox()">✕</button>
  <img id="lb-img" src="" alt="">
  <div id="lb-meta"></div>
  <div id="lb-nav">
    <button class="btn" onclick="lbStep(-1)">← Prev</button>
    <button class="btn" onclick="lbStep(1)">Next →</button>
  </div>
</div>

<script>
// ── data ──────────────────────────────────────────────────────────────────────
const RECORDS = __RECORDS_JSON__;

// derive unique values in canonical order
function unique(key, order) {
  const vals = [...new Set(RECORDS.map(r => r[key]))];
  if (order) vals.sort((a,b) => order.indexOf(a) - order.indexOf(b));
  else vals.sort();
  return vals;
}

const IMAGES     = unique('image');
const ALGOS      = unique('algo',     ['Floyd-Steinberg','Jarvis-Judice-Ninke','Stucki']);
const STRATEGIES = unique('strategy', __STRATEGY_ORDER__);
const CSS_LIST   = unique('cs',       ['RGB','CIELAB','CIExyY']);

// ── filter state ─────────────────────────────────────────────────────────────
const active = { image: new Set(IMAGES), algo: new Set(ALGOS),
                 strategy: new Set(STRATEGIES), cs: new Set(CSS_LIST) };

let currentView   = 'grid';
let lbIndex       = 0;
let visibleRecords = [];

// ── chip builder ─────────────────────────────────────────────────────────────
function buildChips(containerId, values, filterKey) {
  const el = document.getElementById(containerId);
  el.innerHTML = '';
  values.forEach(v => {
    const c = document.createElement('span');
    c.className = 'chip on';
    c.textContent = v;
    c.onclick = () => {
      const on = active[filterKey].has(v);
      on ? active[filterKey].delete(v) : active[filterKey].add(v);
      c.classList.toggle('on', !on);
      applyFilters();
    };
    el.appendChild(c);
  });
}

function selectAll() {
  IMAGES.forEach(v => active.image.add(v));
  ALGOS.forEach(v => active.algo.add(v));
  STRATEGIES.forEach(v => active.strategy.add(v));
  CSS_LIST.forEach(v => active.cs.add(v));
  document.querySelectorAll('.chip').forEach(c => c.classList.add('on'));
  applyFilters();
}

function clearAll() {
  ['image','algo','strategy','cs'].forEach(k => active[k].clear());
  document.querySelectorAll('.chip').forEach(c => c.classList.remove('on'));
  applyFilters();
}

function applyFilters() {
  currentView === 'grid' ? renderGrid() : renderCompare();
}

// ── grid view ────────────────────────────────────────────────────────────────
function renderGrid() {
  visibleRecords = RECORDS.filter(r =>
    active.image.has(r.image) && active.algo.has(r.algo) &&
    active.strategy.has(r.strategy) && active.cs.has(r.cs)
  );

  const gallery = document.getElementById('gallery');
  const empty   = document.getElementById('empty');
  gallery.innerHTML = '';

  document.getElementById('count').textContent =
    `${visibleRecords.length} image${visibleRecords.length !== 1 ? 's' : ''}`;

  if (visibleRecords.length === 0) {
    gallery.style.display = 'none';
    empty.style.display = 'flex';
    return;
  }
  gallery.style.display = 'grid';
  empty.style.display = 'none';

  visibleRecords.forEach((r, i) => {
    const card = document.createElement('div');
    card.className = 'card';
    card.onclick = () => openLightbox(i);

    const csClass = { RGB:'cs-rgb', CIELAB:'cs-cielab', CIExyY:'cs-ciexyy' }[r.cs] || '';
    card.innerHTML = `
      <img src="${r.src}" alt="${r.strategy} ${r.cs}" loading="lazy">
      <span class="cs-badge ${csClass}">${r.cs}</span>
      <div class="card-meta">
        <div class="line1">${r.strategy}</div>
        <div class="line2">${r.image} · ${r.algo}</div>
      </div>`;
    const img = card.querySelector('img');
    attachInspector(img, `${r.strategy} · ${r.cs}`, r.src);
    gallery.appendChild(card);
  });
}

function setCols(n) {
  document.getElementById('gallery').style.setProperty('--cols', n);
  document.getElementById('gallery-wrap').querySelector('#gallery').style.gridTemplateColumns =
    `repeat(${n}, 1fr)`;
}

// ── compare view ─────────────────────────────────────────────────────────────
function renderCompare() {
  const imgSel = document.getElementById('cmp-image');
  const csSel  = document.getElementById('cmp-cs');
  const axis   = document.getElementById('cmp-axis').value;
  const img    = imgSel.value;
  const cs     = csSel.value;

  const gallery = document.getElementById('gallery');
  const empty   = document.getElementById('empty');
  gallery.innerHTML = '';

  if (!img || !cs) {
    gallery.style.display = 'none';
    empty.style.display   = 'flex';
    document.getElementById('count').textContent = '—';
    return;
  }

  // build a lookup: (row_key, col_key) → record
  let rowVals, colVals;
  if (axis === 'strategy') {
    rowVals = STRATEGIES.filter(s => active.strategy.has(s));
    colVals = ALGOS.filter(a => active.algo.has(a));
  } else {
    rowVals = ALGOS.filter(a => active.algo.has(a));
    colVals = STRATEGIES.filter(s => active.strategy.has(s));
  }

  const lookup = {};
  RECORDS.forEach(r => {
    if (r.image !== img || r.cs !== cs) return;
    const rk = axis === 'strategy' ? r.strategy : r.algo;
    const ck = axis === 'strategy' ? r.algo : r.strategy;
    lookup[`${rk}||${ck}`] = r;
  });

  // header row
  const nCols = colVals.length + 1;
  gallery.style.gridTemplateColumns = `160px repeat(${colVals.length}, 1fr)`;
  gallery.style.display = 'grid';
  empty.style.display   = 'none';

  // corner
  const corner = document.createElement('div');
  corner.style.cssText = 'padding:8px;font-size:10px;color:var(--muted);font-family:var(--mono);display:flex;align-items:flex-end;';
  corner.textContent = `${img} · ${cs}`;
  gallery.appendChild(corner);

  colVals.forEach(cv => {
    const h = document.createElement('div');
    h.style.cssText = 'padding:6px 8px;font-family:var(--mono);font-size:10px;color:var(--accent2);text-align:center;border-bottom:1px solid var(--border);';
    h.textContent = cv;
    gallery.appendChild(h);
  });

  visibleRecords = [];
  rowVals.forEach(rv => {
    const rowLabel = document.createElement('div');
    rowLabel.style.cssText = 'padding:6px 8px;font-family:var(--mono);font-size:10px;color:var(--accent);display:flex;align-items:center;border-right:1px solid var(--border);word-break:break-word;';
    rowLabel.textContent = rv;
    gallery.appendChild(rowLabel);

    colVals.forEach(cv => {
      const r = lookup[`${rv}||${cv}`];
      const cell = document.createElement('div');
      cell.style.cssText = 'background:var(--panel);border:1px solid var(--border);border-radius:var(--radius);overflow:hidden;';
      if (r) {
        const idx = visibleRecords.length;
        visibleRecords.push(r);
        cell.innerHTML = `<img src="${r.src}" style="width:100%;display:block;image-rendering:pixelated;cursor:zoom-in;" loading="lazy">`;
        const img = cell.querySelector('img');
        img.onclick = () => openLightbox(idx);
        attachInspector(img, `${r.strategy} · ${r.algo} · ${r.cs}`, r.src);
      } else {
        cell.innerHTML = `<div style="height:80px;display:flex;align-items:center;justify-content:center;color:var(--muted);font-size:11px;font-family:var(--mono);">—</div>`;
      }
      gallery.appendChild(cell);
    });
  });

  document.getElementById('count').textContent =
    `${visibleRecords.length} image${visibleRecords.length !== 1 ? 's' : ''}`;
}

// ── view toggle ───────────────────────────────────────────────────────────────
function setView(view, btn) {
  currentView = view;
  document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
  btn.classList.add('active');

  const bar = document.getElementById('compare-bar');
  if (view === 'compare') {
    bar.classList.add('visible');
    renderCompare();
  } else {
    bar.classList.remove('visible');
    // reset grid cols
    document.getElementById('gallery').style.gridTemplateColumns = '';
    renderGrid();
  }
}

// ── lightbox ──────────────────────────────────────────────────────────────────
function openLightbox(i) {
  lbIndex = i;
  showLb();
  document.getElementById('lightbox').classList.add('open');
  document.addEventListener('keydown', lbKey);
}

function showLb() {
  const r = visibleRecords[lbIndex];
  if (!r) return;
  document.getElementById('lb-img').src = r.src;
  document.getElementById('lb-meta').innerHTML =
    `<strong>${r.strategy}</strong><br>${r.image} · ${r.algo} · ${r.cs}<br>` +
    `<span style="color:var(--muted)">${lbIndex + 1} / ${visibleRecords.length}</span>`;
}

function lbStep(d) {
  lbIndex = (lbIndex + d + visibleRecords.length) % visibleRecords.length;
  showLb();
}

function lbKey(e) {
  if (e.key === 'ArrowRight') lbStep(1);
  if (e.key === 'ArrowLeft')  lbStep(-1);
  if (e.key === 'Escape')     closeLightbox();
}

function closeLightbox() {
  document.getElementById('lightbox').classList.remove('open');
  document.removeEventListener('keydown', lbKey);
}

function closeLb(e) {
  if (e.target === document.getElementById('lightbox')) closeLightbox();
}

// ── compare bar population ────────────────────────────────────────────────────
function populateCompareBar() {
  const imgSel = document.getElementById('cmp-image');
  const csSel  = document.getElementById('cmp-cs');
  IMAGES.forEach(v => {
    const o = document.createElement('option');
    o.value = o.textContent = v;
    imgSel.appendChild(o);
  });
  CSS_LIST.forEach(v => {
    const o = document.createElement('option');
    o.value = o.textContent = v;
    csSel.appendChild(o);
  });
  if (IMAGES.length)   imgSel.value = IMAGES[0];
  if (CSS_LIST.length) csSel.value  = CSS_LIST[0];
}

// ── pixel inspector ──────────────────────────────────────────────────────────
let pxEnabled  = false;
let pxGrid     = 5;

// pixel display size and max panels per row — tuned so tooltip stays compact
function getPxSize(grid) {
  if (grid <= 3) return 25;
  if (grid <= 5) return 20;
  if (grid <= 7) return 17;
  return 12;  // 9x9
}

// max number of panels per row — keeps tooltip roughly square
function getMaxCols(grid) {
  if (grid <= 3) return 8;
  if (grid <= 5) return 7;
  if (grid <= 7) return 6;
  return 6;  // 9x9 — two per row keeps it compact
}

// filter state for inspector — which algos/strategies to show in tooltip
const pxActive = {
  algo:     new Set(),
  strategy: new Set(),
};

function toggleInspector() {
  pxEnabled = !pxEnabled;
  const btn = document.getElementById('inspector-toggle');
  const opts = document.getElementById('inspector-options');
  btn.textContent = pxEnabled ? 'ON' : 'OFF';
  btn.style.background    = pxEnabled ? 'var(--accent)' : 'var(--panel)';
  btn.style.borderColor   = pxEnabled ? 'var(--accent)' : 'var(--border)';
  btn.style.color         = pxEnabled ? '#fff' : 'var(--muted)';
  opts.style.display      = pxEnabled ? 'block' : 'none';
  if (!pxEnabled) hideInspector();
}

function setPxGrid(n, chip) {
  pxGrid = n;
  document.querySelectorAll('.px-size-chip').forEach(c => c.classList.remove('on'));
  chip.classList.add('on');
}

function buildPxChips() {
  // algos
  const ac = document.getElementById('px-algo-chips');
  ac.innerHTML = '';
  ALGOS.forEach(v => {
    pxActive.algo.add(v);
    const c = document.createElement('span');
    c.className = 'chip on';
    c.textContent = v;
    c.onclick = () => {
      const on = pxActive.algo.has(v);
      on ? pxActive.algo.delete(v) : pxActive.algo.add(v);
      c.classList.toggle('on', !on);
    };
    ac.appendChild(c);
  });
  // strategies
  const sc = document.getElementById('px-strategy-chips');
  sc.innerHTML = '';
  STRATEGIES.forEach(v => {
    pxActive.strategy.add(v);
    const c = document.createElement('span');
    c.className = 'chip on';
    c.textContent = v;
    c.onclick = () => {
      const on = pxActive.strategy.has(v);
      on ? pxActive.strategy.delete(v) : pxActive.strategy.add(v);
      c.classList.toggle('on', !on);
    };
    sc.appendChild(c);
  });
}

// cache: src string → HTMLCanvasElement
const imgCache = new Map();

function getCanvas(src) {
  if (imgCache.has(src)) return Promise.resolve(imgCache.get(src));
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => {
      const c = document.createElement('canvas');
      c.width  = img.naturalWidth;
      c.height = img.naturalHeight;
      c.getContext('2d').drawImage(img, 0, 0);
      imgCache.set(src, c);
      resolve(c);
    };
    img.onerror = reject;
    img.src = src;
  });
}

function drawZoom(destCanvas, srcCanvas, cx, cy, grid, pxSize) {
  const ctx  = destCanvas.getContext('2d');
  const half = Math.floor(grid / 2);
  for (let dy = -half; dy <= half; dy++) {
    for (let dx = -half; dx <= half; dx++) {
      const px = Math.max(0, Math.min(srcCanvas.width  - 1, cx + dx));
      const py = Math.max(0, Math.min(srcCanvas.height - 1, cy + dy));
      const [r, g, b, a] = srcCanvas.getContext('2d').getImageData(px, py, 1, 1).data;
      ctx.fillStyle = `rgba(${r},${g},${b},${a/255})`;
      const tx = (dx + half) * pxSize;
      const ty = (dy + half) * pxSize;
      ctx.fillRect(tx, ty, pxSize, pxSize);
      ctx.strokeStyle = 'rgba(0,0,0,0.2)';
      ctx.lineWidth = 0.5;
      ctx.strokeRect(tx, ty, pxSize, pxSize);
    }
  }
  // centre highlight
  const h2 = half * pxSize;
  ctx.strokeStyle = '#ffffff';
  ctx.lineWidth = 1;
  ctx.strokeRect(h2 + 0.5, h2 + 0.5, pxSize - 1, pxSize - 1);
}

function getSiblings(img) {
  if (currentView === 'compare') {
    return Array.from(document.querySelectorAll('#gallery img[data-src]'))
      .filter(el => {
        const r = RECORDS.find(rec => rec.src === el.dataset.src);
        if (!r) return true;
        return pxActive.algo.has(r.algo) && pxActive.strategy.has(r.strategy);
      });
  }
  return [img];
}

async function showInspector(triggerImg, clientX, clientY) {
  if (!pxEnabled) return;
  const siblings = getSiblings(triggerImg);
  if (!siblings.length) return;

  const rect = triggerImg.getBoundingClientRect();
  const relX = (clientX - rect.left) / rect.width;
  const relY = (clientY - rect.top)  / rect.height;

  const panels  = document.getElementById('px-panels');
  const tooltip = document.getElementById('px-tooltip');
  const coords  = document.getElementById('px-coords');
  panels.innerHTML = '';

  const pxSize     = getPxSize(pxGrid);
  const canvasSize = pxGrid * pxSize;
  const maxCols    = getMaxCols(pxGrid);

  // set grid columns on the panels container
  panels.style.display             = 'grid';
  panels.style.gridTemplateColumns = `repeat(${maxCols}, ${canvasSize + 4}px)`;
  panels.style.gap                 = '6px';
  let natW = 0, natH = 0;

  for (const img of siblings) {
    const src = img.dataset.src;
    if (!src) continue;
    let srcCanvas;
    try { srcCanvas = await getCanvas(src); }
    catch { continue; }

    natW = srcCanvas.width;
    natH = srcCanvas.height;
    const cx = Math.round(relX * (natW - 1));
    const cy = Math.round(relY * (natH - 1));

    const group = document.createElement('div');
    group.className = 'px-group';

    const label = document.createElement('div');
    label.className = 'px-label';
    label.style.maxWidth = canvasSize + 'px';
    label.style.width    = canvasSize + 'px';
    label.title = img.dataset.label || '';
    label.textContent = img.dataset.label || '';
    group.appendChild(label);

    const destCanvas = document.createElement('canvas');
    destCanvas.className = 'px-canvas';
    destCanvas.width  = canvasSize;
    destCanvas.height = canvasSize;
    destCanvas.style.width  = canvasSize + 'px';
    destCanvas.style.height = canvasSize + 'px';
    drawZoom(destCanvas, srcCanvas, cx, cy, pxGrid, pxSize);
    group.appendChild(destCanvas);
    panels.appendChild(group);
  }

  if (natW && natH) {
    const cx0 = Math.round(relX * (natW - 1));
    const cy0 = Math.round(relY * (natH - 1));
    coords.textContent = `pixel (${cx0}, ${cy0})  ·  ${natW}x${natH}  ·  grid ${pxGrid}x${pxGrid}`;
  }

  // make visible off-screen first so we can measure its true size
  tooltip.style.left       = '-9999px';
  tooltip.style.top        = '-9999px';
  tooltip.style.maxWidth   = 'none';
  tooltip.classList.add('visible');

  const TIP_OFFSET = 14;
  const tw = tooltip.offsetWidth;
  const th = tooltip.offsetHeight;
  const vw = window.innerWidth;
  const vh = window.innerHeight;

  // horizontal: prefer right of cursor, flip left if overflow
  let tx = clientX + TIP_OFFSET;
  if (tx + tw > vw - 8) tx = clientX - tw - TIP_OFFSET;
  tx = Math.max(8, tx);

  // vertical: prefer ABOVE cursor so it never covers what you're pointing at
  let ty = clientY - th - TIP_OFFSET;
  if (ty < 8) ty = clientY + TIP_OFFSET;  // fall back to below if no room above
  ty = Math.min(ty, vh - th - 8);          // clamp bottom edge

  tooltip.style.left = tx + 'px';
  tooltip.style.top  = ty + 'px';
}

function hideInspector() {
  document.getElementById('px-tooltip').classList.remove('visible');
}

// ── label abbreviation ────────────────────────────────────────────────────────
function abbreviate(label) {
  return label
    // algorithms
    .replace(/floyd[\s\-_]?steinberg/gi,      'FS')
    .replace(/jarvis[\s\-_]?judice[\s\-_]?ninke/gi, 'JJN')
    .replace(/stucki/gi,                      'ST')
    // strategies
    .replace(/nearest[\s\-_]?colour/gi,       'nearest')
    .replace(/weight[\s\-_]?driven/gi,        'weight')
    .replace(/weighted[\s\-_]?nearest/gi,     'w-nearest')
    .replace(/error[\s\-_]?scaled/gi,         'er')
    .replace(/softmax/gi,                     'softmax')
    // colour spaces
    .replace(/cielab/gi,                      'LAB')
    .replace(/ciexyy/gi,                      'xyY')
    .replace(/\brgb\b/gi,                     'RGB')
    // separators — collapse multiple dots/spaces
    .replace(/\s*·\s*/g, ' · ')
    .trim();
}

function attachInspector(img, label, src) {
  img.dataset.src   = src;
  img.dataset.label = abbreviate(label);
  img.addEventListener('mousemove', e => showInspector(img, e.clientX, e.clientY));
  img.addEventListener('mouseleave', hideInspector);
}

// ── init ──────────────────────────────────────────────────────────────────────
buildChips('f-image',    IMAGES,     'image');
buildChips('f-algo',     ALGOS,      'algo');
buildChips('f-strategy', STRATEGIES, 'strategy');
buildChips('f-cs',       CSS_LIST,   'cs');
populateCompareBar();
buildPxChips();
renderGrid();
</script>
</body>
</html>
"""


# Builder
# ─────────

# this function is called by the CLI to scan the output directory and basically builds the gallery HTML
def build(output_dir: str, dest: str) -> None:
    
    print(f"Scanning {output_dir} …")
    
    records = scan(output_dir)
    if not records:
        print("No images found. Check the output directory path.")
        return

    strategy_order_js = json.dumps(
        [STRATEGY_LABELS[sk] for sk in STRATEGY_ORDER if sk in STRATEGY_LABELS]
    )

    html = (HTML
            .replace("__RECORDS_JSON__", json.dumps(records))
            .replace("__STRATEGY_ORDER__", strategy_order_js)
            .replace("__DIR__", str(Path(output_dir).resolve())))

    Path(dest).write_text(html, encoding="utf-8")
    size_mb = Path(dest).stat().st_size / 1_048_576
    
    print(f"Written {dest}  ({size_mb:.1f} MB, {len(records)} images)")
    print("Open it in any browser — no server needed.")


# CLI
# ──────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build a self-contained gallery HTML.")
    parser.add_argument("--output", default="./output", help="Output directory to scan (default: ./output)")
    parser.add_argument("--dest",   default="./gallery.html", help="Destination HTML file (default: ./gallery.html)")
    args = parser.parse_args()
    
    build(args.output, args.dest)