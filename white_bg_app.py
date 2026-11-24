import base64
import io
import os
import zipfile
from pathlib import Path
import numpy as np
from PIL import Image, ImageFilter
import onnxruntime as ort
from flask import Flask, render_template_string, request, jsonify, send_file

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = BASE_DIR / "input_images"
DEFAULT_OUTPUT = BASE_DIR / "output_white_bg"
MODEL_PATH = BASE_DIR / "u2net.onnx"
SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".webp"}

app = Flask(__name__)
_session = None
_input_name = None
_session_threads = None


def get_session(num_threads: int | None = None):
    global _session, _input_name, _session_threads
    if not MODEL_PATH.exists() or MODEL_PATH.stat().st_size < 10_000_000:
        raise FileNotFoundError(f"u2net.onnx missing or incomplete at: {MODEL_PATH}")

    cpu_count = os.cpu_count() or 4
    threads = max(1, min(int(num_threads or cpu_count), cpu_count))

    if _session is None or _session_threads != threads:
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = threads
        opts.inter_op_num_threads = 1
        opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        _session = ort.InferenceSession(
            str(MODEL_PATH),
            sess_options=opts,
            providers=["CPUExecutionProvider"],
        )
        _input_name = _session.get_inputs()[0].name
        _session_threads = threads
    return _session, _input_name


def remove_bg_to_white(
    im: Image.Image,
    low_thresh: float = 0.45,
    high_thresh: float = 0.78,
    num_threads: int | None = None,
) -> Image.Image:
    session, input_name = get_session(num_threads=num_threads)
    orig_w, orig_h = im.size
    rgb = im.convert("RGB")
    resized = rgb.resize((320, 320), Image.Resampling.BILINEAR)

    arr = np.asarray(resized, dtype=np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    arr = (arr - mean) / std
    arr = np.transpose(arr, (2, 0, 1))[np.newaxis, ...].astype(np.float32)

    pred = session.run(None, {input_name: arr})[0][0, 0]
    p_min, p_max = float(pred.min()), float(pred.max())
    pred = (pred - p_min) / (p_max - p_min + 1e-8)

    low = min(low_thresh, high_thresh - 0.05)
    high = max(high_thresh, low + 0.05)
    alpha_small = np.clip((pred - low) / (high - low), 0.0, 1.0)

    mask = Image.fromarray((alpha_small * 255).astype(np.uint8), mode="L").resize(
        (orig_w, orig_h), Image.Resampling.BICUBIC
    )
    mask = mask.filter(ImageFilter.MinFilter(3)).filter(ImageFilter.SMOOTH)

    mask_np = np.asarray(mask, dtype=np.float32) / 255.0
    mask_clean = np.where(mask_np < 0.35, 0.0, mask_np)
    mask_clean = np.where(mask_clean > 0.78, 1.0, mask_clean)

    fg_np = np.asarray(rgb, dtype=np.float32)
    bg_np = np.full_like(fg_np, 255.0, dtype=np.float32)
    alpha_3 = mask_clean[..., np.newaxis]

    out_np = fg_np * alpha_3 + bg_np * (1.0 - alpha_3)
    light_bg = (mask_clean <= 0.05) & (
        (out_np[:, :, 0] > 165) & (out_np[:, :, 1] > 165) & (out_np[:, :, 2] > 165)
    )
    out_np[light_bg] = 255.0

    return Image.fromarray(np.clip(out_np, 0, 255).astype(np.uint8), mode="RGB")


def pil_to_base64_jpeg(im: Image.Image, max_side: int = 920) -> str:
    thumb = im.copy()
    thumb.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    thumb.convert("RGB").save(buf, format="JPEG", quality=92)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def rebuild_zip(out_root: Path, zip_filename: str = "white_bg_output_folder.zip") -> str | None:
    zip_path = out_root / zip_filename
    files = []
    for p in sorted(out_root.rglob("*.jpg")):
        if p.name == zip_filename:
            continue
        rel = p.relative_to(out_root).as_posix()
        files.append((p, rel))
    if not files:
        return None
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for src, arc in files:
            zf.write(src, arcname=arc)
    return zip_filename


HTML_TEMPLATE = r"""
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>PureWhite Studio — Catalog Image Batcher</title>
  <style>
    :root {
      --bg: #f4f6fa; --surface: #ffffff; --surface-alt: #f9fafc;
      --text: #0f172a; --muted: #64748b; --border: #e2e8f0;
      --primary: #2563eb; --primary-hover: #1d4ed8;
      --success: #059669; --success-hover: #047857;
      --danger: #dc2626; --warning: #d97706; --radius: 12px;
    }
    * { box-sizing: border-box; font-family: "Inter", "Segoe UI", Arial, sans-serif; }
    body { margin: 0; background: var(--bg); color: var(--text); }
    .topbar {
      position: sticky; top: 0; z-index: 20; background: #0f172a; color: #fff;
      padding: 14px 28px; display: flex; align-items: center; justify-content: space-between;
      gap: 12px; flex-wrap: wrap;
    }
    .brand { display: flex; align-items: center; gap: 12px; }
    .brand-badge {
      width: 34px; height: 34px; border-radius: 9px;
      background: linear-gradient(135deg, #3b82f6, #10b981);
      display: grid; place-items: center; font-weight: 700; font-size: 15px;
    }
    .brand-title { font-size: 16px; font-weight: 700; }
    .brand-sub { font-size: 12px; color: #94a3b8; }
    .top-right { display: flex; align-items: center; gap: 10px; flex-wrap: wrap; }
    .status-pill {
      display: inline-flex; align-items: center; gap: 7px;
      background: rgba(255,255,255,0.08); padding: 6px 12px; border-radius: 999px;
      font-size: 12px; font-weight: 600; color: #e2e8f0;
    }
    .dot { width: 8px; height: 8px; border-radius: 50%; background: #10b981; }
    .dot.busy { background: #f59e0b; animation: pulse 1s infinite; }
    @keyframes pulse { 0%,100% { opacity: 1; } 50% { opacity: 0.35; } }
    .wrap { max-width: 1320px; margin: 24px auto; padding: 0 20px 48px; }
    .card {
      background: var(--surface); border: 1px solid var(--border);
      border-radius: var(--radius); padding: 22px; margin-bottom: 20px;
      box-shadow: 0 1px 3px rgba(15,23,42,0.04);
    }
    .card-head { display: flex; align-items: center; justify-content: space-between; margin-bottom: 16px; gap: 12px; flex-wrap: wrap; }
    .card-title { font-size: 15px; font-weight: 700; margin: 0; }
    .card-hint { font-size: 12px; color: var(--muted); }
    .picker-box {
      border: 1.5px dashed #93c5fd; background: #f8fbff;
      border-radius: var(--radius); padding: 18px;
    }
    .picker-box label.browse-btn {
      display: inline-flex; align-items: center; gap: 8px;
      background: #2563eb; color: #fff; padding: 10px 16px;
      border-radius: 8px; font-size: 13px; font-weight: 600; cursor: pointer;
    }
    .picker-box label.browse-btn:hover { background: #1d4ed8; }
    .file-meta {
      margin-top: 12px; font-size: 13px; color: #334155;
      background: var(--surface-alt); border: 1px solid var(--border);
      padding: 10px 14px; border-radius: 8px; display: flex;
      justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 8px;
    }
    .path-box { margin-top: 14px; }
    .path-box label { display: block; font-size: 12.5px; font-weight: 700; margin-bottom: 6px; color: #1e293b; }
    .controls-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 18px; margin-top: 18px; }
    .field { background: var(--surface-alt); border: 1px solid var(--border); border-radius: 10px; padding: 14px; }
    .field-top { display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px; }
    .field label { font-size: 12.5px; font-weight: 700; color: #1e293b; }
    .val-badge { font-size: 12px; font-weight: 700; color: var(--primary); background: #dbeafe; padding: 2px 8px; border-radius: 6px; }
    input[type="range"] { width: 100%; accent-color: var(--primary); cursor: pointer; }
    select, input[type="text"] {
      width: 100%; padding: 8px 10px; border: 1px solid #cbd5e1;
      border-radius: 8px; font-size: 13px; background: #fff; color: var(--text);
    }
    .check-row { display: flex; align-items: center; gap: 10px; margin-top: 14px; font-size: 13px; color: #334155; font-weight: 500; }
    .check-row input { width: 16px; height: 16px; accent-color: var(--primary); cursor: pointer; }
    .actions {
      display: flex; align-items: center; justify-content: space-between;
      gap: 12px; margin-top: 20px; padding-top: 16px; border-top: 1px solid var(--border); flex-wrap: wrap;
    }
    .btn-group { display: flex; gap: 10px; flex-wrap: wrap; }
    .btn {
      border: none; border-radius: 9px; padding: 10px 16px;
      font-size: 13px; font-weight: 600; cursor: pointer;
      text-decoration: none; display: inline-flex; align-items: center; gap: 7px;
    }
    .btn-sm { padding: 6px 10px; font-size: 12px; border-radius: 7px; }
    .btn:disabled { background: #94a3b8; cursor: not-allowed; }
    .btn-primary { background: var(--primary); color: #fff; }
    .btn-primary:hover:not(:disabled) { background: var(--primary-hover); }
    .btn-success { background: var(--success); color: #fff; }
    .btn-success:hover { background: var(--success-hover); }
    .btn-danger { background: #fee2e2; color: #b91c1c; border: 1px solid #fca5a5; }
    .btn-danger:hover { background: #fecaca; }
    .btn-ghost { background: #f1f5f9; color: #334155; border: 1px solid var(--border); }
    .btn-ghost:hover { background: #e2e8f0; }
    .kpi-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 14px; margin-bottom: 20px; }
    .kpi { background: var(--surface); border: 1px solid var(--border); border-radius: var(--radius); padding: 16px 18px; }
    .kpi-label { font-size: 12px; font-weight: 600; color: var(--muted); text-transform: uppercase; }
    .kpi-num { font-size: 24px; font-weight: 800; margin-top: 6px; }
    .kpi-num.ok { color: var(--success); }
    .kpi-num.skip { color: var(--warning); }
    .kpi-num.err { color: var(--danger); }
    .log-console {
      background: #0b1120; color: #cbd5e1; border-radius: 10px;
      padding: 14px 16px; font-family: Consolas, monospace;
      font-size: 12px; line-height: 1.55; max-height: 170px; overflow-y: auto; white-space: pre-line;
    }
    .preview-toolbar { display: flex; align-items: center; gap: 10px; flex-wrap: wrap; }
    .color-sections { display: flex; flex-direction: column; gap: 22px; }
    .color-block {
      border: 1px solid var(--border); border-radius: 12px;
      background: var(--surface-alt); padding: 16px;
    }
    .color-block-head {
      display: flex; align-items: center; justify-content: space-between;
      gap: 10px; padding-bottom: 12px; margin-bottom: 14px;
      border-bottom: 1px solid var(--border); flex-wrap: wrap;
    }
    .color-heading {
      font-size: 15px; font-weight: 800; color: #0f172a;
      display: inline-flex; align-items: center; gap: 8px;
    }
    .color-tag {
      background: #dbeafe; color: #1d4ed8; font-size: 11.5px;
      font-weight: 700; padding: 3px 10px; border-radius: 999px;
    }
    .color-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(340px, 1fr)); gap: 18px; }
    .color-grid.large-mode { grid-template-columns: repeat(auto-fill, minmax(520px, 1fr)); }
    .thumb-card {
      border: 1px solid var(--border); border-radius: 12px; padding: 14px;
      background: #fff; box-shadow: 0 1px 4px rgba(15,23,42,0.04);
    }
    .thumb-top {
      display: flex; align-items: center; justify-content: space-between;
      gap: 8px; font-size: 11.5px; color: var(--muted); margin-bottom: 10px;
    }
    .badge-conv { color: #059669; font-weight: 700; }
    .badge-disk { color: #d97706; font-weight: 700; }
    .thumb-box {
      width: 100%; height: 320px; background: #ffffff;
      border: 1px solid #e2e8f0; border-radius: 10px;
      display: grid; place-items: center; overflow: hidden; cursor: zoom-in;
    }
    .color-grid.large-mode .thumb-box { height: 460px; }
    .thumb-box img { max-width: 100%; max-height: 100%; object-fit: contain; transition: transform .18s; }
    .thumb-box:hover img { transform: scale(1.03); }
    .thumb-name {
      margin-top: 10px; font-size: 12.5px; font-weight: 700;
      color: #1e293b; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
    }
    .empty-state { text-align: center; padding: 36px 16px; color: var(--muted); font-size: 13px; }
    .modal-backdrop {
      position: fixed; inset: 0; z-index: 50;
      background: rgba(15, 23, 42, 0.82);
      display: none; align-items: center; justify-content: center; padding: 20px;
    }
    .modal-backdrop.open { display: flex; }
    .modal-panel {
      background: #fff; border-radius: 14px; max-width: 96vw; width: 1060px;
      max-height: 92vh; display: flex; flex-direction: column; overflow: hidden;
      box-shadow: 0 20px 50px rgba(0,0,0,0.35);
    }
    .modal-head {
      padding: 12px 18px; border-bottom: 1px solid var(--border);
      display: flex; align-items: center; justify-content: space-between; gap: 10px; flex-wrap: wrap;
    }
    .modal-title { font-size: 13.5px; font-weight: 700; color: #0f172a; }
    .modal-body {
      flex: 1; overflow: auto; background: #ffffff;
      display: grid; place-items: center; padding: 20px;
    }
    .modal-body img {
      max-width: 100%; max-height: 76vh; object-fit: contain;
      transform-origin: center center; transition: transform .15s ease;
    }
    @media (max-width: 900px) {
      .controls-grid, .kpi-grid { grid-template-columns: 1fr 1fr; }
      .color-grid, .color-grid.large-mode { grid-template-columns: 1fr; }
    }
    @media (max-width: 560px) { .controls-grid, .kpi-grid { grid-template-columns: 1fr; } }
  </style>
</head>
<body>
  <header class="topbar">
    <div class="brand">
      <div class="brand-badge">W</div>
      <div>
        <div class="brand-title">PureWhite Studio</div>
        <div class="brand-sub">Marketplace #FFFFFF Batch Processor · Browser Cache Enabled</div>
      </div>
    </div>
    <div class="top-right">
      <button type="button" class="btn btn-danger btn-sm" id="clearCacheBtn">Clear Cache</button>
      <div class="status-pill">
        <span class="dot" id="statusDot"></span>
        <span id="statusText">Idle · Ready</span>
      </div>
    </div>
  </header>

  <main class="wrap">
    <section class="card">
      <div class="card-head">
        <h2 class="card-title">1. Folder &amp; Processing Settings</h2>
        <span class="card-hint" id="cacheStatusHint">Saved in browser cache (localStorage)</span>
      </div>

      <form id="uploadForm" enctype="multipart/form-data">
        <div class="picker-box">
          <div style="display:flex;align-items:center;justify-content:space-between;gap:12px;flex-wrap:wrap;">
            <div>
              <strong style="font-size:14px;">Option A: Browser Folder Picker</strong><br>
              <span style="font-size:12.5px;color:#64748b;">Click "Select Folder" to pick your main folder with color subfolders</span>
            </div>
            <label for="folderInput" class="browse-btn">Select Folder</label>
          </div>
          <input type="file" id="folderInput" name="files" webkitdirectory directory multiple accept="image/*" style="position:absolute;left:-9999px;opacity:0;">

          <div class="path-box">
            <label for="localFolderPath">Option B: Direct Local Folder Path</label>
            <input type="text" id="localFolderPath" value="{{ default_input }}" placeholder="e.g. C:\Users\ADMIN\Documents\PureWhiteStudio\input_images">
          </div>
        </div>

        <div class="file-meta">
          <span id="folderSummary">Using local path: {{ default_input }}</span>
          <span id="imageCountBadge" style="font-weight:700;color:#2563eb;">Ready</span>
        </div>

        <div class="controls-grid">
          <div class="field">
            <div class="field-top">
              <label for="lowRange">Low Threshold (BG Cut)</label>
              <span class="val-badge" id="lowVal">0.45</span>
            </div>
            <input type="range" id="lowRange" min="0.15" max="0.75" step="0.05" value="0.45">
          </div>

          <div class="field">
            <div class="field-top">
              <label for="highRange">High Threshold (FG Edge)</label>
              <span class="val-badge" id="highVal">0.78</span>
            </div>
            <input type="range" id="highRange" min="0.55" max="0.95" step="0.05" value="0.78">
          </div>

          <div class="field">
            <div class="field-top">
              <label for="threadsSelect">CPU Inference Threads</label>
              <span class="val-badge">{{ cpu_count }} Cores</span>
            </div>
            <select id="threadsSelect">
              {% for t in thread_options %}
                <option value="{{ t }}" {% if t == default_threads %}selected{% endif %}>{{ t }} Threads</option>
              {% endfor %}
            </select>
          </div>
        </div>

        <label class="check-row">
          <input type="checkbox" id="skipExisting">
          Skip already-converted images (keep unchecked to re-run fresh #FFFFFF conversion on all images)
        </label>

        <div class="actions">
          <div class="btn-group">
            <button type="submit" class="btn btn-primary" id="runBtn">Process &amp; Build Folder</button>
            <button type="button" class="btn btn-ghost" id="loadExistingBtn">Load Output Folder</button>
            <button type="button" class="btn btn-ghost" id="resetBtn">Reset</button>
          </div>
          <a class="btn btn-success" id="downloadBtn" href="/api/download/white_bg_output_folder.zip" style="{% if not has_zip %}display:none;{% endif %}">Download Output Folder (.ZIP)</a>
        </div>
      </form>
    </section>

    <section class="kpi-grid">
      <div class="kpi"><div class="kpi-label">Images Detected</div><div class="kpi-num" id="kpiTotal">0</div></div>
      <div class="kpi"><div class="kpi-label">Converted (#FFFFFF)</div><div class="kpi-num ok" id="kpiDone">0</div></div>
      <div class="kpi"><div class="kpi-label">Skipped</div><div class="kpi-num skip" id="kpiSkip">0</div></div>
      <div class="kpi"><div class="kpi-label">Errors</div><div class="kpi-num err" id="kpiErr">0</div></div>
    </section>

    <section class="card">
      <div class="card-head">
        <h2 class="card-title">2. Live Activity Log</h2>
        <button type="button" class="btn btn-ghost" id="clearLogBtn" style="padding:6px 12px;font-size:12px;">Clear Log</button>
      </div>
      <div class="log-console" id="logBox">Ready. Click "Process &amp; Build Folder" to generate fresh #FFFFFF images.</div>
    </section>

    <section class="card">
      <div class="card-head">
        <div>
          <h2 class="card-title">3. Color-wise Sample Pictures</h2>
          <span class="card-hint">Each color has its own separate section with sample pictures below it</span>
        </div>
        <div class="preview-toolbar">
          <select id="colorFilterSelect" style="width:190px;">
            <option value="ALL">All Colors</option>
          </select>
          <button type="button" class="btn btn-ghost btn-sm" id="toggleSizeBtn">Size: Large (2-Col)</button>
          <input type="text" id="filterInput" placeholder="Filter by filename..." style="max-width:210px;">
        </div>
      </div>

      <div class="color-sections" id="colorSections">
        <div class="empty-state">Color sections and sample pictures will appear here.</div>
      </div>
    </section>
  </main>

  <div class="modal-backdrop" id="detailModal">
    <div class="modal-panel">
      <div class="modal-head">
        <span class="modal-title" id="modalTitle">Detail Inspection</span>
        <div style="display:flex;gap:8px;align-items:center;">
          <button type="button" class="btn btn-ghost btn-sm" id="zoomOutBtn">-</button>
          <span id="zoomLabel" style="font-size:12px;font-weight:700;min-width:44px;text-align:center;">100%</span>
          <button type="button" class="btn btn-ghost btn-sm" id="zoomInBtn">+</button>
          <button type="button" class="btn btn-ghost btn-sm" id="zoomResetBtn">Fit</button>
          <button type="button" class="btn btn-primary btn-sm" id="closeModalBtn">Close</button>
        </div>
      </div>
      <div class="modal-body" id="modalBody">
        <img id="modalImg" src="" alt="Detail View">
      </div>
    </div>
  </div>

  <script>
    const CACHE_SETTINGS_KEY = "purewhite_studio_settings_v1";
    const CACHE_STATE_KEY = "purewhite_studio_state_v1";
    const DEFAULT_LOCAL_PATH = {{ default_input | tojson }};
    const DEFAULT_THREADS = {{ default_threads }};

    const exts = [".jpg", ".jpeg", ".png", ".webp"];
    const folderInput = document.getElementById("folderInput");
    const localFolderPath = document.getElementById("localFolderPath");
    const folderSummary = document.getElementById("folderSummary");
    const imageCountBadge = document.getElementById("imageCountBadge");
    const cacheStatusHint = document.getElementById("cacheStatusHint");
    const lowRange = document.getElementById("lowRange");
    const highRange = document.getElementById("highRange");
    const lowVal = document.getElementById("lowVal");
    const highVal = document.getElementById("highVal");
    const threadsSelect = document.getElementById("threadsSelect");
    const skipExisting = document.getElementById("skipExisting");
    const uploadForm = document.getElementById("uploadForm");
    const runBtn = document.getElementById("runBtn");
    const loadExistingBtn = document.getElementById("loadExistingBtn");
    const resetBtn = document.getElementById("resetBtn");
    const clearCacheBtn = document.getElementById("clearCacheBtn");
    const downloadBtn = document.getElementById("downloadBtn");
    const clearLogBtn = document.getElementById("clearLogBtn");
    const toggleSizeBtn = document.getElementById("toggleSizeBtn");
    const colorFilterSelect = document.getElementById("colorFilterSelect");
    const filterInput = document.getElementById("filterInput");
    const logBox = document.getElementById("logBox");
    const colorSections = document.getElementById("colorSections");
    const statusDot = document.getElementById("statusDot");
    const statusText = document.getElementById("statusText");

    const detailModal = document.getElementById("detailModal");
    const modalTitle = document.getElementById("modalTitle");
    const modalImg = document.getElementById("modalImg");
    const zoomLabel = document.getElementById("zoomLabel");
    const zoomInBtn = document.getElementById("zoomInBtn");
    const zoomOutBtn = document.getElementById("zoomOutBtn");
    const zoomResetBtn = document.getElementById("zoomResetBtn");
    const closeModalBtn = document.getElementById("closeModalBtn");

    let allPreviews = [];
    let largeMode = false;
    let currentZoom = 1.0;

    function setStatus(busy, text) {
      statusDot.classList.toggle("busy", busy);
      statusText.textContent = text;
    }

    function saveSettingsToCache() {
      try {
        const payload = {
          localFolderPath: localFolderPath.value,
          lowThresh: lowRange.value,
          highThresh: highRange.value,
          numThreads: threadsSelect.value,
          skipExisting: skipExisting.checked,
          largeMode: largeMode,
          selectedColor: colorFilterSelect.value,
          searchFilter: filterInput.value
        };
        localStorage.setItem(CACHE_SETTINGS_KEY, JSON.stringify(payload));
        cacheStatusHint.textContent = "Saved in browser cache (localStorage)";
      } catch (_) {
        cacheStatusHint.textContent = "Cache storage warning";
      }
    }

    function saveStateToCache(zipName) {
      try {
        const kpis = {
          total: document.getElementById("kpiTotal").textContent,
          done: document.getElementById("kpiDone").textContent,
          skip: document.getElementById("kpiSkip").textContent,
          err: document.getElementById("kpiErr").textContent
        };
        const trimmedLogs = logBox.textContent.slice(-25000);
        localStorage.setItem(CACHE_STATE_KEY, JSON.stringify({
          kpis,
          logs: trimmedLogs,
          previews: allPreviews,
          zipName: zipName || null
        }));
      } catch (_) {
        try {
          localStorage.setItem(CACHE_STATE_KEY, JSON.stringify({
            kpis: {
              total: document.getElementById("kpiTotal").textContent,
              done: document.getElementById("kpiDone").textContent,
              skip: document.getElementById("kpiSkip").textContent,
              err: document.getElementById("kpiErr").textContent
            },
            logs: logBox.textContent.slice(-15000),
            previews: [],
            zipName: zipName || null
          }));
        } catch (_) {}
      }
    }

    function restoreFromCache() {
      let restoredAnything = false;
      try {
        const rawSettings = localStorage.getItem(CACHE_SETTINGS_KEY);
        if (rawSettings) {
          const s = JSON.parse(rawSettings);
          if (typeof s.localFolderPath === "string") localFolderPath.value = s.localFolderPath;
          if (s.lowThresh !== undefined) {
            lowRange.value = s.lowThresh;
            lowVal.textContent = parseFloat(lowRange.value).toFixed(2);
          }
          if (s.highThresh !== undefined) {
            highRange.value = s.highThresh;
            highVal.textContent = parseFloat(highRange.value).toFixed(2);
          }
          if (s.numThreads !== undefined) threadsSelect.value = s.numThreads;
          if (typeof s.skipExisting === "boolean") skipExisting.checked = s.skipExisting;
          if (typeof s.largeMode === "boolean") {
            largeMode = s.largeMode;
            toggleSizeBtn.textContent = largeMode ? "Size: Medium (3-Col)" : "Size: Large (2-Col)";
          }
          if (typeof s.searchFilter === "string") filterInput.value = s.searchFilter;
          folderSummary.textContent = `Using local path: ${localFolderPath.value.trim() || "None"}`;
          restoredAnything = true;
        }
      } catch (_) {}

      try {
        const rawState = localStorage.getItem(CACHE_STATE_KEY);
        if (rawState) {
          const st = JSON.parse(rawState);
          if (st.kpis) {
            document.getElementById("kpiTotal").textContent = st.kpis.total || "0";
            document.getElementById("kpiDone").textContent = st.kpis.done || "0";
            document.getElementById("kpiSkip").textContent = st.kpis.skip || "0";
            document.getElementById("kpiErr").textContent = st.kpis.err || "0";
          }
          if (typeof st.logs === "string" && st.logs.length > 0) {
            logBox.textContent = st.logs;
          }
          if (Array.isArray(st.previews) && st.previews.length > 0) {
            allPreviews = st.previews;
            syncColorDropdown();
            try {
              const rawSettings = JSON.parse(localStorage.getItem(CACHE_SETTINGS_KEY) || "{}");
              if (rawSettings.selectedColor) {
                const exists = Array.from(colorFilterSelect.options).some(o => o.value === rawSettings.selectedColor);
                if (exists) colorFilterSelect.value = rawSettings.selectedColor;
              }
            } catch (_) {}
            renderColorSections();
            restoredAnything = true;
          }
          if (st.zipName) {
            downloadBtn.href = "/api/download/" + encodeURIComponent(st.zipName);
            downloadBtn.style.display = "inline-flex";
          }
        }
      } catch (_) {}

      if (restoredAnything) {
        cacheStatusHint.textContent = "Restored from browser cache";
        setStatus(false, "Restored from Cache");
      }
      return restoredAnything;
    }

    clearCacheBtn.addEventListener("click", () => {
      localStorage.removeItem(CACHE_SETTINGS_KEY);
      localStorage.removeItem(CACHE_STATE_KEY);
      folderInput.value = "";
      localFolderPath.value = DEFAULT_LOCAL_PATH;
      folderSummary.textContent = `Using local path: ${DEFAULT_LOCAL_PATH}`; imageCountBadge.textContent = "Ready"; lowRange.value = "0.45"; highRange.value = "0.78"; lowVal.textContent = "0.45"; highVal.textContent = "0.78"; threadsSelect.value = String(DEFAULT_THREADS); skipExisting.checked = false; largeMode = false; toggleSizeBtn.textContent = "Size: Large (2-Col)"; filterInput.value = ""; allPreviews = []; syncColorDropdown(); renderColorSections(); ["kpiTotal", "kpiDone", "kpiSkip", "kpiErr"].forEach(id => { document.getElementById(id).textContent = "0"; }); logBox.textContent = "Browser cache cleared. Fresh session started."; cacheStatusHint.textContent = "Browser cache cleared"; setStatus(false, "Cache Cleared"); });  function applyZoom() { zoomLabel.textContent = `${Math.round(currentZoom * 100)}%`;
      if (currentZoom <= 1.0) {
        modalImg.style.maxWidth = "100%";
        modalImg.style.maxHeight = "76vh";
        modalImg.style.transform = "none";
      } else {
        modalImg.style.maxWidth = "none";
        modalImg.style.maxHeight = "none";
        modalImg.style.transform = `scale(${currentZoom})`; } }  window.openDetailModal = function(idx) { const item = allPreviews[idx]; if (!item) return; modalTitle.textContent = `${item.folder || "Root"} — ${item.rel_path}`; modalImg.src = `data:image/jpeg;base64,${item.image}`;
      currentZoom = 1.0;
      applyZoom();
      detailModal.classList.add("open");
    };

    zoomInBtn.addEventListener("click", () => {
      currentZoom = Math.min(4.0, +(currentZoom + 0.5).toFixed(2));
      applyZoom();
    });
    zoomOutBtn.addEventListener("click", () => {
      currentZoom = Math.max(0.5, +(currentZoom - 0.5).toFixed(2));
      applyZoom();
    });
    zoomResetBtn.addEventListener("click", () => {
      currentZoom = 1.0;
      applyZoom();
    });
    closeModalBtn.addEventListener("click", () => detailModal.classList.remove("open"));
    detailModal.addEventListener("click", (e) => {
      if (e.target === detailModal) detailModal.classList.remove("open");
    });

    function syncColorDropdown() {
      const prev = colorFilterSelect.value || "ALL";
      const colors = Array.from(new Set(allPreviews.map(p => p.folder || "Root")));
      colorFilterSelect.innerHTML = `<option value="ALL">All Colors (${colors.length})</option>` +
        colors.map(c => `<option value="${c}">${c}</option>`).join("");
      if (colors.includes(prev)) colorFilterSelect.value = prev;
      else colorFilterSelect.value = "ALL";
    }

    toggleSizeBtn.addEventListener("click", () => {
      largeMode = !largeMode;
      toggleSizeBtn.textContent = largeMode ? "Size: Medium (3-Col)" : "Size: Large (2-Col)";
      renderColorSections();
      saveSettingsToCache();
    });

    lowRange.addEventListener("input", () => {
      lowVal.textContent = parseFloat(lowRange.value).toFixed(2);
      saveSettingsToCache();
    });
    highRange.addEventListener("input", () => {
      highVal.textContent = parseFloat(highRange.value).toFixed(2);
      saveSettingsToCache();
    });
    threadsSelect.addEventListener("change", saveSettingsToCache);
    skipExisting.addEventListener("change", saveSettingsToCache);
    colorFilterSelect.addEventListener("change", () => {
      renderColorSections();
      saveSettingsToCache();
    });
    filterInput.addEventListener("input", () => {
      renderColorSections();
      saveSettingsToCache();
    });

    folderInput.addEventListener("change", () => {
      const files = Array.from(folderInput.files || []);
      const imgs = files.filter(f => exts.some(ext => f.name.toLowerCase().endsWith(ext)));
      if (!files.length) return;
      const rootName = (files[0].webkitRelativePath || files[0].name).split("/")[0];
      folderSummary.textContent = `Uploaded Folder: ${rootName} (${files.length} items)`; imageCountBadge.textContent = `${imgs.length} images detected`;
      document.getElementById("kpiTotal").textContent = String(imgs.length);
      saveStateToCache(null);
    });

    localFolderPath.addEventListener("input", () => {
      if (!folderInput.files || folderInput.files.length === 0) {
        folderSummary.textContent = `Local Path: ${localFolderPath.value.trim() || "None"}`;
        imageCountBadge.textContent = "Ready";
      }
      saveSettingsToCache();
    });

    function renderColorSections() {
      const selectedColor = colorFilterSelect.value || "ALL";
      const q = filterInput.value.trim().toLowerCase();

      const grouped = new Map();
      allPreviews.forEach((p, idx) => {
        const colorName = p.folder || "Root";
        if (selectedColor !== "ALL" && colorName !== selectedColor) return;
        if (q && !p.rel_path.toLowerCase().includes(q) && !colorName.toLowerCase().includes(q)) return;
        if (!grouped.has(colorName)) grouped.set(colorName, []);
        grouped.get(colorName).push({ p, idx });
      });

      if (!grouped.size) {
        colorSections.innerHTML = `<div class="empty-state">No matching color sample pictures to display.</div>`;
        return;
      }

      const gridClass = largeMode ? "color-grid large-mode" : "color-grid";
      let html = "";
      for (const [colorName, entries] of grouped.entries()) {
        html += `
          <div class="color-block">
            <div class="color-block-head">
              <div class="color-heading">
                <span>${colorName}</span>
                <span class="color-tag">${entries.length} Sample${entries.length > 1 ? "s" : ""}</span>
              </div>
              <span class="card-hint">Sample pictures for ${colorName}</span>
            </div>
            <div class="${gridClass}">
              ${entries.map(({ p, idx }) => {
                const fileOnly = p.rel_path.split("/").pop();
                const isConv = (p.status || "").startsWith("CONVERTED");
                const badgeCls = isConv ? "badge-conv" : "badge-disk";
                return `
                  <div class="thumb-card">
                    <div class="thumb-top">
                      <span class="${badgeCls}">${p.status || "DISK FILE"} · Zoom</span>
                      <span>#FFFFFF</span>
                    </div>
                    <div class="thumb-box" onclick="openDetailModal(${idx})">
                      <img src="data:image/jpeg;base64,${p.image}" alt="${p.rel_path}">
                    </div>
                    <div class="thumb-name" title="${p.rel_path}">${fileOnly}</div>
                  </div>
                `;
              }).join("")}
            </div>
          </div>
        `;
      }
      colorSections.innerHTML = html;
    }

    async function loadExistingOutput() {
      setStatus(true, "Loading disk output samples...");
      try {
        const res = await fetch("/api/existing-output");
        const data = await res.json();
        allPreviews = data.previews || [];
        if (data.zip_name) {
          downloadBtn.href = "/api/download/" + encodeURIComponent(data.zip_name);
          downloadBtn.style.display = "inline-flex";
        }
        syncColorDropdown();
        renderColorSections();
        saveStateToCache(data.zip_name || null);
        setStatus(false, "Loaded Disk Output");
      } catch (err) {
        logBox.textContent += "\n[LOAD ERR] " + err.message;
        setStatus(false, "Error");
      }
    }

    loadExistingBtn.addEventListener("click", loadExistingOutput);
    clearLogBtn.addEventListener("click", () => {
      logBox.textContent = "";
      saveStateToCache(null);
    });

    resetBtn.addEventListener("click", () => {
      folderInput.value = "";
      folderSummary.textContent = `Local Path: ${localFolderPath.value.trim() \vert{}\vert{} "None"}`; imageCountBadge.textContent = "Ready"; allPreviews = []; syncColorDropdown(); renderColorSections(); ["kpiTotal", "kpiDone", "kpiSkip", "kpiErr"].forEach(id => { document.getElementById(id).textContent = "0"; }); logBox.textContent = "Reset complete."; saveStateToCache(null); setStatus(false, "Idle · Ready"); });  uploadForm.addEventListener("submit", async (e) => { e.preventDefault(); const hasUploadedFiles = folderInput.files && folderInput.files.length > 0; const typedPath = localFolderPath.value.trim();  if (!hasUploadedFiles && !typedPath) { alert("Please select a folder or enter a local folder path."); return; }  saveSettingsToCache(); runBtn.disabled = true; setStatus(true, "Processing Fresh Batch..."); logBox.textContent = "Starting fresh #FFFFFF background conversion...\n"; allPreviews = []; syncColorDropdown(); renderColorSections();  const formData = new FormData(); formData.append("low_thresh", lowRange.value); formData.append("high_thresh", highRange.value); formData.append("num_threads", threadsSelect.value); formData.append("skip_existing", skipExisting.checked ? "1" : "0");  let endpoint = "/api/process-local"; if (hasUploadedFiles) { endpoint = "/api/process-folder"; for (const file of folderInput.files) { const relPath = file.webkitRelativePath \vert{}\vert{} file.name; formData.append("files", file, relPath); formData.append("paths", relPath); } } else { formData.append("local_dir", typedPath); }  try { const res = await fetch(endpoint, { method: "POST", body: formData }); const data = await res.json(); if (!res.ok \vert{}\vert{} data.error) { logBox.textContent += "\n[ERROR] " + (data.error \vert{}\vert{} "Unknown error"); saveStateToCache(null); setStatus(false, "Error"); } else { document.getElementById("kpiTotal").textContent = String(data.total_images); document.getElementById("kpiDone").textContent = String(data.processed_count); document.getElementById("kpiSkip").textContent = String(data.skipped_count); document.getElementById("kpiErr").textContent = String(data.error_count); logBox.textContent = (data.logs \vert{}\vert{} []).join("\n") + `\n\nSummary -> Converted: ${data.processed_count} | Skipped: ${data.skipped_count} \vert{} Errors:${data.error_count}`;

          if (data.zip_name) {
            downloadBtn.href = "/api/download/" + encodeURIComponent(data.zip_name);
            downloadBtn.style.display = "inline-flex";
          }

          allPreviews = data.previews || [];
          syncColorDropdown();
          renderColorSections();
          saveStateToCache(data.zip_name || null);
          setStatus(false, "Fresh Conversion Done");
        }
      } catch (err) {
        logBox.textContent += "\n[REQUEST FAILED] " + err.message;
        saveStateToCache(null);
        setStatus(false, "Connection Error");
      } finally {
        runBtn.disabled = false;
      }
    });

    const hadCache = restoreFromCache();
    if (!hadCache) {
      loadExistingOutput();
    }
  </script>
</body>
</html>
"""


def build_preview_item(out_root: Path, file_path: Path, status: str = "CONVERTED") -> dict | None:
    try:
        rel = file_path.relative_to(out_root).as_posix()
        parts = Path(rel).parts
        folder_name = "/".join(parts[:-1]) if len(parts) > 1 else "Root"
        with Image.open(file_path) as im:
            b64 = pil_to_base64_jpeg(im)
        return {
            "rel_path": rel,
            "folder": folder_name,
            "status": status,
            "image": b64,
        }
    except Exception:
        return None


def collect_existing_previews(out_root: Path, per_color_limit: int = 8) -> list[dict]:
    if not out_root.exists():
        return []
    by_color = {}
    for p in sorted(out_root.rglob("*.jpg")):
        if p.name == "white_bg_output_folder.zip":
            continue
        item = build_preview_item(out_root, p, status="DISK FILE")
        if not item:
            continue
        color = item["folder"]
        by_color.setdefault(color, [])
        if len(by_color[color]) < per_color_limit:
            by_color[color].append(item)

    merged = []
    for color in sorted(by_color.keys()):
        merged.extend(by_color[color])
    return merged


def run_batch_from_items(items, out_root: Path, low_thresh: float, high_thresh: float, num_threads: int, skip_existing: bool):
    out_root.mkdir(parents=True, exist_ok=True)
    logs = []
    previews = []
    per_color_counts = {}
    total_images = processed_count = skipped_count = error_count = 0

    for arc_rel, source_obj in items:
        rel_p = Path(arc_rel.replace("\\", "/"))
        if rel_p.suffix.lower() not in SUPPORTED_EXTS:
            continue

        total_images += 1
        target_dir = out_root / rel_p.parent
        target_dir.mkdir(parents=True, exist_ok=True)

        out_name = rel_p.stem + ".jpg"
        out_file = target_dir / out_name
        arc_path = (rel_p.parent / out_name).as_posix() if str(rel_p.parent) != "." else out_name
        parts = Path(arc_path).parts
        folder_name = "/".join(parts[:-1]) if len(parts) > 1 else "Root"

        try:
            if skip_existing and out_file.exists():
                skipped_count += 1
                logs.append(f"[SKIP] {arc_path} (already exists)")
                if per_color_counts.get(folder_name, 0) < 8:
                    prev = build_preview_item(out_root, out_file, status="DISK FILE")
                    if prev:
                        previews.append(prev)
                        per_color_counts[folder_name] = per_color_counts.get(folder_name, 0) + 1
                continue

            with Image.open(source_obj) as im:
                result = remove_bg_to_white(
                    im,
                    low_thresh=low_thresh,
                    high_thresh=high_thresh,
                    num_threads=num_threads,
                )
                result.save(out_file, format="JPEG", quality=95)

                if per_color_counts.get(folder_name, 0) < 8:
                    previews.append({
                        "rel_path": arc_path,
                        "folder": folder_name,
                        "status": "CONVERTED",
                        "image": pil_to_base64_jpeg(result),
                    })
                    per_color_counts[folder_name] = per_color_counts.get(folder_name, 0) + 1

            processed_count += 1
            logs.append(f"[OK] {arc_path}")
        except Exception as e:
            error_count += 1
            logs.append(f"[ERR] {arc_path}: {e}")

    zip_name = rebuild_zip(out_root)
    return {
        "total_images": total_images,
        "processed_count": processed_count,
        "skipped_count": skipped_count,
        "error_count": error_count,
        "logs": logs,
        "previews": previews,
        "zip_name": zip_name,
    }


@app.route("/")
def index():
    cpu_count = os.cpu_count() or 4
    options = sorted({1, 2, 4, max(1, cpu_count // 2), cpu_count})
    default_threads = max(1, min(cpu_count, max(2, cpu_count - 1)))
    has_zip = (DEFAULT_OUTPUT / "white_bg_output_folder.zip").exists()
    return render_template_string(
        HTML_TEMPLATE,
        default_input=str(DEFAULT_INPUT),
        cpu_count=cpu_count,
        thread_options=options,
        default_threads=default_threads,
        has_zip=has_zip,
    )


@app.route("/api/existing-output", methods=["GET"])
def api_existing_output():
    zip_path = DEFAULT_OUTPUT / "white_bg_output_folder.zip"
    zip_name = zip_path.name if zip_path.exists() else rebuild_zip(DEFAULT_OUTPUT)
    return jsonify({
        "previews": collect_existing_previews(DEFAULT_OUTPUT),
        "zip_name": zip_name,
    })


@app.route("/api/process-local", methods=["POST"])
def api_process_local():
    low_thresh = float(request.form.get("low_thresh", 0.45))
    high_thresh = float(request.form.get("high_thresh", 0.78))
    num_threads = int(request.form.get("num_threads", os.cpu_count() or 4))
    skip_existing = request.form.get("skip_existing", "0") == "1"
    local_dir = Path(request.form.get("local_dir", str(DEFAULT_INPUT)))

    if not local_dir.exists() or not local_dir.is_dir():
        return jsonify({"error": f"Folder not found: {local_dir}"}), 400

    try:
        get_session(num_threads=num_threads)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    items = []
    for p in sorted(local_dir.rglob("*")):
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS:
            items.append((p.relative_to(local_dir).as_posix(), str(p)))

    result = run_batch_from_items(items, DEFAULT_OUTPUT, low_thresh, high_thresh, num_threads, skip_existing)
    return jsonify(result)


@app.route("/api/process-folder", methods=["POST"])
def api_process_folder():
    low_thresh = float(request.form.get("low_thresh", 0.45))
    high_thresh = float(request.form.get("high_thresh", 0.78))
    num_threads = int(request.form.get("num_threads", os.cpu_count() or 4))
    skip_existing = request.form.get("skip_existing", "0") == "1"

    try:
        get_session(num_threads=num_threads)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    uploaded_files = request.files.getlist("files")
    rel_paths = request.form.getlist("paths")
    if not uploaded_files:
        return jsonify({"error": "No files received from folder."}), 400

    items = []
    for idx, f in enumerate(uploaded_files):
        rel_str = rel_paths[idx] if idx < len(rel_paths) and rel_paths[idx] else f.filename
        rel_p = Path(rel_str.replace("\\", "/"))
        parts = rel_p.parts
        sub_rel = Path(*parts[1:]).as_posix() if len(parts) > 1 else rel_p.as_posix()
        items.append((sub_rel, f.stream))

    result = run_batch_from_items(items, DEFAULT_OUTPUT, low_thresh, high_thresh, num_threads, skip_existing)
    return jsonify(result)


@app.route("/api/download/<zip_name>")
def download_zip(zip_name):
    safe_name = Path(zip_name).name
    target = DEFAULT_OUTPUT / safe_name
    if not target.exists():
        return "ZIP file not found", 404
    return send_file(target, as_attachment=True, download_name=safe_name)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
