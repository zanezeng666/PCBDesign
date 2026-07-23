/* ═══════════════════════════════════════════════════════════════════════
   Battery Protection Board Designer — Interactive App Logic
   Flow: Upload → Input dimensions → Scan black frame → Calibrate → Generate
   ═══════════════════════════════════════════════════════════════════════ */

const API = '/api/vision';
let state = {
  calibration: { front: null, back: null },
  detection: { front: null, back: null },
  padDetection: { front: null, back: null },
  frameW: 60, frameH: 30,
  frameScanned: false,
  frameScanResult: null,
  rectifiedPreview: { front: false, back: false },
  originalSrc: { front: '', back: '' },
};

/** Safely extract an error message from a failed Response. */
async function safeErrorText(res) {
  try {
    const body = await res.json();
    if (body.error) {
      let msg = body.error.message || '';
      const details = body.error.details;
      if (details && typeof details === 'object') {
        const parts = [];
        if (details.detected_aspect !== undefined) parts.push(`检测宽高比=${details.detected_aspect}`);
        if (details.expected_aspect !== undefined) parts.push(`期望=${details.expected_aspect}`);
        if (details.aspect_error_pct !== undefined) parts.push(`误差=${details.aspect_error_pct}%`);
        if (details.detected_w_px !== undefined) parts.push(`检测=${details.detected_w_px}x${details.detected_h_px}px`);
        if (parts.length) msg += ' [' + parts.join(', ') + ']';
      }
      return msg || body.detail || JSON.stringify(body);
    }
    return body.detail || JSON.stringify(body);
  } catch {
    try {
      const text = await res.text();
      return (text || '').substring(0, 200) || `HTTP ${res.status}`;
    } catch {
      return `HTTP ${res.status}`;
    }
  }
}

// ═══ Import Helpers ════════════════════════════════════════════════
function applySuggestedDimensions(w, h) {
  document.getElementById('frame-w').value = w;
  document.getElementById('frame-h').value = h;
  state.frameW = w;
  state.frameH = h;
  toast(`尺寸已更新为 ${w}×${h}mm，请重新扫描`, 'info');
  scanBlackFrame();
}

// ═══ Init ═══════════════════════════════════════════════════════
document.addEventListener('DOMContentLoaded', () => {
  loadIcCatalog();
  setupUploadZones();
  setupFrameControls();
  setupCalibrateButtons();
  setupExtractPcbButtons();
  setupDetectHolesButtons();
  setupPadButtons();
  setupExport();
  setupRectifyToggles();
  runSimulation();
});

// ═══ Toast ═════════════════════════════════════════════════════
function toast(msg, kind = '') {
  const el = document.getElementById('toast');
  el.textContent = msg;
  el.className = 'toast show ' + kind;
  setTimeout(() => el.classList.remove('show'), 3000);
}

// ═══ Pre-Simulation on Page Load ═══════════════════════════════
async function runSimulation() {
  const simSection = document.getElementById('step-sim');
  const simIcon = document.getElementById('sim-icon');
  const simText = document.getElementById('sim-text');
  const simDetails = document.getElementById('sim-details');

  simSection.hidden = false;
  simIcon.textContent = '⏳';
  simText.textContent = '正在模拟完整流程，确保一切就绪...';

  try {
    const form = new FormData();
    form.append('frame_w_mm', state.frameW);
    form.append('frame_h_mm', state.frameH);

    const res = await fetch('/api/simulate', { method: 'POST', body: form });
    const result = await res.json();

    if (!result.success) {
      simIcon.textContent = '⚠️';
      simText.textContent = `模拟完成：${result.total_images}张图片中有问题，请检查详情`;
    } else {
      simIcon.textContent = '✅';
      simText.textContent = `模拟通过！${result.total_images}张图片全部检测成功`;

      // Auto-scan succeeded → enable the UI
      state.frameScanned = true;
      state.frameScanResult = result;

      // Update frame comparison
      updateFrameComparison(result);

      // Show Step 3
      document.getElementById('step-detect').hidden = false;
      document.getElementById('step-export').hidden = false;

      // Enable calibrate buttons for images that were calibrated
      for (const step of result.steps) {
        const side = step.side === 'front' ? 'front' :
                     step.side === 'back' ? 'back' : null;
        if (side && step.calibration_success) {
          state.calibration[side] = {
            calibration_id: step.calibration_id,
            pixels_per_mm: step.pixels_per_mm,
            width_mm: step.rectified_w_mm,
            height_mm: step.rectified_h_mm,
            confidence: step.confidence,
            rectified_png_base64: step.rectified_png_base64,
          };
          document.getElementById(`btn-calibrate-${side}`).disabled = false;
          if (step.rectified_png_base64) {
            showRectifiedBadge(side);
          }
          document.getElementById(`btn-pads-${side}`).disabled = false;
        }
      }
    }

    // Build details HTML
    let detailsHtml = '';
    for (const step of result.steps) {
      const sideLabel = step.side === 'front' ? '正面' :
                        step.side === 'back' ? '背面' : step.side;
      const frameOk = step.frame_detected && step.aspect_ok
                    ? '<span class="pass">✔ 黑框检测通过</span>'
                    : '<span class="fail">✘ 黑框检测失败</span>';
      const calOk = step.calibration_success
                  ? '<span class="pass">✔ 标定成功</span>'
                  : `<span class="fail">✘ 标定失败: ${step.calibration_error_msg || 'N/A'}</span>`;
      const aspectInfo = step.frame_detected
        ? `检测=${step.detected_aspect_ratio} vs 期望=${step.expected_aspect_ratio} (误差${step.aspect_error_pct}%)`
        : '未检测到方框';

      let hintHtml = '';
      if (step.orientation_hint) {
        hintHtml = `<br><span class="warn" style="display:inline-block;margin-top:4px">💡 ${step.orientation_hint}</span>`;
      }
      if (step.suggested_w_mm && step.suggested_h_mm) {
        hintHtml += `<br><button class="btn btn-sm" style="margin-top:4px;padding:2px 8px;font-size:.7rem"
          onclick="applySuggestedDimensions(${step.suggested_w_mm},${step.suggested_h_mm})">
          直接应用 ${step.suggested_w_mm}×${step.suggested_h_mm}mm</button>`;
      }

      detailsHtml += `<div class="sim-step">
        📄 <strong>${step.image}</strong> (${sideLabel}):
        ${frameOk} ·
        ${calOk} ·
        <span class="info">${aspectInfo}</span>
        ${hintHtml}
      </div>`;
    }
    simDetails.innerHTML = detailsHtml;

  } catch (e) {
    simIcon.textContent = '❌';
    simText.textContent = `模拟失败: ${e.message}`;
    simDetails.innerHTML = `<div class="sim-step"><span class="fail">${e.message}</span></div>`;
  }
}

function updateFrameComparison(result) {
  const compareEl = document.getElementById('frame-compare');
  compareEl.hidden = false;

  // Aggregate info from the first successful step
  const okSteps = result.steps.filter(s => s.frame_detected);
  if (okSteps.length > 0) {
    const s = okSteps[0];
    document.getElementById('cmp-expected').textContent =
      `${s.frame_w_mm}×${s.frame_h_mm} mm (宽高比 ${s.expected_aspect_ratio})`;
    document.getElementById('cmp-detected').textContent =
      `~${s.detected_w_px}×${s.detected_h_px} px (宽高比 ${s.detected_aspect_ratio})`;
    document.getElementById('cmp-aspect-error').textContent = `${s.aspect_error_pct}%`;
    document.getElementById('cmp-aspect-error').className =
      'compare-value ' + (s.aspect_ok ? 'ok' : 'ng');
    document.getElementById('cmp-status').textContent = s.aspect_ok ? '✅ 匹配' : '❌ 不匹配';
    document.getElementById('cmp-status').className =
      'compare-value ' + (s.aspect_ok ? 'ok' : 'ng');
  } else {
    document.getElementById('cmp-expected').textContent = `${result.frame_w_mm}×${result.frame_h_mm} mm`;
    document.getElementById('cmp-detected').textContent = '—';
    document.getElementById('cmp-aspect-error').textContent = '—';
    document.getElementById('cmp-status').textContent = '❌ 未检测到';
    document.getElementById('cmp-status').className = 'compare-value ng';
  }
}

// ═══ Load IC Catalog ════════════════════════════════════════════
async function loadIcCatalog() {
  try {
    const res = await fetch('/data/ic_catalog/.index.json');
    if (!res.ok) return;
    const list = await res.json();
    const icSel = document.getElementById('param-ic');
    (list || []).forEach(item => {
      const opt = document.createElement('option');
      opt.value = item.model || item;
      opt.textContent = item.model || item;
      icSel.appendChild(opt);
    });
  } catch (e) { /* silent */ }

  try {
    const res = await fetch('/api/mos/list');
    if (res.ok) {
      const mosList = await res.json();
      const mosSel = document.getElementById('param-mos');
      (mosList || []).forEach(m => {
        const opt = document.createElement('option');
        opt.value = m.model || m;
        opt.textContent = m.model || m;
        mosSel.appendChild(opt);
      });
    }
  } catch (e) { /* silent */ }
}

// ═══ Image Upload ═══════════════════════════════════════════════
function setupUploadZones() {
  for (const side of ['front', 'back']) {
    const zone = document.getElementById(`zone-${side}`);
    const fileInput = document.getElementById(`file-${side}`);
    const preview = document.getElementById(`preview-${side}`);

    zone.addEventListener('click', (e) => {
      if (e.target.closest('button')) return; // don't open file dialog for button clicks
      fileInput.click();
    });

    zone.addEventListener('dragover', e => { e.preventDefault(); zone.style.borderColor = '#3b82f6'; });
    zone.addEventListener('dragleave', () => zone.style.borderColor = '');
    zone.addEventListener('drop', e => {
      e.preventDefault();
      zone.style.borderColor = '';
      if (e.dataTransfer.files.length) handleFile(side, e.dataTransfer.files[0]);
    });

    fileInput.addEventListener('change', () => {
      if (fileInput.files.length) handleFile(side, fileInput.files[0]);
    });
  }
}

function handleFile(side, file) {
  if (!file.type.match(/image\/(jpeg|png)/)) {
    toast('仅支持 JPEG/PNG 格式', 'error');
    return;
  }
  const reader = new FileReader();
  reader.onload = e => {
    const preview = document.getElementById(`preview-${side}`);
    preview.src = e.target.result;
    state.originalSrc[side] = e.target.result;
    document.getElementById(`zone-${side}`).classList.add('has-image');
    // Hide rectified badge on new upload
    // Show rectify toggle button immediately after upload
    document.getElementById(`badge-rectify-${side}`).hidden = true;
    document.getElementById(`btn-rectify-${side}`).hidden = false;
    state.rectifiedPreview[side] = false;
  };
  reader.readAsDataURL(file);

  // NO auto-scan — user must input dimensions first in Step 2
  checkCalibrateReady();
}

// ═══ Frame Controls: Dimensions First, Then Scan ═══════════════
function setupFrameControls() {
  const wInput = document.getElementById('frame-w');
  const hInput = document.getElementById('frame-h');

  wInput.addEventListener('input', e => {
    state.frameW = parseInt(e.target.value) || 60;
    document.getElementById('frame-w').value = state.frameW;
  });
  hInput.addEventListener('input', e => {
    state.frameH = parseInt(e.target.value) || 30;
    document.getElementById('frame-h').value = state.frameH;
  });

  // Initialize with current state
  wInput.value = state.frameW;
  hInput.value = state.frameH;

  document.getElementById('btn-scan-frame').addEventListener('click', scanBlackFrame);
}

async function scanBlackFrame() {
  const w = state.frameW;
  const h = state.frameH;

  // Validate dimensions first
  if (!w || !h || w < 10 || h < 10 || w > 200 || h > 200) {
    toast('请先填入有效的黑色方框尺寸 (10-200mm)', 'error');
    return;
  }

  const infoEl = document.getElementById('frame-info');
  infoEl.innerHTML = '<span class="spinner"></span> 正在扫描黑色方框...';
  infoEl.className = 'frame-info';

  const compareEl = document.getElementById('frame-compare');
  compareEl.hidden = true;

  // Scan all uploaded images
  const sides = [];
  const frontFile = document.getElementById('file-front').files[0];
  const backFile = document.getElementById('file-back').files[0];
  if (frontFile) sides.push({ side: 'front', file: frontFile });
  if (backFile) sides.push({ side: 'back', file: backFile });

  if (sides.length === 0) {
    toast('请先上传图片', 'error');
    infoEl.innerHTML = '<span class="muted">请先上传图片，再扫描方框</span>';
    return;
  }

  let anyFound = false;
  let allResults = [];
  let firstOkResult = null;

  for (const { side, file } of sides) {
    const form = new FormData();
    form.append('file', file);

    try {
      const res = await fetch(`${API}/preview-black-frame`, { method: 'POST', body: form });
      if (!res.ok) {
        const msg = await safeErrorText(res);
        allResults.push({ side, found: false, error: msg });
        continue;
      }
      const data = await res.json();
      allResults.push({ side, ...data });

      if (data.found) {
        anyFound = true;

        // Compare aspect ratio
        const detectedAspect = data.aspect_ratio;
        const expectedAspect = w / Math.max(h, 1);
        const aspectError = Math.abs(detectedAspect - expectedAspect) / Math.max(expectedAspect, 0.1);
        const aspectOk = aspectError <= 0.35;

        if (!firstOkResult && aspectOk) {
          firstOkResult = {
            side,
            expectedAspect,
            detectedAspect,
            aspectError,
            aspectOk,
            avgW: data.avg_width_px,
            avgH: data.avg_height_px,
          };
        }
      }
    } catch (e) {
      allResults.push({ side, found: false, error: e.message });
    }
  }

  // Update frame comparison section
  if (firstOkResult) {
    compareEl.hidden = false;
    document.getElementById('cmp-expected').textContent =
      `${w}×${h} mm (宽高比 ${firstOkResult.expectedAspect.toFixed(3)})`;
    document.getElementById('cmp-detected').textContent =
      `~${firstOkResult.avgW.toFixed(0)}×${firstOkResult.avgH.toFixed(0)} px (宽高比 ${firstOkResult.detectedAspect.toFixed(3)})`;
    document.getElementById('cmp-aspect-error').textContent = `${(firstOkResult.aspectError * 100).toFixed(1)}%`;
    document.getElementById('cmp-aspect-error').className =
      'compare-value ' + (firstOkResult.aspectOk ? 'ok' : 'ng');
    document.getElementById('cmp-status').textContent =
      firstOkResult.aspectOk ? '✅ 匹配 — 宽高比一致' : '⚠️ 不匹配 — 请检查尺寸';
    document.getElementById('cmp-status').className =
      'compare-value ' + (firstOkResult.aspectOk ? 'ok' : 'ng');
  } else if (allResults.length > 0) {
    compareEl.hidden = false;
    document.getElementById('cmp-expected').textContent = `${w}×${h} mm`;
    document.getElementById('cmp-detected').textContent = '—';
    document.getElementById('cmp-aspect-error').textContent = '—';
    document.getElementById('cmp-status').textContent = '❌ 未检测到黑色方框';
    document.getElementById('cmp-status').className = 'compare-value ng';
  }

  if (anyFound) {
    state.frameScanned = true;
    // Build detailed message
    let msg = [];
    for (const r of allResults) {
      if (r.found) {
        const epx = w / Math.max(h, 1);
        const err = Math.abs(r.aspect_ratio - epx) / Math.max(epx, 0.1);
        msg.push(`${r.side === 'front' ? '正面' : '背面'}: 检测到 (宽高比${r.aspect_ratio.toFixed(2)}, 误差${(err*100).toFixed(1)}%)`);
      } else {
        msg.push(`${r.side === 'front' ? '正面' : '背面'}: 未检测到`);
      }
    }
    infoEl.innerHTML = '✅ ' + msg.join(' | ');
    infoEl.className = 'frame-info success';

    // Show Step 3 and enable calibrate buttons
    document.getElementById('step-detect').hidden = false;
    document.getElementById('step-export').hidden = false;
    checkCalibrateReady();
  } else {
    infoEl.innerHTML = '⚠️ 未检测到黑色方框，请确认图片和尺寸';
    infoEl.className = 'frame-info error';
  }
}

function checkCalibrateReady() {
  const hasFront = !!document.getElementById('file-front').files[0];
  const hasBack = !!document.getElementById('file-back').files[0];
  document.getElementById('btn-calibrate-front').disabled = !hasFront;
  document.getElementById('btn-calibrate-back').disabled = !hasBack;
}

// ═══ Rectification Preview Toggle ═════════════════════════════
function setupRectifyToggles() {
  for (const side of ['front', 'back']) {
    const btn = document.getElementById(`btn-rectify-${side}`);
    btn.addEventListener('click', (e) => toggleRectifyPreview(side, e));
  }
}

function showRectifiedBadge(side) {
  document.getElementById(`badge-rectify-${side}`).hidden = false;
  document.getElementById(`btn-rectify-${side}`).hidden = false;
}

function toggleRectifyPreview(side, e) {
  // Stop event bubbling to prevent upload zone from opening file dialog
  if (e) e.stopPropagation();

  const calData = state.calibration[side];
  const preview = document.getElementById(`preview-${side}`);
  const btn = document.getElementById(`btn-rectify-${side}`);
  const isRectified = state.rectifiedPreview[side];

  if (isRectified) {
    // Switch back to original
    preview.src = state.originalSrc[side] || '';
    btn.textContent = '🔄 矫正预览';
    btn.classList.remove('active');
    state.rectifiedPreview[side] = false;
  } else {
    if (calData && calData.rectified_png_base64) {
      // Show rectified image (calibration already done)
      preview.src = `data:image/png;base64,${calData.rectified_png_base64}`;
      btn.textContent = '🔄 显示原图';
      btn.classList.add('active');
      state.rectifiedPreview[side] = true;
    } else {
      // No calibration yet — just show original in "preview mode"
      btn.textContent = '🔄 显示原图';
      btn.classList.add('active');
      state.rectifiedPreview[side] = true;
    }
  }
}

// ═══ Calibrate + Detect Outline & Holes ═════════════════════
function setupCalibrateButtons() {
  document.getElementById('btn-calibrate-front').addEventListener('click', () => calibrateSide('front'));
  document.getElementById('btn-calibrate-back').addEventListener('click', () => calibrateSide('back'));
}

async function calibrateSide(side) {
  const file = document.getElementById(`file-${side}`).files[0];
  if (!file) { toast('请先上传图片', 'error'); return; }

  const btn = document.getElementById(`btn-calibrate-${side}`);
  const overlay = document.getElementById(`overlay-${side}`);
  const stats = document.getElementById(`stats-${side}`);
  btn.classList.add('loading');
  btn.disabled = true;
  overlay.innerHTML = '<span class="spinner"></span> 标定中...';
  overlay.classList.remove('hidden');

  try {
    // Step 1: calibrate with black frame (rectification only)
    const calForm = new FormData();
    calForm.append('file', file);
    calForm.append('frame_w_mm', state.frameW);
    calForm.append('frame_h_mm', state.frameH);

    const calRes = await fetch(`${API}/calibrate-black-frame`, { method: 'POST', body: calForm });
    if (!calRes.ok) {
      const errMsg = await safeErrorText(calRes);
      throw new Error(`标定失败 (${calRes.status}): ${errMsg}`);
    }
    const calData = await calRes.json();
    state.calibration[side] = calData;

    // Show rectified preview toggle
    showRectifiedBadge(side);
    // Auto-switch to rectified preview
    toggleRectifyPreview(side);

    stats.innerHTML = `分辨率 ${calData.pixels_per_mm?.toFixed(1)} px/mm — 点击下方按钮逐步识别`;

    overlay.classList.add('hidden');
    drawCalibrationCanvas(side, calData, null);

    // Enable the step-by-step identification buttons
    document.getElementById(`btn-extract-pcb-${side}`).disabled = false;
    document.getElementById(`btn-detect-holes-${side}`).disabled = true;
    document.getElementById(`btn-pads-${side}`).disabled = true;

    // Reset step results
    document.getElementById(`extract-result-${side}`).hidden = true;
    document.getElementById(`holes-result-${side}`).hidden = true;

    toast(`✅ ${side === 'front' ? '正面' : '背面'} 透视矫正完成`, 'success');
  } catch (e) {
    overlay.innerHTML = `❌ ${e.message}`;
    toast(e.message, 'error');
  } finally {
    btn.classList.remove('loading');
    btn.disabled = false;
  }
}

// ═══ Canvas Drawing ═══════════════════════════════════════════
function drawCalibrationCanvas(side, calData, detData) {
  const canvas = document.getElementById(`canvas-${side}`);
  const ctx = canvas.getContext('2d');
  canvas.width = canvas.parentElement.clientWidth;
  canvas.height = canvas.parentElement.clientHeight;

  if (calData.rectified_png_base64) {
    const img = new Image();
    img.onload = () => {
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      const scale = Math.min(canvas.width / img.width, canvas.height / img.height);
      const dx = (canvas.width - img.width * scale) / 2;
      const dy = (canvas.height - img.height * scale) / 2;
      ctx.drawImage(img, dx, dy, img.width * scale, img.height * scale);

      if (detData?.outline?.length) {
        drawPolygon(ctx, detData.outline[0], calData, scale, dx, dy, '#00ff88', 2.5, 'rgba(0,255,136,.15)');
      }
      (detData?.holes || []).forEach(h => {
        drawPolygon(ctx, h, calData, scale, dx, dy, '#ff6b6b', 1.5, 'rgba(255,107,107,.2)');
      });
    };
    img.src = `data:image/png;base64,${calData.rectified_png_base64}`;
  }
}

function drawPadCanvas(side, calData, padDetect) {
  const canvas = document.getElementById(`canvas-${side}`);
  const ctx = canvas.getContext('2d');
  canvas.width = canvas.parentElement.clientWidth;
  canvas.height = canvas.parentElement.clientHeight;

  if (calData.rectified_png_base64) {
    const img = new Image();
    img.onload = () => {
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      const scale = Math.min(canvas.width / img.width, canvas.height / img.height);
      const dx = (canvas.width - img.width * scale) / 2;
      const dy = (canvas.height - img.height * scale) / 2;
      ctx.drawImage(img, dx, dy, img.width * scale, img.height * scale);

      const det = state.detection[side];
      if (det?.outline?.length) {
        drawPolygon(ctx, det.outline[0], calData, scale, dx, dy, '#00ff88', 2, 'rgba(0,255,136,.1)');
      }
      if (det?.holes) {
        det.holes.forEach(h => drawPolygon(ctx, h, calData, scale, dx, dy, '#ff6b6b', 1, 'rgba(255,107,107,.15)'));
      }

      const pads = padDetect?.pads || [];
      const colors = ['#facc15', '#60a5fa', '#f472b6', '#34d399', '#a78bfa',
                       '#fb923c', '#38bdf8', '#f87171'];
      pads.forEach((pad, i) => {
        const color = colors[i % colors.length];
        drawPolygon(ctx, pad, calData, scale, dx, dy, color, 2.5, 'rgba(255,255,255,.08)');

        const vr = pad.visible_region || {};
        const center = vr.center || pad.visible_position || {};
        const cx = center.x_mm * (calData.pixels_per_mm || 1) * scale + dx;
        const cy = center.y_mm * (calData.pixels_per_mm || 1) * scale + dy;
        const label = pad.label || `P${i+1}`;

        ctx.fillStyle = color;
        ctx.font = `bold ${Math.max(11, 14*scale)}px sans-serif`;
        const tw = ctx.measureText(label).width;
        ctx.fillStyle = 'rgba(0,0,0,.7)';
        ctx.fillRect(cx - tw/2 - 4, cy - 8, tw + 8, 18);
        ctx.fillStyle = color;
        ctx.fillText(label, cx - tw/2, cy + 5);
      });
    };
    img.src = `data:image/png;base64,${calData.rectified_png_base64}`;
  }
}

function drawPolygon(ctx, item, calData, scale, dx, dy, strokeColor, lineWidth, fillColor) {
  const vr = item.visible_region || item;
  const poly = vr.polygon || [];
  if (poly.length < 3) return;

  const ppm = calData.pixels_per_mm || 1;

  ctx.beginPath();
  poly.forEach((pt, i) => {
    const px = pt.x_mm * ppm * scale + dx;
    const py = pt.y_mm * ppm * scale + dy;
    if (i === 0) ctx.moveTo(px, py);
    else ctx.lineTo(px, py);
  });
  ctx.closePath();

  if (fillColor) { ctx.fillStyle = fillColor; ctx.fill(); }
  ctx.strokeStyle = strokeColor;
  ctx.lineWidth = lineWidth;
  ctx.lineJoin = 'round';
  ctx.stroke();
}

// ═══ Step-by-Step PCB Extraction ══════════════════════════════
function setupExtractPcbButtons() {
  for (const side of ['front', 'back']) {
    document.getElementById(`btn-extract-pcb-${side}`).addEventListener('click', () => extractPcb(side));
  }
}

async function extractPcb(side) {
  const calData = state.calibration[side];
  if (!calData) { toast('请先完成标定', 'error'); return; }

  const btn = document.getElementById(`btn-extract-pcb-${side}`);
  const resultDiv = document.getElementById(`extract-result-${side}`);
  const stats = document.getElementById(`stats-${side}`);
  btn.classList.add('loading');
  btn.disabled = true;

  try {
    const form = new FormData();
    form.append('calibration_id', calData.calibration_id);

    const res = await fetch(`${API}/extract-pcb`, { method: 'POST', body: form });
    if (!res.ok) throw new Error(`轮廓提取失败 (${res.status}): ${await safeErrorText(res)}`);
    const data = await res.json();

    // Store outline in detection state
    if (!state.detection[side]) state.detection[side] = {};
    state.detection[side].outline = [{
      visible_region: { polygon: data.outline, type: 'board_outline' },
      grooves: data.grooves || [],
    }];
    state.detection[side]._pcbCutout = data;

    // Build step-by-step result HTML
    let html = '';
    const stepLabels = ['① 去除白色底色', '② 去除阴影', '③ 识别凹槽'];
    (data.steps || []).forEach((step, i) => {
      const s = step.stats || {};
      let statsHtml = '';
      if (s.foreground_ratio !== undefined) statsHtml += `<span>前景: ${(s.foreground_ratio * 100).toFixed(1)}%</span>`;
      if (s.shadow_removed_pct !== undefined) statsHtml += `<span>阴影: -${s.shadow_removed_pct}%</span>`;
      if (s.groove_count !== undefined) statsHtml += `<span>凹槽: ${s.groove_count}</span>`;
      if (s.solidity !== undefined) statsHtml += `<span>solidity: ${s.solidity}</span>`;

      html += `<div class="step-item">
        <span class="step-num done">${i + 1}</span>
        <img class="step-thumb" src="data:image/png;base64,${step.mask_png_base64}" alt="${step.step_name}" title="点击放大"
             onclick="window.open('data:image/png;base64,${step.mask_png_base64}', '_blank')">
        <div class="step-info">
          <div class="step-title">${stepLabels[i] || step.step_name}</div>
          <div class="step-desc">${step.description}</div>
          <div class="step-stats">${statsHtml}</div>
        </div>
      </div>`;
    });

    // Add transparent PNG preview
    if (data.transparent_png_base64) {
      html += `<div class="transparent-preview">
        <img src="data:image/png;base64,${data.transparent_png_base64}"
             alt="PCB透明背景" title="透明PCB图"
             onclick="window.open('data:image/png;base64,${data.transparent_png_base64}', '_blank')">
      </div>`;
    }

    resultDiv.innerHTML = html;
    resultDiv.hidden = false;

    // Update stats
    const grooveCount = (data.grooves || []).length;
    const outlineCount = (data.outline || []).length;
    const ppm = calData.pixels_per_mm?.toFixed(1) || '?';
    stats.innerHTML = `分辨率 ${ppm} px/mm · 轮廓 ${outlineCount}点 · 凹槽 ${grooveCount}个`;

    // Always show result (groove warnings still shown)
    if (data.groove_warning) {
      resultDiv.innerHTML += `<div class="step-item" style="background:#fff3cd">
        <span class="step-num">⚠</span>
        <div class="step-info">
          <div class="step-title" style="color:#cc8800">凹槽提示</div>
          <div class="step-desc">${data.groove_warning}</div>
        </div>
      </div>`;
    }

    // Draw outline on canvas
    drawCalibrationCanvas(side, calData, state.detection[side]);

    // Enable next steps
    document.getElementById(`btn-detect-holes-${side}`).disabled = false;
    document.getElementById(`btn-pads-${side}`).disabled = false;

    toast(`✅ 轮廓提取完成: ${outlineCount}点, ${grooveCount}个凹槽`, 'success');
  } catch (e) {
    toast(e.message, 'error');
    resultDiv.innerHTML = `<div class="step-item"><span class="step-num">!</span><div class="step-info"><div class="step-title" style="color:#e74c3c">提取失败</div><div class="step-desc">${e.message}</div></div></div>`;
    resultDiv.hidden = false;
  } finally {
    btn.classList.remove('loading');
    btn.disabled = false;
  }
}

// ═══ Hole Detection ═══════════════════════════════════════════
function setupDetectHolesButtons() {
  for (const side of ['front', 'back']) {
    document.getElementById(`btn-detect-holes-${side}`).addEventListener('click', () => detectHoles(side));
  }
}

async function detectHoles(side) {
  const calData = state.calibration[side];
  if (!calData) { toast('请先完成标定', 'error'); return; }

  const det = state.detection[side];
  if (!det?.outline?.length) { toast('请先识别轮廓', 'error'); return; }

  const btn = document.getElementById(`btn-detect-holes-${side}`);
  const resultDiv = document.getElementById(`holes-result-${side}`);
  const stats = document.getElementById(`stats-${side}`);
  btn.classList.add('loading');
  btn.disabled = true;

  try {
    const form = new FormData();
    form.append('calibration_id', calData.calibration_id);
    // Send the outline polygon
    const outlinePoly = det.outline[0]?.visible_region?.polygon || det.outline[0] || [];
    form.append('outline_json', JSON.stringify({ outline: outlinePoly }));

    const res = await fetch(`${API}/detect-holes`, { method: 'POST', body: form });
    if (!res.ok) throw new Error(`孔槽检测失败 (${res.status}): ${await safeErrorText(res)}`);
    const data = await res.json();

    state.detection[side].holes = data.holes || [];

    // Build result HTML
    const holes = data.holes || [];
    let html = `<div class="step-item">
      <span class="step-num done">✓</span>
      <div class="step-info">
        <div class="step-title">检测到 ${holes.length} 个孔槽</div>`;

    // Group by type
    const byType = {};
    holes.forEach(h => { byType[h.hole_type || h.shape || '?'] = (byType[h.hole_type || h.shape || '?'] || 0) + 1; });
    const typeStr = Object.entries(byType).map(([t, c]) => `${t}: ${c}`).join(' · ');
    html += `<div class="step-desc">${typeStr}</div>`;

    if (holes.length > 0) {
      html += '<div class="step-stats">';
      holes.slice(0, 8).forEach((h, i) => {
        const c = h.visible_region?.center || h.visible_position || {};
        html += `<span>${h.label || 'h' + (i + 1)}: (${c.x_mm?.toFixed(1) || '?'}, ${c.y_mm?.toFixed(1) || '?'}) ${h.width_mm?.toFixed(1) || ''}×${h.height_mm?.toFixed(1) || ''}mm</span>`;
      });
      if (holes.length > 8) html += `<span>... +${holes.length - 8} more</span>`;
      html += '</div>';
    }
    html += '</div></div>';

    resultDiv.innerHTML = html;
    resultDiv.hidden = false;

    // Update canvas
    drawCalibrationCanvas(side, calData, state.detection[side]);

    // Update stats
    const ppm = calData.pixels_per_mm?.toFixed(1) || '?';
    const outlineCount = (det.outline || []).length;
    stats.innerHTML = `分辨率 ${ppm} px/mm · 轮廓 ${outlineCount}点 · 孔槽 ${holes.length}个`;

    checkGenerateReady();
    toast(`✅ 检测到 ${holes.length} 个孔槽`, 'success');
  } catch (e) {
    toast(e.message, 'error');
    resultDiv.innerHTML = `<div class="step-item"><span class="step-num">!</span><div class="step-info"><div class="step-title" style="color:#e74c3c">检测失败</div><div class="step-desc">${e.message}</div></div></div>`;
    resultDiv.hidden = false;
  } finally {
    btn.classList.remove('loading');
    btn.disabled = false;
  }
}

// ═══ Pad Detection ════════════════════════════════════════════
function setupPadButtons() {
  document.getElementById('btn-pads-front').addEventListener('click', () => detectPads('front'));
  document.getElementById('btn-pads-back').addEventListener('click', () => detectPads('back'));
}

async function detectPads(side) {
  const calData = state.calibration[side];
  if (!calData) { toast('请先完成标定', 'error'); return; }

  const btn = document.getElementById(`btn-pads-${side}`);
  const stats = document.getElementById(`stats-${side}`);
  btn.classList.add('loading');
  btn.disabled = true;

  try {
    // Always call VLM for pad detection
    const form = new FormData();
    form.append('calibration_id', calData.calibration_id);
    form.append('side', side);
    const res = await fetch(`${API}/detect-all`, { method: 'POST', body: form });
    if (!res.ok) throw new Error(`焊盘检测失败 (${res.status}): ${await safeErrorText(res)}`);
    const vlmResult = await res.json();
    state.padDetection[side] = vlmResult;

    // Merge VLM pads into detection state (keep our CV outline/holes)
    if (!state.detection[side]) state.detection[side] = {};
    if (!state.detection[side].outline && vlmResult.outline) {
      state.detection[side].outline = vlmResult.outline;
    }
    if (!state.detection[side].holes && vlmResult.holes) {
      state.detection[side].holes = vlmResult.holes;
    }

    const pads = vlmResult.pads || [];
    const det = state.detection[side] || {};
    const outlineCount = (det.outline || []).length;
    const holeCount = (det.holes || []).length;
    const ppm = calData.pixels_per_mm?.toFixed(1) || '?';
    stats.innerHTML = `分辨率 ${ppm} px/mm · 轮廓 ${outlineCount} · 孔槽 ${holeCount} · 焊盘 ${pads.length}`;

    drawPadCanvas(side, calData, vlmResult);
    renderPadTable(side, pads);

    checkGenerateReady();
    toast(`✅ ${pads.length} 个焊盘已标注`, 'success');
  } catch (e) {
    toast(e.message, 'error');
  } finally {
    btn.classList.remove('loading');
    btn.disabled = false;
  }
}

function renderPadTable(side, pads) {
  const tableDiv = document.getElementById(`table-${side}`);
  const resultsTables = document.getElementById('results-tables');
  if (!pads.length) {
    tableDiv.innerHTML = '<h4>' + (side === 'front' ? '🟢 正面' : '🔴 背面') + ' 焊盘</h4><p class="muted">未检测到焊盘</p>';
    resultsTables.hidden = false;
    return;
  }

  let html = `<h4>${side === 'front' ? '🟢 正面' : '🔴 背面'} 焊盘 (${pads.length}个)</h4>`;
  html += '<table class="pad-table"><thead><tr><th>标签</th><th>中心 X</th><th>中心 Y</th><th>宽度</th><th>高度</th><th>置信度</th></tr></thead><tbody>';
  pads.forEach(p => {
    const vr = p.visible_region || {};
    const center = vr.center || p.visible_position || {};
    const bbox = vr.bbox || {};
    html += `<tr>
      <td><strong>${p.label || '?'}</strong></td>
      <td>${center.x_mm?.toFixed(2) || '-'}</td>
      <td>${center.y_mm?.toFixed(2) || '-'}</td>
      <td>${bbox.width_mm?.toFixed(2) || '-'}</td>
      <td>${bbox.height_mm?.toFixed(2) || '-'}</td>
      <td class="pad-conf">${(p.confidence ?? 0).toFixed(2)}</td>
    </tr>`;
  });
  html += '</tbody></table>';
  tableDiv.innerHTML = html;
  resultsTables.hidden = false;
}

// ═══ Export ════════════════════════════════════════════════════
function setupExport() {
  document.getElementById('btn-generate').addEventListener('click', generateProject);
}

function checkGenerateReady() {
  const frontOk = state.calibration.front && state.detection.front && state.padDetection.front;
  const backOk = state.calibration.back && state.detection.back && state.padDetection.back;
  const hasBack = !!document.getElementById('file-back').files[0];
  // Enable generate when front side has outline + holes + pads
  const frontReady = state.calibration.front && state.detection.front?.outline && state.padDetection.front?.pads;
  document.getElementById('btn-generate').disabled = !frontReady;
}

async function generateProject() {
  const btn = document.getElementById('btn-generate');
  const status = document.getElementById('export-status');
  btn.disabled = true;
  btn.classList.add('loading');
  btn.innerHTML = '<span class="spinner"></span> 生成中...';

  try {
    const spec = buildDesignSpec();
    const res = await fetch('/api/projects', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(spec),
    });
    if (!res.ok) throw new Error(`创建失败 (${res.status}): ${await safeErrorText(res)}`);
    const project = await res.json();

    status.innerHTML = `✅ 项目已创建: <strong>${project.id}</strong>`;
    toast('KiCad 工程已生成！', 'success');
  } catch (e) {
    status.innerHTML = `❌ ${e.message}`;
    toast(e.message, 'error');
  } finally {
    btn.classList.remove('loading');
    btn.disabled = false;
  }
}

function buildDesignSpec() {
  const ic = document.getElementById('param-ic').value || 'DW01-G';
  const cellConfig = document.getElementById('param-cell').value;
  const cellCount = parseInt(cellConfig) || 1;
  const maxCurrent = parseFloat(document.getElementById('param-current').value) || 5;
  const vmin = parseFloat(document.getElementById('param-vmin').value) || 3.0;
  const vmax = parseFloat(document.getElementById('param-vmax').value) || 4.25;

  return {
    protection_ic: ic,
    cell_count: cellCount,
    max_discharge_current_a: maxCurrent,
    undervoltage_threshold_v: vmin,
    overvoltage_threshold_v: vmax,
    port_topology: 'common_port',
    photo_capture: {
      front_calibration_id: state.calibration.front?.calibration_id || '',
      back_calibration_id: state.calibration.back?.calibration_id || '',
      back_transform: 'none',
    },
  };
}
