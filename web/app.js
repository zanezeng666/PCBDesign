/* ──────────────────────────────────────────────────
 *  Battery Designer  —  Frontend App Logic
 *  FLOW:
 *    Step 1: Frame dims → Upload → Preview Frame → Scan Frame → Compare
 *    Step 2: Get PCB per side → Combined Extract Outlines → Holes/Pads/Components → Contour Match
 *    Step 3: Generate KiCad
 * ────────────────────────────────────────────────── */

const $ = (sel, ctx = document) => ctx.querySelector(sel);
const $$ = (sel, ctx = document) => Array.from(ctx.querySelectorAll(sel));

// ── Global state ──
const STATE = {
  autoTest: false,
  frameW: 60, frameH: 30,
  uploaded: { front: false, back: false },
  // Original image blob URLs for toggle preview
  origSrc: { front: null, back: null },
  // Calibration data per side
  cal: { front: null, back: null },
  // Frame detection per side (from preview-black-frame)
  scanned: { front: null, back: null },
  // Extract-pcb results per side
  extract: { front: null, back: null },
  // Holes & pads results per side
  holes: { front: null, back: null },
  pads: { front: null, back: null },
  // Component detection results per side
  components: { front: null, back: null },
  // Rectify toggle state per side (step 1)
  rectShow: { front: true, back: true },
  // Raw File objects for deferred calibration (step 2)
  rawFile: { front: null, back: null },
  // Step 3: design parameters
  cellParams: null,   // AI cell lookup result
  icDevice: null,     // resolved IC device package
  inferredTopology: null,  // 'common' | 'separate' from pad labels
};

// ── Helpers ──
function setBadge(stepId, text, cls) {
  const b = $(`#badge-${stepId}`);
  if (!b) return;
  b.textContent = text;
  b.className = `step-badge ${cls}`;
}
function show(el) { el.hidden = false; }
function hide(el) { el.hidden = true; }
function setHTML(el, html) { el.innerHTML = html; }
function toggle(el, visible) { el.hidden = !visible; }

// ══════════════════════════════════════════════════════════
//  INITIALIZATION
// ══════════════════════════════════════════════════════════
async function init() {
  // Read frame dimensions from inputs
  const fw = $('#frame-w'), fh = $('#frame-h');
  STATE.frameW = parseFloat(fw.value) || 40;
  STATE.frameH = parseFloat(fh.value) || 25;
  fw.addEventListener('input', () => { STATE.frameW = parseFloat(fw.value) || 40; });
  fh.addEventListener('input', () => { STATE.frameH = parseFloat(fh.value) || 25; });

  setupUploadZones();

  // Step 1: rectify toggle inside upload zones
  $('#rect-toggle-front').addEventListener('change', e => {
    STATE.rectShow.front = e.target.checked;
    drawUploadCanvas('front');
  });
  $('#rect-toggle-back').addEventListener('change', e => {
    STATE.rectShow.back = e.target.checked;
    drawUploadCanvas('back');
  });

  // Scan black frame button
  $('#btn-scan-frame').addEventListener('click', scanBlackFrame);

  // Step 2: calibrate buttons per side
  $('#btn-cal-front').addEventListener('click', () => calibrateSide('front'));
  $('#btn-cal-back').addEventListener('click', () => calibrateSide('back'));

  // Step 2: combined extract outlines
  $('#btn-extract-both').addEventListener('click', extractBoth);

  // Step 2: detect all (holes + pads + components) for both sides
  $('#btn-detect-all').addEventListener('click', detectAll);

  // Step 3: design parameter buttons
  $('#btn-cell-lookup').addEventListener('click', lookupCell);
  $('#btn-ic-resolve').addEventListener('click', resolveIc);
  $('#btn-generate').addEventListener('click', generateProject);

  // Auto-test: check if input/front.jpg and input/back.jpg exist
  try {
    const resp = await fetch('/api/health');
    if (!resp.ok) throw new Error('server not ready');
    const imgFront = new Image();
    imgFront.src = '/static/../input/front.jpg?' + Date.now();
    await new Promise((resolve, reject) => {
      imgFront.onload = resolve;
      imgFront.onerror = reject;
      setTimeout(() => reject(new Error('timeout')), 2000);
    });
    STATE.autoTest = true;
    console.log('Auto-test mode: input images found');
    $('#btn-scan-frame').disabled = false;
  } catch {
    STATE.autoTest = false;
    console.log('Upload mode: no pre-placed images');
  }
}

// ══════════════════════════════════════════════════════════
//  STEP 1 — Upload & Rectify Preview
// ══════════════════════════════════════════════════════════
function setupUploadZones() {
  ['front', 'back'].forEach(side => {
    const drop = $(`#drop-${side}`);
    const input = $(`#file-${side}`);
    const status = $(`#status-${side}`);

    drop.addEventListener('click', () => input.click());
    drop.addEventListener('dragover', e => { e.preventDefault(); drop.classList.add('drag-over'); });
    drop.addEventListener('dragleave', () => drop.classList.remove('drag-over'));
    drop.addEventListener('drop', e => {
      e.preventDefault();
      drop.classList.remove('drag-over');
      if (e.dataTransfer.files.length) handleUpload(side, e.dataTransfer.files[0]);
    });
    input.addEventListener('change', () => {
      if (input.files.length) handleUpload(side, input.files[0]);
    });
  });
}

async function handleUpload(side, file) {
  const status = $(`#status-${side}`);
  setHTML(status, '<span class="spinner"></span> 上传中...');

  // Store original image for toggle preview (blob URL)
  if (STATE.origSrc[side]) URL.revokeObjectURL(STATE.origSrc[side]);
  STATE.origSrc[side] = URL.createObjectURL(file);

  // Store file reference for later calibration (step 2)
  STATE.rawFile[side] = file;

  try {
    STATE.uploaded[side] = true;
    setHTML(status, `<span class="ok">已上传 (${(file.size / 1024).toFixed(0)} KB)</span>`);

    // Enable scan button
    $('#btn-scan-frame').disabled = false;
    setBadge('step1', '可扫描', 'ready');

    // Preview black frame to get detection data (no calibration yet)
    const pForm = new FormData();
    pForm.append('file', file);
    pForm.append('frame_w_mm', STATE.frameW);
    pForm.append('frame_h_mm', STATE.frameH);
    try {
      const pResp = await fetch('/api/vision/preview-black-frame', { method: 'POST', body: pForm });
      if (pResp.ok) {
        const pData = await pResp.json();
        STATE.scanned[side] = {
          frame_detected: pData.found,
          detected_aspect_ratio: pData.aspect_ratio || 0,
          avg_width_px: pData.avg_width_px || 0,
          avg_height_px: pData.avg_height_px || 0,
        };
      }
    } catch { /* preview failure is non-fatal */ }

    // Show original uploaded image immediately
    show($(`#rectify-row-${side}`));
    show($(`#canvas-wrap-${side}`));
    $(`#rect-toggle-${side}`).checked = false;
    STATE.rectShow[side] = false;
    drawUploadCanvas(side);

  } catch (err) {
    setHTML(status, `<span class="ng">${err.message}</span>`);
  }
}

/**
 * Draw the upload-zone canvas (step 1).
 * Toggle between original image (from blob URL) and rectified image (from base64).
 */
function drawUploadCanvas(side) {
  const canvas = $(`#canvas-${side}`);
  if (!canvas) return;

  const showRect = STATE.rectShow[side];
  const cal = STATE.cal[side];
  const orig = STATE.origSrc[side];

  let src = null;
  if (cal) {
    const rect = cal.rect_b64 ? `data:image/png;base64,${cal.rect_b64}` : '';
    src = (showRect && rect) ? rect : (orig || rect);
  } else {
    // No calibration yet – show original uploaded image
    src = orig;
  }
  if (!src) return;

  const img = new Image();
  img.onload = () => {
    const maxW = 900, maxH = 520;
    const scale = Math.min(maxW / img.width, maxH / img.height);
    canvas.width = Math.round(img.width * scale);
    canvas.height = Math.round(img.height * scale);
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
  };
  img.src = src;
}

// ══════════════════════════════════════════════════════════
//  STEP 1 — Scan Black Frame & Compare
// ══════════════════════════════════════════════════════════
async function scanBlackFrame() {
  const btn = $('#btn-scan-frame');
  btn.disabled = true;
  btn.textContent = '扫描中...';

  // Auto-test mode: call /api/simulate (reads input/*.jpg)
  if (STATE.autoTest) {
    try {
      const fm = new FormData();
      fm.append('frame_w_mm', STATE.frameW);
      fm.append('frame_h_mm', STATE.frameH);
      const resp = await fetch('/api/simulate', { method: 'POST', body: fm });
      const data = await resp.json();
      console.log('Scan result:', data);

      data.steps.forEach(s => {
        STATE.scanned[s.side] = s;
        if (s.calibration_success) {
          STATE.cal[s.side] = {
            id: s.calibration_id,
            ppm: s.pixels_per_mm,
            rect_b64: s.rectified_png_base64,
            transparent_pcb_b64: s.transparent_pcb_base64 || '',
            transparent_pcb_outline: s.transparent_pcb_outline_mm || [],
            w_mm: s.rectified_w_mm || STATE.frameW,
            h_mm: s.rectified_h_mm || STATE.frameH,
            frameW: s.frame_w_mm || STATE.frameW,
            frameH: s.frame_h_mm || STATE.frameH,
          };
          STATE.uploaded[s.side] = true;
        }
      });
      STATE.contourMatch = data.contour_match || null;

      const allOk = data.steps.every(s => s.calibration_success);
      if (allOk) {
        setHTML($('#frame-info'), '<span class="ok">方框检测成功</span>');
        setBadge('step1', '已完成', 'ready');
      } else {
        const badSide = data.steps.find(s => !s.calibration_success);
        setHTML($('#frame-info'), `<span class="ng">${badSide?.side || ''} 方框检测失败</span>`);
        setBadge('step1', '需修正', 'waiting');
      }
    } catch (err) {
      btn.textContent = '矫正预览';
      btn.disabled = false;
      setHTML($('#frame-info'), `<span class="ng">扫描失败: ${err.message}</span>`);
      return;
    }
  }

  // Upload mode: actually calibrate both sides at once
  if (!STATE.autoTest) {
    const bothOk = STATE.uploaded.front && STATE.uploaded.back;
    if (bothOk) {
      try {
        await Promise.all([
          calibrateSide('front'),
          calibrateSide('back'),
        ]);
        setHTML($('#frame-info'), '<span class="ok">矫正预览完成</span>');
        setBadge('step1', '已完成', 'ready');
      } catch (err) {
        setHTML($('#frame-info'), `<span class="ng">矫正失败: ${err.message}</span>`);
        setBadge('step1', '需修正', 'waiting');
        btn.textContent = '矫正预览';
        btn.disabled = false;
        return;
      }
    } else {
      setHTML($('#frame-info'), '<span class="ng">请先上传正反面图片</span>');
      setBadge('step1', '需修正', 'waiting');
      btn.textContent = '矫正预览';
      btn.disabled = false;
      return;
    }
  }

  // Show frame comparison results (per side, in step 1)
  updateFrameCompare();

  // Auto-test mode: show canvases with rectified images
  if (STATE.autoTest) {
    ['front', 'back'].forEach(side => {
      if (STATE.cal[side]) {
        show($(`#rectify-row-${side}`));
        show($(`#canvas-wrap-${side}`));
        $(`#rect-toggle-${side}`).checked = true;
        STATE.rectShow[side] = true;
        drawUploadCanvas(side);
      }
    });

    // Auto-test mode: enable step 2 (upload mode handles via calibrateSide)
    show($('#step-detect'));
    setBadge('step2', '可获取', 'ready');
    enableCalButtons();
    checkExtractReady();
    checkGenerateReady();
  } else {
    // Upload mode: show step 2 (calibrateSide already enabled cal buttons etc.)
    show($('#step-detect'));
  }

  btn.textContent = STATE.autoTest ? '重新扫描' : '矫正预览';
  btn.disabled = false;
}

function updateFrameCompare() {
  const hasData = STATE.scanned.front || STATE.scanned.back;
  toggle($('#frame-compare-grid'), hasData);
  if (STATE.scanned.front) displayFrameScanResult('front');
  if (STATE.scanned.back) displayFrameScanResult('back');
}

function displayFrameScanResult(side) {
  const data = STATE.scanned[side];
  if (!data) return;
  const container = $(`#frame-comp-${side}-content`);
  if (!container) return;

  const frameDetected = data.frame_detected ?? data.calibration_success;
  if (frameDetected) {
    const ar = data.detected_aspect_ratio || 0;
    const expectedAR = STATE.frameW / STATE.frameH;
    const errPct = expectedAR > 0 ? Math.abs(ar - expectedAR) / expectedAR * 100 : 0;
    const cls = errPct <= 25 ? 'ok' : 'ng';
    setHTML(container, `
      <span class="${cls}">方框已检测</span><br>
      检测尺寸: ${(data.avg_width_px || 0).toFixed(0)} &times; ${(data.avg_height_px || 0).toFixed(0)} px<br>
      宽高比: ${ar.toFixed(3)} (期望 ${expectedAR.toFixed(3)}, 误差 ${errPct.toFixed(1)}%)<br>
      设定: ${STATE.frameW} &times; ${STATE.frameH} mm
    `);
  } else {
    setHTML(container, '<span class="ng">未检测到方框</span>');
  }
}

// ══════════════════════════════════════════════════════════
//  STEP 2 — Get PCB per side
// ══════════════════════════════════════════════════════════
function enableCalButtons() {
  ['front', 'back'].forEach(side => {
    const btn = $(`#btn-cal-${side}`);
    if (STATE.uploaded[side]) {
      btn.disabled = false;
      btn.textContent = side === 'front' ? '获取PCB正面' : '获取PCB反面';
    } else {
      btn.disabled = true;
    }
  });
}

async function calibrateSide(side) {
  const btn = $(`#btn-cal-${side}`);
  btn.textContent = '处理中...';
  btn.disabled = true;
  btn.classList.remove('btn-done');

  // Read file from stored reference
  const file = STATE.rawFile[side];
  if (!file) {
    btn.textContent = '未上传文件';
    btn.disabled = false;
    return;
  }

  const form = new FormData();
  form.append('file', file);
  form.append('frame_w_mm', STATE.frameW);
  form.append('frame_h_mm', STATE.frameH);

  try {
    // ── Phase 1: Black frame calibration (perspective rectification) ──
    const resp = await fetch('/api/vision/calibrate-black-frame', {
      method: 'POST',
      body: form,
    });
    if (!resp.ok) {
      const err = await resp.json().catch(() => ({ detail: resp.statusText }));
      throw new Error(err.detail || err.message || '校准失败');
    }
    const data = await resp.json();

    // Store calibration result (basic HSV extraction as fallback)
    STATE.cal[side] = {
      id: data.calibration_id,
      ppm: data.pixels_per_mm,
      rect_b64: data.rectified_png_base64,
      transparent_pcb_b64: data.transparent_pcb_b64 || '',
      transparent_pcb_outline: data.transparent_pcb_outline_mm || [],
      w_mm: data.rectified_w_mm || STATE.frameW,
      h_mm: data.rectified_h_mm || STATE.frameH,
      frameW: data.rectified_w_mm || STATE.frameW,
      frameH: data.rectified_h_mm || STATE.frameH,
    };

    // Show rectify toggle + rectified canvas in upload zone
    show($(`#rectify-row-${side}`));
    show($(`#canvas-wrap-${side}`));
    STATE.rectShow[side] = true;
    $(`#rect-toggle-${side}`).checked = true;
    drawUploadCanvas(side);

    // ── Phase 2: Paper model shadow removal + refined transparent PCB ──
    // This is the original "标定" logic: build paper background model,
    // subtract shadows, produce clean transparent PCB with accurate outline.
    btn.textContent = '去除阴影中...';
    try {
      const fm2 = new FormData();
      fm2.append('calibration_id', data.calibration_id);
      const resp2 = await fetch('/api/vision/extract-pcb', { method: 'POST', body: fm2 });
      if (resp2.ok) {
        const edata = await resp2.json();
        // Overwrite with paper-model shadow-removed transparent PCB
        if (edata.transparent_pcb_b64) {
          STATE.cal[side].transparent_pcb_b64 = edata.transparent_pcb_b64;
        }
        if (edata.outline && edata.outline.length > 0) {
          STATE.cal[side].transparent_pcb_outline = edata.outline;
        }
        // Show paper model success label
        if (edata.paper_model) {
          show($('#paper-model'));
          setHTML($('#paper-model-content'), `
            <span style="font-size:0.78rem;color:#64748b;">
              ${side === 'front' ? '正面' : '反面'} - 阴影去除完成（纸张底色建模）
            </span>
          `);
        }
      }
    } catch (e) {
      // Non-fatal: continue with basic HSV extraction from Phase 1
      console.warn(`Shadow removal for ${side} (non-fatal):`, e);
    }

    // Show calibration canvas with shadow-removed transparent PCB
    show($(`#cal-canvas-wrap-${side}`));
    drawCalCanvas(side);

    checkExtractReady();
    checkGenerateReady();

    btn.textContent = '获取完成';
    btn.classList.add('btn-done');

    // Check if both sides calibrated
    if (STATE.cal.front && STATE.cal.back) {
      setBadge('step2', '可识别', 'ready');
    }
  } catch (err) {
    btn.textContent = '获取失败';
    btn.disabled = false;
    console.error(`calibrateSide ${side}:`, err);
  }
}

/**
 * Draw calibration canvas in step 2 (shows rectified image)
 */
function drawCalCanvas(side) {
  const cal = STATE.cal[side];
  if (!cal) return;

  const canvas = $(`#cal-canvas-${side}`);
  if (!canvas) return;

  // If pads detected with PCB image, use PCB-only image as background
  const pads = STATE.pads[side];
  const usePcbImg = pads?.pcb_image_b64;

  const src = usePcbImg
    ? `data:image/png;base64,${pads.pcb_image_b64}`
    : (cal.transparent_pcb_b64
      ? `data:image/png;base64,${cal.transparent_pcb_b64}`
      : `data:image/png;base64,${cal.rect_b64}`);

  const img = new Image();
  img.onload = () => {
    const maxW = 900, maxH = 520;
    const scale = Math.min(maxW / img.width, maxH / img.height);
    canvas.width = Math.round(img.width * scale);
    canvas.height = Math.round(img.height * scale);
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(img, 0, 0, canvas.width, canvas.height);

    // Draw pad markers if pads have been detected
    const candidates = pads?.candidates || [];
    if (!candidates.length) return;

    // ppm: use PCB coordinate system if available (PCB-relative coords)
    const ppm = usePcbImg && pads.coordinate_system
      ? img.width / pads.coordinate_system.pcb_width_mm
      : (cal.ppm || (img.width / (cal.frameW || 60)));
    const colors = ['#ef4444', '#3b82f6', '#22c55e', '#f59e0b', '#8b5cf6', '#ec4899'];
    candidates.forEach((c, idx) => {
      const color = colors[idx % colors.length];
      const vr = c.visible_region || c;
      const polygon = vr.polygon;
      const bbox = vr.bbox || {};
      const cx = vr.center?.x_mm ?? (bbox.x_mm + (bbox.width_mm || 0) / 2);
      const cy = vr.center?.y_mm ?? (bbox.y_mm + (bbox.height_mm || 0) / 2);

      // mm → canvas px helper
      const toX = (mm) => mm * ppm * scale;
      const toY = (mm) => mm * ppm * scale;

      ctx.strokeStyle = color;
      ctx.lineWidth = 2;

      // Draw symmetric rounded rect (PCB pads are symmetric)
      // Compute bbox from polygon, then symmetrize around center.
      // Damage (e.g. chipped copper) makes polygon asymmetric;
      // the intact side's max extent determines the true pad size.
      let bx1, by1, bx2, by2;
      if (polygon && polygon.length >= 3) {
        const pxs = polygon.map(p => toX(p.x_mm));
        const pys = polygon.map(p => toY(p.y_mm));
        const cxp = toX(cx), cyp = toY(cy);
        // Max half-extent from center on each axis (symmetry repair)
        let hw = 0, hh = 0;
        for (let i = 0; i < pxs.length; i++) {
          hw = Math.max(hw, Math.abs(pxs[i] - cxp));
          hh = Math.max(hh, Math.abs(pys[i] - cyp));
        }
        bx1 = cxp - hw; by1 = cyp - hh;
        bx2 = cxp + hw; by2 = cyp + hh;
      } else {
        const bw = (bbox.width_mm || 3) * ppm * scale;
        const bh = (bbox.height_mm || 3) * ppm * scale;
        bx1 = toX(cx) - bw / 2; by1 = toY(cy) - bh / 2;
        bx2 = toX(cx) + bw / 2; by2 = toY(cy) + bh / 2;
      }
      const sw = bx2 - bx1, sh = by2 - by1;
      // Corner radius: use unified corner_radius_mm from alignment if available,
      // otherwise estimate from polygon shape with symmetry constraint.
      let r;
      if (c.corner_radius_mm != null) {
        // Backend-computed unified corner radius (mm → canvas px)
        r = c.corner_radius_mm * ppm * scale;
      } else if (polygon && polygon.length >= 4) {
        const short = Math.min(sw, sh);
        const searchR = short * 0.45;
        const corners = [[bx1,by1],[bx2,by1],[bx2,by2],[bx1,by2]];
        const radii = [];
        for (const [ccx, ccy] of corners) {
          let minD = Infinity;
          for (let i = 0; i < polygon.length; i++) {
            const ppx = toX(polygon[i].x_mm), ppy = toY(polygon[i].y_mm);
            if (Math.abs(ppx - ccx) > searchR || Math.abs(ppy - ccy) > searchR) continue;
            const d = Math.hypot(ppx - ccx, ppy - ccy);
            if (d < minD) minD = d;
          }
          if (minD > 0 && minD < Infinity) {
            radii.push(minD / (Math.SQRT2 - 1));
          }
        }
        if (radii.length >= 3) {
          const sorted = [...radii].sort((a,b) => a - b);
          const n = sorted.length;
          const median = n % 2 === 1 ? sorted[Math.floor(n/2)]
            : (sorted[n/2 - 1] + sorted[n/2]) / 2;
          const good = radii.filter(v => v >= 0.6 * median && v <= 1.4 * median);
          const avg = good.length > 0
            ? good.reduce((a,b)=>a+b,0) / good.length
            : median;
          r = Math.min(avg, short / 2);
        } else if (radii.length > 0) {
          r = Math.min(radii.reduce((a,b)=>a+b,0) / radii.length, short / 2);
        } else {
          r = short * 0.1;
        }
      } else {
        r = Math.min(sw, sh) * 0.1;
      }
      r = Math.max(r, 1);

      // Draw rounded rectangle with estimated radius
      ctx.beginPath();
      ctx.moveTo(bx1 + r, by1);
      ctx.lineTo(bx2 - r, by1);
      ctx.arcTo(bx2, by1, bx2, by1 + r, r);
      ctx.lineTo(bx2, by2 - r);
      ctx.arcTo(bx2, by2, bx2 - r, by2, r);
      ctx.lineTo(bx1 + r, by2);
      ctx.arcTo(bx1, by2, bx1, by2 - r, r);
      ctx.lineTo(bx1, by1 + r);
      ctx.arcTo(bx1, by1, bx1 + r, by1, r);
      ctx.closePath();
      ctx.fillStyle = color + '30';
      ctx.fill();
      ctx.stroke();

      // Label text above the pad
      const label = c.label || `#${idx + 1}`;
      const labelY = polygon && polygon.length >= 3
        ? Math.min(...polygon.map(p => toY(p.y_mm))) - 4
        : toY(cy) - (bbox.height_mm || 3) * ppm * scale / 2 - 4;
      ctx.font = 'bold 12px sans-serif';
      ctx.fillStyle = color;
      ctx.textAlign = 'center';
      ctx.fillText(label, toX(cx), labelY);
    });
  };
  img.src = src;
}

// ══════════════════════════════════════════════════════════
//  STEP 2 — Combined Extract Outlines (centered button)
// ══════════════════════════════════════════════════════════
function checkExtractReady() {
  const bothOk = STATE.cal.front && STATE.cal.back;
  const btn = $('#btn-extract-both');
  btn.disabled = !bothOk;
  if (bothOk) {
    setBadge('step2', '可识别', 'ready');
  }
}

async function extractBoth() {
  const btn = $('#btn-extract-both');
  btn.disabled = true;
  btn.textContent = '识别轮廓中（正反面）...';

  try {
    // Pass 1: extract both sides independently (in parallel). This writes each
    // side's pcb_outline.json to disk so the other side can read it.
    const [frontData1, backData1] = await Promise.all([
      extractPcbSide('front'),
      extractPcbSide('back'),
    ]);

    // Pass 2: front⇄back cross-validation (mask consensus). Each side is
    // re-extracted using the other side's outline (now on disk) to remove
    // one-sided edge artifacts (burrs/shadows visible in only one photo).
    let frontData = frontData1, backData = backData1;
    const frontId = STATE.cal.front?.id, backId = STATE.cal.back?.id;
    if (frontId && backId) {
      try {
        btn.textContent = '正反面交叉校验中...';
        const [frontData2, backData2] = await Promise.all([
          extractPcbSide('front', backId),
          extractPcbSide('back', frontId),
        ]);
        frontData = frontData2;
        backData = backData2;
      } catch (e) {
        console.warn('extractBoth: consensus pass failed, using pass-1 results', e);
      }
    }

    STATE.extract.front = frontData;
    STATE.extract.back = backData;

    // Update cal state with refined transparent PCB from extract-pcb
    // so that drawCalCanvas (used by pad detection overlay) shows the
    // contour-extracted transparent image instead of the initial calibration one.
    if (frontData?.transparent_pcb_b64 && STATE.cal.front) {
      STATE.cal.front.transparent_pcb_b64 = frontData.transparent_pcb_b64;
    }
    if (frontData?.outline?.length > 0 && STATE.cal.front) {
      STATE.cal.front.transparent_pcb_outline = frontData.outline;
    }
    if (backData?.transparent_pcb_b64 && STATE.cal.back) {
      STATE.cal.back.transparent_pcb_b64 = backData.transparent_pcb_b64;
    }
    if (backData?.outline?.length > 0 && STATE.cal.back) {
      STATE.cal.back.transparent_pcb_outline = backData.outline;
    }
    // Redraw canvases with updated transparent PCB
    drawCalCanvas('front');
    drawCalCanvas('back');

    // ── Run Edge Chamfer Distance contour matching ──
    // If contourMatch was already set by /api/simulate (auto-test), keep it.
    // Otherwise (upload mode), call the dedicated endpoint.
    if (!STATE.contourMatch || !STATE.contourMatch.message) {
      try {
        const matchFm = new FormData();
        matchFm.append('front_calibration_id', STATE.cal.front?.id || '');
        matchFm.append('back_calibration_id', STATE.cal.back?.id || '');
        const matchResp = await fetch('/api/vision/contour-match', {
          method: 'POST', body: matchFm,
        });
        if (matchResp.ok) {
          STATE.contourMatch = await matchResp.json();
        }
      } catch { /* non-fatal */ }
    }

    // Show per-side results
    displayExtractResult('front', frontData);
    displayExtractResult('back', backData);
    show($('#extract-results-grid'));

    // Enable holes & pads buttons
    enableHolesPadsButtons();

    // Show contour consistency check (Chamfer-based)
    displayContourMatch();

    btn.textContent = '轮廓识别完成';
    btn.classList.add('btn-done');
    setBadge('step2', '识别完成', 'ready');
  } catch (err) {
    btn.textContent = '识别失败，点击重试';
    btn.disabled = false;
    console.error('extractBoth error:', err);
  }
}

async function extractPcbSide(side, otherCalibrationId) {
  const cal = STATE.cal[side];
  if (!cal) throw new Error(`No calibration for ${side}`);

  const fm = new FormData();
  fm.append('calibration_id', cal.id);
  if (otherCalibrationId) {
    // Enable front⇄back cross-validation (mask consensus) on the backend.
    fm.append('other_calibration_id', otherCalibrationId);
  }

  const resp = await fetch('/api/vision/extract-pcb', { method: 'POST', body: fm });
  if (!resp.ok) {
    const err = await resp.json().catch(() => ({ detail: resp.statusText }));
    throw new Error(err.detail || `Extract ${side} failed`);
  }
  const data = await resp.json();

  // Show paper model if available (step 1 area)
  if (data.debug_steps && data.debug_steps['02_paper_model']) {
    const pm = data.debug_steps['02_paper_model'];
    if (pm && pm.base64) {
      show($('#paper-model'));
      setHTML($('#paper-model-content'), `
        <div class="img-grid">
          <img src="data:image/png;base64,${pm.base64}" alt="纸张底色模型" />
        </div>
        <p style="font-size:0.78rem;color:#64748b;margin:4px 0 0 0;">
          ${pm.note || '纸张底色模型提取完成'}
        </p>
      `);
    }
  }

  return data;
}

function displayExtractResult(side, data) {
  const container = $(`#extract-result-${side}`);
  const outline = data.outline || [];
  const grooves = data.grooves || [];
  const consensus = data.consensus_msg || '';
  let imgHtml = '';
  if (data.transparent_pcb_b64) {
    imgHtml = `<div class="img-grid">
      <img src="data:image/png;base64,${data.transparent_pcb_b64}" alt="${side} PCB" />
    </div>`;
  } else if (data.pcb_mask_b64) {
    imgHtml = `<div class="img-grid">
      <img src="data:image/png;base64,${data.pcb_mask_b64}" alt="${side} mask" />
    </div>`;
  }
  setHTML(container, `
    <p style="font-size:0.85rem; margin:4px 0;">
      顶点: <strong>${outline.length}</strong> &nbsp; 槽口: <strong>${grooves.length}</strong>
      ${consensus ? `<br><span style="color:#6366f1;">${consensus}</span>` : ''}
    </p>
    ${imgHtml}
  `);
}

// ── Contour Consistency Check (shown after extract) ──
// Uses Edge Chamfer Distance Matching results from backend
function displayContourMatch() {
  const cm = STATE.contourMatch;
  show($('#contour-match'));

  // Use backend Chamfer data if available
  if (cm && cm.message) {
    const areaPct = cm.mismatch_area_pct != null ? cm.mismatch_area_pct : 0;
    const mergedPts = cm.merged_outline_mm?.length || 0;
    const mergedArea = cm.merged_area_mm2 || 0;
    const ok = cm.ok;

    setHTML($('#contour-match-content'), `
      <p>
        <strong>Edge Chamfer 匹配</strong> —
        <span class="${ok ? 'match-ok' : 'match-ng'}">${ok ? '通过' : '偏差较大'}</span><br>
        ${cm.message}<br>
        合并后轮廓: <strong>${mergedPts}</strong> 顶点 &nbsp;|&nbsp;
        面积: <strong>${mergedArea.toFixed(0)} mm²</strong> &nbsp;|&nbsp;
        偏差: <strong>${areaPct.toFixed(1)}%</strong>
      </p>
    `);
    return;
  }

  // Fallback: no Chamfer data yet (upload mode before scan)
  const frontOutline = STATE.extract.front?.outline || [];
  const backOutline = STATE.extract.back?.outline || [];
  const frontPts = frontOutline.length;
  const backPts = backOutline.length;

  if (frontPts >= 3 && backPts >= 3) {
    const w = STATE.cal.front?.frameW || STATE.frameW;
    const h = STATE.cal.front?.frameH || STATE.frameH;
    setHTML($('#contour-match-content'), `
      <p>
        正面顶点: <strong>${frontPts}</strong> &nbsp;|&nbsp;
        背面顶点: <strong>${backPts}</strong><br>
        方框尺寸: ${w.toFixed(1)} &times; ${h.toFixed(1)} mm<br>
        <span style="color:#94a3b8;font-size:0.78rem;">
          Edge Chamfer 匹配数据待扫描后获取
        </span>
      </p>
    `);
  } else {
    setHTML($('#contour-match-content'), `
      <p class="match-ng">轮廓数据不完整，无法校验
      (正面${frontPts}顶点, 背面${backPts}顶点)</p>
    `);
  }
}

// ══════════════════════════════════════════════════════════
//  STEP 2 — Holes & Pads (per side)
// ══════════════════════════════════════════════════════════
function enableHolesPadsButtons() {
  // 只要任意一面有轮廓提取结果就启用一键识别按钮
  const anyExtract = !!(STATE.extract.front || STATE.extract.back);
  $('#btn-detect-all').disabled = !anyExtract;
}

async function detectAll() {
  const btn = $('#btn-detect-all');
  btn.disabled = true;
  btn.textContent = '识别中...';
  const sides = [];
  if (STATE.extract.front) sides.push('front');
  if (STATE.extract.back) sides.push('back');
  try {
    for (const side of sides) {
      btn.textContent = `识别${side === 'front' ? '正面' : '背面'}中...`;
      await detectHoles(side);
      await detectPads(side);
      await detectComponents(side);
    }
    btn.textContent = '识别完成';
    btn.classList.add('btn-done');
  } catch (e) {
    btn.textContent = '部分识别失败';
  } finally {
    btn.disabled = false;
  }
}

async function detectHoles(side) {
  const btn = document.getElementById(`btn-holes-${side}`);  // may be null (merged into detectAll)
  if (btn) { btn.disabled = true; btn.textContent = '检测中...'; }

  const cal = STATE.cal[side];
  const extract = STATE.extract[side];
  if (!cal || !extract) return;

  try {
    const fm = new FormData();
    fm.append('calibration_id', cal.id);
    fm.append('outline_json', JSON.stringify({ outline: extract.outline }));

    const resp = await fetch('/api/vision/detect-holes', { method: 'POST', body: fm });
    if (!resp.ok) throw new Error('Holes detection failed');
    const data = await resp.json();
    STATE.holes[side] = data;

    setHTML($(`#holes-result-${side}`), `
      <span class="ok">孔位: ${data.hole_count || data.holes?.length || 0}</span>
    `);
    if (btn) { btn.textContent = '重新检测'; btn.disabled = false; }

    updateResultsTables();
  } catch (err) {
    if (btn) { btn.textContent = '检测失败'; btn.disabled = false; }
  }
}

async function detectPads(side) {
  const btn = document.getElementById(`btn-pads-${side}`);  // may be null (merged into detectAll)
  if (btn) { btn.disabled = true; btn.textContent = '识别中...'; }

  const cal = STATE.cal[side];
  if (!cal) return;

  try {
    const fm = new FormData();
    fm.append('calibration_id', cal.id);
    fm.append('side', side);
    const resp = await fetch('/api/vision/detect-terminals', { method: 'POST', body: fm });
    if (!resp.ok) throw new Error('Pad detection failed');
    const data = await resp.json();
    STATE.pads[side] = data;

    const tc = data.candidate_count || data.candidates?.length || 0;
    if (btn) { btn.textContent = `${tc} 焊盘`; btn.classList.add('btn-done'); btn.disabled = false; }

    updateResultsTables();
    checkGenerateReady();
    drawCalCanvas(side);
  } catch (err) {
    if (btn) { btn.textContent = '识别失败'; btn.disabled = false; }
  }
}


// ── Component detection (元器件识别) ──────────────────────
const COMP_TYPE_LABELS = {
  ic: 'IC芯片', mosfet: 'MOSFET', resistor: '电阻', capacitor: '电容',
  diode: '二极管', ntc: 'NTC热敏', led: 'LED', other: '其他'
};

async function detectComponents(side) {
  const btn = document.getElementById(`btn-components-${side}`);  // may be null (merged into detectAll)
  if (btn) { btn.disabled = true; btn.textContent = '识别中...'; }

  const cal = STATE.cal[side];
  if (!cal) return;

  try {
    const fm = new FormData();
    fm.append('calibration_id', cal.id);
    fm.append('side', side);
    const resp = await fetch('/api/vision/detect-components', { method: 'POST', body: fm });
    if (!resp.ok) throw new Error('Component detection failed');
    const data = await resp.json();
    STATE.components[side] = data;

    const comps = data.components || [];
    if (btn) { btn.textContent = `${comps.length} 元器件`; btn.classList.add('btn-done'); btn.disabled = false; }

    // Display component results
    const resultEl = $(`#components-result-${side}`);
    if (resultEl && comps.length > 0) {
      let html = `<div style="margin-top:4px;"><b>${side === 'front' ? '正面' : '背面'}元器件:</b></div>`;
      html += '<table style="font-size:0.8rem; border-collapse:collapse; width:100%; margin-top:4px;">';
      html += '<tr style="background:#f1f5f9;"><th style="padding:2px 6px; text-align:left;">类型</th><th style="padding:2px 6px; text-align:left;">丝印</th><th style="padding:2px 6px; text-align:left;">封装</th><th style="padding:2px 6px;">置信度</th></tr>';
      comps.forEach(c => {
        const label = COMP_TYPE_LABELS[c.type] || c.type;
        const silk = c.silkscreen || '<span style="color:#94a3b8;">未读取</span>';
        const pkg = c.package || '-';
        const conf = (c.confidence * 100).toFixed(0) + '%';
        html += `<tr><td style="padding:2px 6px;">${label}</td><td style="padding:2px 6px; font-family:monospace; font-weight:600;">${silk}</td><td style="padding:2px 6px;">${pkg}</td><td style="padding:2px 6px; text-align:center;">${conf}</td></tr>`;
      });
      html += '</table>';
      setHTML(resultEl, html);
    }

    // Auto-fill IC and MOS model fields
    autoFillFromComponents();

    updateResultsTables();
    checkGenerateReady();
  } catch (err) {
    if (btn) { btn.textContent = '识别失败'; btn.disabled = false; }
    console.error('detectComponents error:', err);
  }
}

function autoFillFromComponents() {
  // Scan both sides for IC and MOSFET components, auto-fill input fields
  let icModel = null;
  let mosModel = null;

  for (const side of ['front', 'back']) {
    const data = STATE.components[side];
    if (!data) continue;
    for (const comp of (data.components || [])) {
      if (comp.type === 'ic' && comp.silkscreen && !icModel) {
        icModel = comp.silkscreen;
      }
      if (comp.type === 'mosfet' && comp.silkscreen && !mosModel) {
        mosModel = comp.silkscreen;
      }
    }
  }

  // Fill IC model field if empty and we found one
  const icInput = $('#ic-model');
  if (icModel && icInput && (!icInput.value || icInput.value === 'DW01')) {
    icInput.value = icModel;
    // Show a hint that it was auto-filled
    icInput.style.borderColor = '#22c55e';
    setTimeout(() => { icInput.style.borderColor = ''; }, 3000);
  }

  // Fill MOS model field if empty and we found one
  const mosInput = $('#mos-model');
  if (mosModel && mosInput && !mosInput.value) {
    mosInput.value = mosModel;
    mosInput.style.borderColor = '#22c55e';
    setTimeout(() => { mosInput.style.borderColor = ''; }, 3000);
  }
}

function updateResultsTables() {
  show($('#results-tables'));
  const content = $('#results-tables-content');
  let html = '<table class="results-table"><thead><tr><th>面</th><th>轮廓顶点</th><th>槽口</th><th>孔位</th><th>焊盘</th><th>元器件</th></tr></thead><tbody>';
  ['front', 'back'].forEach(side => {
    const ext = STATE.extract[side];
    const holes = STATE.holes[side];
    const pads = STATE.pads[side];
    const comps = STATE.components[side];
    html += `<tr>
      <td>${side === 'front' ? '正面' : '背面'}</td>
      <td>${ext?.outline?.length || '-'}</td>
      <td>${ext?.grooves?.length ?? '-'}</td>
      <td>${holes?.hole_count ?? '-'}</td>
      <td>${pads?.candidate_count ?? pads?.candidates?.length ?? '-'}</td>
      <td>${comps?.components?.length ?? '-'}</td>
    </tr>`;
  });
  html += '</tbody></table>';
  setHTML(content, html);
}

// ══════════════════════════════════════════════════════════
//  STEP 3 — Design Parameters & Generate KiCad
// ══════════════════════════════════════════════════════════

// ── Cell lookup (AI) ──
// 将AI返回的电芯参数格式化为分组展示HTML（聚焦保护板设计相关参数，null显示为'-'）
function fmtCellParams(d) {
  const v = (x, unit) => (x === null || x === undefined || x === '') ? '-' : `${x}${unit || ''}`;
  const range = (a, unit) => (Array.isArray(a) && a.length === 2) ? `${a[0]}~${a[1]}${unit || ''}` : '-';
  const L = [];
  L.push(`<b>✅ ${d.manufacturer || ''} ${d.model || ''}</b>`);
  L.push(`【基本】容量 ${v(d.nominal_capacity_mah, 'mAh')} ｜ 标称 ${v(d.nominal_voltage_v, 'V')} ｜ 化学 ${v(d.chemistry)} ｜ 封装 ${v(d.form_factor)}`);
  L.push(`【电压保护】充电截止 ${v(d.charge_cutoff_voltage_v, 'V')} ｜ 放电截止 ${v(d.discharge_cutoff_voltage_v, 'V')}`);
  L.push(`【电流保护】标准充电 ${v(d.standard_charge_current_a, 'A')} ｜ 最大充电 ${v(d.max_charge_current_a, 'A')} ｜ 标准放电 ${v(d.standard_discharge_current_a, 'A')} ｜ 持续放电 ${v(d.max_continuous_discharge_a, 'A')} ｜ 脉冲 ${v(d.max_pulse_discharge_a, 'A')} ｜ 内阻 ${v(d.internal_resistance_mohm, 'mΩ')}`);
  L.push(`【温度保护】充电 ${range(d.operating_temp_charge_c, '℃')} ｜ 放电 ${range(d.operating_temp_discharge_c, '℃')}`);
  if (d.notes) L.push(`<span style="color:#8a8f98">💡 ${d.notes}</span>`);
  return L.join('<br/>');
}

async function lookupCell() {
  const btn = $('#btn-cell-lookup');
  const resultEl = $('#cell-result');
  const manufacturer = $('#cell-manufacturer').value.trim();
  const model = $('#cell-model').value.trim();
  if (!model) {
    show(resultEl);
    resultEl.className = 'param-result result-error';
    setHTML(resultEl, '请输入电芯型号');
    return;
  }
  btn.disabled = true;
  btn.textContent = 'AI查询中...';
  show(resultEl);
  resultEl.className = 'param-result';
  setHTML(resultEl, '<span class="spinner"></span> 正在通过AI获取电芯参数...');
  try {
    const resp = await fetch('/api/cell/lookup', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ manufacturer, model }),
    });
    const data = await resp.json();
    if (!resp.ok) throw new Error(data.error?.message || data.detail || '查询失败');
    STATE.cellParams = data;
    setHTML(resultEl, fmtCellParams(data));
    checkGenerateReady();
  } catch (err) {
    STATE.cellParams = null;
    resultEl.className = 'param-result result-error';
    setHTML(resultEl, `❌ ${err.message}`);
  }
  btn.disabled = false;
  btn.textContent = 'AI查询参数';
}

// ── IC resolve (marking → real MPN) ──
async function resolveIc() {
  const btn = $('#btn-ic-resolve');
  const resultEl = $('#ic-result');
  const model = $('#ic-model').value.trim();
  if (!model) return;
  btn.disabled = true;
  btn.textContent = '解析中...';
  show(resultEl);
  resultEl.className = 'param-result';
  setHTML(resultEl, '正在解析IC型号...');
  try {
    const resp = await fetch(`/api/ic/resolve?model=${encodeURIComponent(model)}`);
    const data = await resp.json();
    if (!resp.ok) throw new Error(data.error?.message || data.detail || '解析失败');
    STATE.icDevice = data;
    const markingNote = data.resolved_from === 'marking'
      ? `<br/>🔍 丝印 <b>${model}</b> → 实际型号 <b>${data.full_mpn}</b>`
      : '';
    setHTML(resultEl, `
      <b>✅ ${data.full_mpn}</b> (${data.manufacturer})<br/>
      封装: ${data.package} ｜ 支持串数: ${data.supported_series.join('/')}S ｜ 端口: ${data.port_topologies.join('/')}${markingNote}
    `);
    checkGenerateReady();
  } catch (err) {
    STATE.icDevice = null;
    resultEl.className = 'param-result result-error';
    setHTML(resultEl, `❌ ${err.message}`);
  }
  btn.disabled = false;
  btn.textContent = '解析IC';
}

// ── Port topology inference from detected pads ──
function inferPortTopology() {
  const pads = STATE.pads.front?.candidates || STATE.pads.front?.pads || [];
  const labels = pads.map(p => (p.label || '').toUpperCase());
  const hasC = labels.includes('C+') && labels.includes('C-');
  const hasP = labels.includes('P+') && labels.includes('P-');
  let inferred = null;
  if (hasP && hasC) inferred = 'separate';
  else if (hasP && !hasC) inferred = 'common';
  if (inferred) {
    STATE.inferredTopology = inferred;
    const el = $('#topology-inferred');
    show(el);
    setHTML(el, `🔍 根据焊盘丝印推断: <b>${inferred === 'common' ? '共口' : '分口'}</b>（${hasC ? '检测到C+/C-和P+/P-' : '仅检测到P+/P-'}），可手动覆盖`);
  }
  return inferred;
}

function getSelectedTopology() {
  const sel = document.querySelector('input[name="port-topology"]:checked')?.value || 'auto';
  if (sel !== 'auto') return sel;
  return STATE.inferredTopology || 'common';
}

function checkGenerateReady() {
  const recognitionOk = !!(STATE.cal.front && STATE.extract.front);
  const cellOk = !!STATE.cellParams;
  const icOk = !!STATE.icDevice;
  const ok = recognitionOk && cellOk && icOk;
  const btn = $('#btn-generate');
  btn.disabled = !ok;
  if (recognitionOk) {
    show($('#step-export'));
    inferPortTopology();
  }
  if (ok) {
    setBadge('step3', '可生成', 'ready');
  } else {
    const missing = [];
    if (!recognitionOk) missing.push('PCB识别');
    if (!cellOk) missing.push('电芯参数');
    if (!icOk) missing.push('IC解析');
    setBadge('step3', `缺少: ${missing.join(', ')}`, 'waiting');
  }
}

// ── Map detected pad candidates → DesignSpec terminals ──
const LABEL_ROLES = {
  'B+':  { roles: ['battery'], polarity: 'positive' },
  'B-':  { roles: ['battery'], polarity: 'negative' },
  'P+':  { roles: ['charge', 'discharge'], polarity: 'positive' },
  'P-':  { roles: ['charge', 'discharge'], polarity: 'negative' },
  'C+':  { roles: ['charge'], polarity: 'positive' },
  'C-':  { roles: ['charge'], polarity: 'negative' },
  'NTC': { roles: ['temperature'], polarity: null },
  'TH':  { roles: ['temperature'], polarity: null },
  'N':   { roles: ['temperature'], polarity: null },
  'ID':  { roles: ['identification'], polarity: null },
};

function buildTerminals() {
  // 合并正反面焊盘，side 直接使用检测结果来源面。
  const terminals = [];
  let idx = 0;
  for (const side of ['front', 'back']) {
    const padData = STATE.pads[side];
    const pads = padData?.candidates || [];
    // detect-terminals 返回 PCB 相对坐标（origin=焊盘裁剪框左上角），而轮廓是
    // 全幅框架坐标；加上裁剪偏移 crop_offset_mm 统一坐标系，否则端子会落在
    // 轮廓外导致 point_in_polygon 校验失败。
    const off = padData?.coordinate_system?.crop_offset_mm || { x: 0, y: 0 };
    for (const pad of pads) {
      const label = (pad.label || '').toUpperCase().trim();
      const mapping = LABEL_ROLES[label];
      if (!mapping) continue;
      const region = pad.visible_region || (pad.matched_regions || [])[0] || {};
      const center = region.center || {};
      const poly = region.polygon || [];
      if (!center.x_mm && center.x_mm !== 0) continue;
      // Estimate width/height from polygon bbox
      let w = 2.0, h = 2.0;
      if (poly.length >= 3) {
        const xs = poly.map(p => p.x_mm), ys = poly.map(p => p.y_mm);
        w = Math.max(Math.max(...xs) - Math.min(...xs), 0.5);
        h = Math.max(Math.max(...ys) - Math.min(...ys), 0.5);
      }
      // Build source_region with polygon for accurate pad shape rendering
      const sourceRegion = {
        type: 'solder_pad',
        visual_class: 'pad',
        shape: 'rect',
        center: { x_mm: center.x_mm + off.x, y_mm: center.y_mm + off.y },
        bbox: { x_mm: center.x_mm + off.x - w / 2, y_mm: center.y_mm + off.y - h / 2, width_mm: w, height_mm: h },
        polygon: poly.map(p => ({ x_mm: p.x_mm + off.x, y_mm: p.y_mm + off.y })),
        source: 'vlm',
      };
      terminals.push({
        id: `T${++idx}_${label.replace(/[^A-Z0-9]/g, '')}`,
        position: { x_mm: center.x_mm + off.x, y_mm: center.y_mm + off.y },
        roles: mapping.roles,
        polarity: mapping.polarity,
        side,
        shape: 'rect',
        width_mm: Math.min(w, 50),
        height_mm: Math.min(h, 50),
        source_region: sourceRegion,
      });
    }
  }
  return terminals;
}

function deriveBatteryType() {
  const cell = STATE.cellParams;
  if (!cell) return '18650';
  const chem = (cell.chemistry || '').toLowerCase();
  if (chem.includes('fepo4') || chem.includes('lfp')) return 'LFP';
  if (chem.includes('lipo') || chem.includes('polymer') || (cell.form_factor || '').includes('软包')) return 'LiPo';
  const ff = (cell.form_factor || '').toLowerCase();
  if (ff.includes('21700')) return '21700';
  return '18650';
}

async function generateProject() {
  const btn = $('#btn-generate');
  btn.disabled = true;
  btn.textContent = '生成中...';

  const fId = STATE.cal.front?.id;
  const bId = STATE.cal.back?.id;
  const ic = STATE.icDevice?.full_mpn || $('#ic-model').value || 'DW01';

  // Build outline from extraction result
  const outlinePts = STATE.extract.front?.outline || [];
  // Build terminals from detected pads
  const terminals = buildTerminals();
  const topology = getSelectedTopology();
  const count = parseInt($('#cell-count').value) || 1;
  const connection = $('#cell-connection').value || 'series';
  const batteryType = deriveBatteryType();
  const balance = document.querySelector('input[name="balance"]:checked')?.value === 'yes';
  const targetCurrent = parseFloat($('#target-current').value) || null;
  const mosModel = $('#mos-model').value.trim() || null;

  // ── 合并正反面元器件识别结果 ──
  const detectedComponents = [];
  for (const side of ['front', 'back']) {
    const cr = STATE.components[side];
    if (cr && Array.isArray(cr.components)) {
      for (const c of cr.components) {
        detectedComponents.push({
          type: c.type || 'other',
          silkscreen: c.silkscreen || '',
          package: c.package || '',
          confidence: c.confidence ?? 0.5,
        });
      }
    }
  }
  // 从识别结果推断 mos_count：统计 type=mosfet 的个数，至少为 1，否则默认 2
  const mosDetected = detectedComponents.filter(c => c.type === 'mosfet').length;
  const mosCount = mosDetected >= 1 ? mosDetected : 2;

  try {
    const spec = {
      name: `抄板_${ic}_${count}${connection === 'series' ? 'S' : 'P'}_${new Date().toISOString().slice(0, 10)}`,
      protection_ic: ic,
      battery: { count, connection, battery_type: batteryType },
      mos_count: mosCount,
      mos_mpn: mosModel,
      outline: { points: outlinePts, source: 'photo', confirmed: true },
      terminals,
      photo_capture: {
        front_calibration_id: fId || null,
        back_calibration_id: bId || null,
      },
      detected_components: detectedComponents,
    };

    const resp = await fetch('/api/projects', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(spec),
    });
    if (!resp.ok) {
      const err = await resp.json().catch(() => ({ detail: resp.statusText }));
      throw new Error(err.detail || err.error?.message || 'Generation failed');
    }
    const proj = await resp.json();
    const projId = proj.id || proj.project_id;

    let statusHtml = `<span class="ok">项目已创建: ${projId}</span>`;
    statusHtml += ` ｜ 端口拓扑: <b>${proj.port_topology === 'common' ? '共口' : proj.port_topology === 'separate' ? '分口' : proj.port_topology}</b>`;
    if (topology && proj.port_topology && topology !== 'auto' && proj.port_topology !== topology) {
      statusHtml += ` <span class="ng">（与选择不一致，请检查焊盘识别）</span>`;
    }
    if (proj.directory) {
      statusHtml += `<br/>项目目录: <code style="font-size:.8rem">${proj.directory}</code>`;
    }
    setHTML($('#gen-status'), statusHtml);
    show($('#gen-actions'));
    
    if (projId) {
      // 生成制造文件；器件为候选模板时先自动审批样板试产，再重试。
      let mResp = await fetch(`/api/projects/${projId}/manufacturing`, { method: 'POST' });
      if (!mResp.ok) {
        const mErr = await mResp.json().catch(() => ({}));
        if ((mErr.error?.code || '') === 'CANDIDATE_APPROVAL_REQUIRED') {
          await fetch(`/api/projects/${projId}/approve-candidate`, { method: 'POST' });
          mResp = await fetch(`/api/projects/${projId}/manufacturing`, { method: 'POST' });
        }
      }
      if (mResp.ok) {
        const mData = await mResp.json();
        const files = mData.manifest?.files || [];
        let dlHtml = '<h4 style="margin:0 0 6px 0">生成文件清单（共 ' + files.length + ' 个）</h4>';
        dlHtml += files.map(f =>
          `<a class="btn btn-outline" style="margin:3px" href="/api/projects/${projId}/artifacts/output/${f.path}" download>${f.path}</a>`
        ).join('');
        if (mData.package) {
          dlHtml += `<div style="margin-top:8px"><a class="btn btn-primary" href="/api/projects/${projId}/artifacts/${mData.package}" download>⬇ 下载完整包 (ZIP)</a></div>`;
        }
        setHTML($('#gen-downloads'), dlHtml);
      } else {
        const mErr = await mResp.json().catch(() => ({}));
        setHTML($('#gen-downloads'),
          `<span class="ng">制造文件未生成：${mErr.error?.message || mResp.statusText}</span>`);
      }
    }

    btn.textContent = '生成完成';
    btn.classList.add('btn-done');
    setBadge('step3', '已完成', 'ready');
  } catch (err) {
    setHTML($('#gen-status'), `<span class="ng">${err.message}</span>`);
    btn.textContent = '重试生成';
    btn.disabled = false;
  }
}

// ── Image Lightbox (click to zoom) ──
function initLightbox() {
  // Create overlay element
  const overlay = document.createElement('div');
  overlay.id = 'lightbox-overlay';
  overlay.innerHTML = '<img src="" alt="放大预览" />';
  document.body.appendChild(overlay);

  // Click on any .img-grid img → open lightbox
  document.addEventListener('click', e => {
    const img = e.target.closest('.img-grid img');
    if (img) {
      overlay.querySelector('img').src = img.src;
      overlay.classList.add('active');
      return;
    }
    // Click on .cal-canvas-wrap canvas → open lightbox (convert canvas to image)
    const canvas = e.target.closest('.cal-canvas-wrap canvas');
    if (canvas && canvas.width > 0) {
      overlay.querySelector('img').src = canvas.toDataURL('image/png');
      overlay.classList.add('active');
    }
  });

  // Click overlay → close
  overlay.addEventListener('click', () => {
    overlay.classList.remove('active');
  });

  // ESC → close
  document.addEventListener('keydown', e => {
    if (e.key === 'Escape') overlay.classList.remove('active');
  });
}

// ── Initialize ──
document.addEventListener('DOMContentLoaded', () => {
  init();
  initLightbox();
});
