/* ═══════════════════════════════════════════════════════════════
 * Battery Designer — 前端逻辑（简化版三步流程）
 *
 * Step 1: 黑框尺寸 + 正反面图片 → recognize-pcb → 展示识别结果
 * Step 2: 焊盘识别（detect-terminals）
 * Step 3: 元器件识别（detect-components）
 * Step 4: 设计参数 & 生成 KiCad
 * ═══════════════════════════════════════════════════════════════ */

// ── 全局状态 ──
const state = {
  frontFile: null,
  backFile: null,
  // recognize-pcb 结果
  recognizeResult: null,
  // calibration_id (front/back)
  frontCalibrationId: null,
  backCalibrationId: null,
  // 前反面纠正后图片 base64
  frontRectifiedB64: "",
  backRectifiedB64: "",
  // 焊盘结果
  padsFront: null,
  padsBack: null,
  // 元器件结果
  componentsFront: null,
  componentsBack: null,
};

// ── DOM helpers ──
const $ = (id) => document.getElementById(id);
const show = (el) => el && (el.hidden = false);
const hide = (el) => el && (el.hidden = true);
const setBadge = (badgeId, text, ok) => {
  const el = $(badgeId);
  if (!el) return;
  el.textContent = text;
  el.className = "step-badge " + (ok ? "ready" : "waiting");
};

/* ═══════════════════════════════════════════════════════════════
 * Step 1: PCB 轮廓识别
 * ═══════════════════════════════════════════════════════════════ */

// ── 文件上传：拖放 & 点击 ──
function setupDropZone(zoneId, dropId, fileId, statusId, fileKey) {
  const drop = $(dropId);
  const fileInput = $(fileId);
  const status = $(statusId);

  drop.addEventListener("click", () => fileInput.click());
  drop.addEventListener("dragover", (e) => {
    e.preventDefault();
    drop.classList.add("drag-over");
  });
  drop.addEventListener("dragleave", () => drop.classList.remove("drag-over"));
  drop.addEventListener("drop", (e) => {
    e.preventDefault();
    drop.classList.remove("drag-over");
    if (e.dataTransfer.files.length) handleFile(e.dataTransfer.files[0], status, fileKey);
  });
  fileInput.addEventListener("change", () => {
    if (fileInput.files.length) handleFile(fileInput.files[0], status, fileKey);
  });
}

function handleFile(file, statusEl, fileKey) {
  if (!file.type.startsWith("image/")) {
    statusEl.innerHTML = '<span class="ng">请上传图片文件</span>';
    return;
  }
  state[fileKey] = file;
  statusEl.innerHTML =
    `✓ ${file.name} <span style="color:#94a3b8">(${(file.size / 1024).toFixed(0)} KB)</span>`;

  // 显示图片预览缩略图
  const previewId = fileKey === "frontFile" ? "preview-front" : "preview-back";
  const previewImg = $(previewId);
  const dropZone = previewImg.closest(".zone-drop");
  if (previewImg) {
    previewImg.src = URL.createObjectURL(file);
    dropZone.classList.add("has-preview");
  }
  checkRecognizeReady();
}

function checkRecognizeReady() {
  const ready = state.frontFile && state.backFile;
  $("btn-recognize").disabled = !ready;
  setBadge("badge-step1", ready ? "可以识别" : "等待输入", false);
}

// ── 调用 recognize-pcb ──
// 三阶段（正面→背面→交叉验证），每阶段包含若干子步骤
const RECOGNIZE_PHASES = [
  {
    name: "正面识别",
    steps: ["方向检测", "黑框检测", "透视校正", "轮廓提取", "纸色模型", "透明PNG"],
    estSec: 10,
  },
  {
    name: "背面识别",
    steps: ["方向检测", "黑框检测", "透视校正", "轮廓提取", "纸色模型", "透明PNG"],
    estSec: 10,
  },
  {
    name: "交叉验证",
    steps: ["轮廓匹配", "生成共识图"],
    estSec: 4,
  },
];

function buildProgressHtml() {
  let html = '<div class="recognize-progress">';
  html += '<div class="recog-timer"><span class="spinner"></span> 已用时 <strong id="recog-elapsed">0</strong> 秒</div>';
  html += '<div class="recog-bar-track"><div class="recog-bar-fill" id="recog-bar"></div></div>';
  html += '<div class="recog-phases">';
  RECOGNIZE_PHASES.forEach((phase, pi) => {
    html += `<div class="recog-phase" data-phase="${pi}">`;
    html += `  <div class="recog-phase-head"><span class="recog-phase-icon"></span>${phase.name}</div>`;
    html += `  <div class="recog-phase-steps">`;
    phase.steps.forEach((s, si) => {
      html += `<span class="step-chip" data-phase="${pi}" data-step="${si}">○ ${s}</span>`;
    });
    html += `  </div>`;
    html += `</div>`;
  });
  html += '</div>';
  html += '<div class="recog-overtime" id="recog-overtime">⏳ 处理时间超出预期，请耐心等待...</div>';
  html += '</div>';
  return html;
}

function updateProgressUI(elapsed) {
  const totalEst = RECOGNIZE_PHASES.reduce((s, p) => s + p.estSec, 0);

  // 计算当前阶段和阶段内进度
  let cumSec = 0;
  let curPhase = 0;
  let phaseFrac = 0;
  for (let pi = 0; pi < RECOGNIZE_PHASES.length; pi++) {
    if (elapsed < cumSec + RECOGNIZE_PHASES[pi].estSec) {
      curPhase = pi;
      phaseFrac = (elapsed - cumSec) / RECOGNIZE_PHASES[pi].estSec;
      break;
    }
    cumSec += RECOGNIZE_PHASES[pi].estSec;
    curPhase = pi;
    phaseFrac = 1;
  }

  // 进度条：线性推进，封顶 95%
  const pct = Math.min(95, (elapsed / totalEst) * 95);
  const bar = $("recog-bar");
  if (bar) bar.style.width = pct.toFixed(0) + "%";

  // 更新各阶段状态
  RECOGNIZE_PHASES.forEach((phase, pi) => {
    const phaseEl = document.querySelector(`.recog-phase[data-phase="${pi}"]`);
    if (!phaseEl) return;
    phaseEl.classList.remove("done", "active");
    const chips = phaseEl.querySelectorAll(".step-chip");

    if (pi < curPhase) {
      // 该阶段已完成
      phaseEl.classList.add("done");
      chips.forEach((c) => {
        c.className = "step-chip done";
        c.textContent = c.textContent.replace("○", "✓");
      });
    } else if (pi === curPhase) {
      // 当前正在处理的阶段
      phaseEl.classList.add("active");
      const stepIdx = Math.min(phase.steps.length - 1, Math.floor(phaseFrac * phase.steps.length));
      chips.forEach((c, si) => {
        const label = phase.steps[si];
        if (si < stepIdx) {
          c.className = "step-chip done";
          c.textContent = "✓ " + label;
        } else if (si === stepIdx) {
          c.className = "step-chip active";
          c.textContent = "🔄 " + label;
        } else {
          c.className = "step-chip";
          c.textContent = "○ " + label;
        }
      });
    }
  });

  // 超时提示
  const ot = $("recog-overtime");
  if (ot) ot.style.display = elapsed >= totalEst ? "block" : "none";
}

async function recognizePCB() {
  const btn = $("btn-recognize");
  const info = $("pcb-info");
  btn.disabled = true;
  btn.innerHTML = '<span class="spinner"></span> 识别中...';

  // 构建进度 UI
  info.innerHTML = buildProgressHtml();

  // 计时器 + 线性进度更新（不循环）
  const startTime = Date.now();
  const progressTimer = setInterval(() => {
    const elapsed = (Date.now() - startTime) / 1000;
    const elEl = $("recog-elapsed");
    if (elEl) elEl.textContent = elapsed.toFixed(0);
    updateProgressUI(elapsed);
  }, 500);

  const frameW = parseFloat($("frame-w").value) || 60;
  const frameH = parseFloat($("frame-h").value) || 30;

  const fd = new FormData();
  fd.append("front_image", state.frontFile, "front.jpg");
  fd.append("back_image", state.backFile, "back.jpg");
  fd.append("frame_w_mm", frameW);
  fd.append("frame_h_mm", frameH);

  try {
    const resp = await fetch("/api/vision/recognize-pcb", { method: "POST", body: fd });
    const data = await resp.json();
    clearInterval(progressTimer);

    // 完成 → 进度条满格 + 全部标记完成
    const bar = $("recog-bar");
    if (bar) bar.style.width = "100%";
    document.querySelectorAll(".recog-phase").forEach((el) => {
      el.classList.remove("active");
      el.classList.add("done");
    });
    document.querySelectorAll(".step-chip").forEach((c) => {
      c.className = "step-chip done";
      c.textContent = "✓ " + c.textContent.replace(/^[○🔄✓]\s*/, "");
    });
    const ot = $("recog-overtime");
    if (ot) ot.style.display = "none";

    if (!resp.ok || !data.success) {
      throw new Error(data.detail || data.error || "识别失败");
    }

    state.recognizeResult = data;
    state.frontCalibrationId = data.front?.calibration_id || "";
    state.backCalibrationId = data.back?.calibration_id || "";
    state.frontRectifiedB64 = data.front?.rectified_png_b64 || "";
    state.backRectifiedB64 = data.back?.rectified_png_b64 || "";

    displayRecognizeResults(data);
    info.innerHTML = `✓ PCB识别成功 — 尺寸: <strong>${data.pcb_width_mm}mm × ${data.pcb_height_mm}mm</strong>，轮廓顶点: ${data.outline_vertex_count}`;

    // 标记步骤1完成
    btn.classList.add("btn-done");
    btn.innerHTML = "✓ 识别完成";
    setBadge("badge-step1", "已完成", true);

    // 显示步骤2
    show($("step-pads"));
    $("btn-detect-pads").disabled = false;

  } catch (err) {
    clearInterval(progressTimer);
    info.innerHTML = `<span class="ng">✗ ${err.message}</span>`;
    btn.disabled = false;
    btn.innerHTML = "识别PCB轮廓";
  }
}

function displayRecognizeResults(data) {
  // 正面
  const frontHtml = buildSideResultHtml(data.front, "front");
  $("recognize-result-front").innerHTML = frontHtml;
  // 背面
  const backHtml = buildSideResultHtml(data.back, "back");
  $("recognize-result-back").innerHTML = backHtml;
  show($("recognize-results"));

  // 一致性校验
  if (data.consensus) {
    const c = data.consensus;
    const consensusImg = c.transparent_pcb_b64
      ? `<div class="img-grid" style="margin-top:8px;"><img class="zoomable-img" src="data:image/png;base64,${c.transparent_pcb_b64}" onclick="openLightbox(this.src)" title="点击放大" style="max-width:100%; border-radius:6px; border:1px solid #cbd5e1;" /></div>`
      : "";
    $("consensus-content").innerHTML = `
      <p>${c.message}</p>
      <div style="display:flex; gap:24px; margin-top:8px; font-size:0.82rem; color:#475569;">
        <span>正面面积: ${c.front_area_mm2} mm²</span>
        <span>背面面积: ${c.back_area_mm2} mm²</span>
        <span>共识面积: ${c.consensus_area_mm2} mm²</span>
        <span>偏差: ${c.deviation_pct}%</span>
        <span class="${c.ok ? "ok" : "ng"}">${c.ok ? "✓ 匹配" : "⚠ 有偏差"}</span>
      </div>
      ${consensusImg}
    `;
    show($("consensus-section"));
  }
}

function buildSideResultHtml(sideData, sideName) {
  if (!sideData || !sideData.calibration_success) {
    return `<p class="ng">识别失败: ${sideData?.error || "未知错误"}</p>`;
  }
  const overlay = sideData.overlay_b64
    ? `<div class="img-grid"><img class="zoomable-img" src="data:image/png;base64,${sideData.overlay_b64}" onclick="openLightbox(this.src)" title="点击放大" style="max-width:100%; border-radius:6px; border:1px solid #cbd5e1;" /></div>`
    : "";
  const transparent = sideData.transparent_pcb_b64
    ? `<div class="img-grid" style="margin-top:8px;"><img class="zoomable-img" src="data:image/png;base64,${sideData.transparent_pcb_b64}" onclick="openLightbox(this.src)" title="点击放大" style="max-width:100%; border-radius:6px; border:1px solid #cbd5e1;" /></div>`
    : "";
  return `
    <div class="result-info">
      <span>像素精度: ${sideData.pixels_per_mm} px/mm</span> |
      <span>轮廓顶点: ${sideData.outline?.length || 0}</span>
    </div>
    ${overlay}
    ${transparent}
  `;
}

/* ═══════════════════════════════════════════════════════════════
 * Step 2: 焊盘识别
 * ═══════════════════════════════════════════════════════════════ */

async function detectPads() {
  const btn = $("btn-detect-pads");
  btn.disabled = true;
  btn.innerHTML = '<span class="spinner"></span> 识别焊盘中...';

  try {
    const [frontResult, backResult] = await Promise.all([
      detectTerminalsOneSide(state.frontCalibrationId, "front"),
      detectTerminalsOneSide(state.backCalibrationId, "back"),
    ]);

    state.padsFront = frontResult;
    state.padsBack = backResult;

    // 渲染
    drawPadsOnCanvas($("pads-canvas-front"), state.frontRectifiedB64, frontResult);
    $("pads-info-front").innerHTML = formatPadsInfo(frontResult);
    drawPadsOnCanvas($("pads-canvas-back"), state.backRectifiedB64, backResult);
    $("pads-info-back").innerHTML = formatPadsInfo(backResult);
    show($("pads-results"));

    btn.classList.add("btn-done");
    btn.innerHTML = "✓ 焊盘识别完成";
    setBadge("badge-step2", "已完成", true);

    // 显示步骤3
    show($("step-components"));
    $("btn-detect-components").disabled = false;

  } catch (err) {
    btn.disabled = false;
    btn.innerHTML = "识别焊盘（正反面）";
    alert("焊盘识别失败: " + err.message);
  }
}

async function detectTerminalsOneSide(calibrationId, side) {
  const fd = new FormData();
  fd.append("calibration_id", calibrationId);
  fd.append("side", side);

  const resp = await fetch("/api/vision/detect-terminals", { method: "POST", body: fd });
  const data = await resp.json();
  if (!resp.ok) throw new Error(data.detail || data.error || `焊盘检测失败 (${side})`);
  return data;
}

function drawPadsOnCanvas(canvas, rectifiedB64, padResult) {
  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  if (!rectifiedB64) return;

  const img = new Image();
  img.onload = () => {
    const scale = Math.min(canvas.width / img.width, canvas.height / img.height);
    const dw = img.width * scale;
    const dh = img.height * scale;
    const dx = (canvas.width - dw) / 2;
    const dy = (canvas.height - dh) / 2;
    ctx.drawImage(img, dx, dy, dw, dh);

    // 绘制焊盘
    const terminals = padResult.terminals || padResult.pads || [];
    const ppm = padResult.pixels_per_mm || state.recognizeResult?.front?.pixels_per_mm || 10;
    const frameWmm = parseFloat($("frame-w").value) || 60;
    const frameHmm = parseFloat($("frame-h").value) || 30;

    terminals.forEach((t, i) => {
      const cx = dx + (t.x_mm / frameWmm) * dw;
      const cy = dy + (t.y_mm / frameHmm) * dh;
      const r = Math.max(4, (t.r_mm || 0.5) * scale * ppm * 0.1 + 4);

      ctx.beginPath();
      ctx.arc(cx, cy, r, 0, Math.PI * 2);
      ctx.fillStyle = "rgba(255, 200, 0, 0.4)";
      ctx.fill();
      ctx.strokeStyle = "#f59e0b";
      ctx.lineWidth = 1.5;
      ctx.stroke();

      // 标签
      ctx.fillStyle = "#dc2626";
      ctx.font = "bold 11px sans-serif";
      ctx.textAlign = "center";
      ctx.fillText(t.label || t.name || `P${i + 1}`, cx, cy - r - 3);
    });
  };
  img.src = "data:image/png;base64," + rectifiedB64;
}

function formatPadsInfo(result) {
  const terminals = result.terminals || result.pads || [];
  if (!terminals.length) return "未检测到焊盘";
  const labels = terminals.map((t) => t.label || t.name || "?").join(", ");
  return `检测到 ${terminals.length} 个焊盘: ${labels}`;
}

/* ═══════════════════════════════════════════════════════════════
 * Step 3: 元器件识别
 * ═══════════════════════════════════════════════════════════════ */

async function detectComponents() {
  const btn = $("btn-detect-components");
  btn.disabled = true;
  btn.innerHTML = '<span class="spinner"></span> 识别元器件中...';

  try {
    const [frontResult, backResult] = await Promise.all([
      detectComponentsOneSide(state.frontCalibrationId, "front"),
      detectComponentsOneSide(state.backCalibrationId, "back"),
    ]);

    state.componentsFront = frontResult;
    state.componentsBack = backResult;

    renderComponentsTable(frontResult, backResult);
    show($("components-results"));

    btn.classList.add("btn-done");
    btn.innerHTML = "✓ 元器件识别完成";
    setBadge("badge-step3", "已完成", true);

    // 显示步骤4
    show($("step-export"));
    $("btn-generate").disabled = false;
    setBadge("badge-step4", "请填写参数", false);

  } catch (err) {
    btn.disabled = false;
    btn.innerHTML = "识别元器件（正反面）";
    alert("元器件识别失败: " + err.message);
  }
}

async function detectComponentsOneSide(calibrationId, side) {
  const fd = new FormData();
  fd.append("calibration_id", calibrationId);
  fd.append("side", side);

  const resp = await fetch("/api/vision/detect-components", { method: "POST", body: fd });
  const data = await resp.json();
  if (!resp.ok) throw new Error(data.detail || data.error || `元器件检测失败 (${side})`);
  return data;
}

function renderComponentsTable(frontData, backData) {
  const frontComps = frontData.components || [];
  const backComps = backData.components || [];
  const all = [
    ...frontComps.map((c) => ({ ...c, side: "正面" })),
    ...backComps.map((c) => ({ ...c, side: "背面" })),
  ];

  if (!all.length) {
    $("components-table").innerHTML = '<p style="color:#94a3b8;">未检测到元器件</p>';
    return;
  }

  let html = `<table class="results-table"><thead><tr>
    <th>面</th><th>编号</th><th>型号/丝印</th><th>封装</th>
    <th>X (mm)</th><th>Y (mm)</th><th>旋转°</th><th>置信度</th>
  </tr></thead><tbody>`;
  all.forEach((c) => {
    html += `<tr>
      <td>${c.side}</td>
      <td>${c.designator || c.ref || "-"}</td>
      <td>${c.value || c.model || c.silkscreen || "-"}</td>
      <td>${c.footprint || c.package || "-"}</td>
      <td>${(c.x_mm ?? 0).toFixed(2)}</td>
      <td>${(c.y_mm ?? 0).toFixed(2)}</td>
      <td>${c.rotation ?? 0}</td>
      <td>${c.confidence ? (c.confidence * 100).toFixed(0) + "%" : "-"}</td>
    </tr>`;
  });
  html += "</tbody></table>";
  $("components-table").innerHTML = html;
}

/* ═══════════════════════════════════════════════════════════════
 * Step 4: 设计参数 & 生成
 * ═══════════════════════════════════════════════════════════════ */

// ── AI 电芯查询 ──
async function lookupCell() {
  const mfr = $("cell-manufacturer").value.trim();
  const model = $("cell-model").value.trim();
  const resultEl = $("cell-result");

  if (!mfr || !model) {
    show(resultEl);
    resultEl.className = "param-result result-error";
    resultEl.textContent = "请填写厂商和型号";
    return;
  }

  resultEl.className = "param-result";
  resultEl.innerHTML = '<span class="spinner"></span> 查询中...';
  show(resultEl);

  try {
    const resp = await fetch("/api/ai/cell-lookup", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ manufacturer: mfr, model }),
    });
    const data = await resp.json();
    if (!resp.ok) throw new Error(data.detail || "查询失败");

    resultEl.innerHTML = formatCellParams(data);
    checkGenerateReady();
  } catch (err) {
    resultEl.className = "param-result result-error";
    resultEl.textContent = "查询失败: " + err.message;
  }
}

function formatCellParams(data) {
  const items = [
    ["标称容量", data.capacity_mAh ? data.capacity_mAh + " mAh" : null],
    ["标称电压", data.nominal_voltage_v ? data.nominal_voltage_v + " V" : null],
    ["充电截止", data.charge_cutoff_v ? data.charge_cutoff_v + " V" : null],
    ["放电截止", data.discharge_cutoff_v ? data.discharge_cutoff_v + " V" : null],
    ["最大放电", data.max_discharge_a ? data.max_discharge_a + " A" : null],
    ["内阻", data.internal_resistance_mohm ? data.internal_resistance_mohm + " mΩ" : null],
    ["化学体系", data.chemistry || null],
  ].filter(([, v]) => v);

  if (!items.length) return '<span class="ng">未找到匹配参数</span>';
  return "✓ " + items.map(([k, v]) => `${k}: ${v}`).join(" | ");
}

// ── IC 解析 ──
async function resolveIC() {
  const icModel = $("ic-model").value.trim();
  const resultEl = $("ic-result");

  if (!icModel) {
    show(resultEl);
    resultEl.className = "param-result result-error";
    resultEl.textContent = "请输入IC型号";
    return;
  }

  resultEl.className = "param-result";
  resultEl.innerHTML = '<span class="spinner"></span> 解析中...';
  show(resultEl);

  try {
    const resp = await fetch("/api/ai/resolve-ic", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ ic_model: icModel, cell_count: parseInt($("cell-count").value) }),
    });
    const data = await resp.json();
    if (!resp.ok) throw new Error(data.detail || "解析失败");

    resultEl.innerHTML = formatICResult(data);
    checkGenerateReady();
  } catch (err) {
    resultEl.className = "param-result result-error";
    resultEl.textContent = "解析失败: " + err.message;
  }
}

function formatICResult(data) {
  const parts = [];
  if (data.resolved_model) parts.push(`型号: ${data.resolved_model}`);
  if (data.manufacturer) parts.push(`厂商: ${data.manufacturer}`);
  if (data.ovp) parts.push(`过压保护: ${data.ovp}V`);
  if (data.uvp) parts.push(`欠压保护: ${data.uvp}V`);
  if (data.mos_config) parts.push(`MOS: ${data.mos_config}`);
  return parts.length ? "✓ " + parts.join(" | ") : '<span class="ng">未能解析</span>';
}

function checkGenerateReady() {
  // 至少填写了电芯或IC就可以生成（用户也可以跳过）
  $("btn-generate").disabled = false;
}

// ── 生成 KiCad ──
async function generateDesign() {
  const btn = $("btn-generate");
  const statusEl = $("gen-status");
  btn.disabled = true;
  statusEl.innerHTML = '<span class="spinner"></span> 正在生成 KiCad 工程...';

  const r = state.recognizeResult;
  const fd = new FormData();
  fd.append("front_calibration_id", state.frontCalibrationId);
  fd.append("back_calibration_id", state.backCalibrationId);
  fd.append("outline_json", JSON.stringify(r.pcb_outline || []));
  fd.append("pcb_width_mm", r.pcb_width_mm);
  fd.append("pcb_height_mm", r.pcb_height_mm);
  fd.append("frame_w_mm", r.frame_w_mm);
  fd.append("frame_h_mm", r.frame_h_mm);

  // 设计参数
  const cellMfr = $("cell-manufacturer").value.trim();
  const cellModel = $("cell-model").value.trim();
  const icModel = $("ic-model").value.trim();
  const cellCount = $("cell-count").value;
  const cellConn = $("cell-connection").value;
  const mosModel = $("mos-model").value.trim();
  const targetCurrent = $("target-current").value;
  const portTopology = document.querySelector('input[name="port-topology"]:checked').value;
  const balance = document.querySelector('input[name="balance"]:checked').value;

  fd.append("cell_manufacturer", cellMfr);
  fd.append("cell_model", cellModel);
  fd.append("ic_model", icModel);
  fd.append("cell_count", cellCount);
  fd.append("cell_connection", cellConn);
  fd.append("mos_model", mosModel);
  fd.append("target_current", targetCurrent);
  fd.append("port_topology", portTopology);
  fd.append("balance", balance);

  // 元器件 & 焊盘
  fd.append("components_front_json", JSON.stringify(state.componentsFront?.components || []));
  fd.append("components_back_json", JSON.stringify(state.componentsBack?.components || []));
  fd.append("terminals_front_json", JSON.stringify(state.padsFront?.terminals || []));
  fd.append("terminals_back_json", JSON.stringify(state.padsBack?.terminals || []));

  try {
    const resp = await fetch("/api/generate", { method: "POST", body: fd });
    const data = await resp.json();
    if (!resp.ok) throw new Error(data.detail || "生成失败");

    statusEl.innerHTML = "✓ 生成成功！";
    renderDownloads(data);
    setBadge("badge-step4", "已完成", true);
  } catch (err) {
    statusEl.innerHTML = `<span class="ng">✗ ${err.message}</span>`;
    btn.disabled = false;
  }
}

function renderDownloads(data) {
  const el = $("gen-downloads");
  const files = data.files || [];
  if (!files.length) {
    el.innerHTML = '<p style="color:#94a3b8;">无文件生成</p>';
  } else {
    let html = "";
    files.forEach((f) => {
      html += `<a href="${f.url}" class="btn btn-secondary btn-sm" download>${f.name}</a> `;
    });
    el.innerHTML = html;
  }
  show($("gen-actions"));
}

/* ═══════════════════════════════════════════════════════════════
 * Lightbox (图片点击放大)
 * ═══════════════════════════════════════════════════════════════ */

function openLightbox(src) {
  const overlay = $("lightbox-overlay");
  const img = $("lightbox-img");
  if (!overlay || !img) return;
  img.src = src;
  overlay.classList.add("active");
}

function closeLightbox() {
  const overlay = $("lightbox-overlay");
  if (!overlay) return;
  overlay.classList.remove("active");
  $("lightbox-img").src = "";
}

document.addEventListener("keydown", (e) => {
  if (e.key === "Escape") closeLightbox();
});

/* ═══════════════════════════════════════════════════════════════
 * 初始化
 * ═══════════════════════════════════════════════════════════════ */

document.addEventListener("DOMContentLoaded", () => {
  // 上传区域
  setupDropZone("zone-front", "drop-front", "file-front", "status-front", "frontFile");
  setupDropZone("zone-back", "drop-back", "file-back", "status-back", "backFile");

  // 按钮
  $("btn-recognize").addEventListener("click", recognizePCB);
  $("btn-detect-pads").addEventListener("click", detectPads);
  $("btn-detect-components").addEventListener("click", detectComponents);
  $("btn-cell-lookup").addEventListener("click", lookupCell);
  $("btn-ic-resolve").addEventListener("click", resolveIC);
  $("btn-generate").addEventListener("click", generateDesign);

  // 尺寸变化时更新状态
  $("frame-w").addEventListener("input", checkRecognizeReady);
  $("frame-h").addEventListener("input", checkRecognizeReady);
});
