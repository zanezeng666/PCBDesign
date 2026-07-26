/* ──────────────────────────────────────────────────
 *  Step-3 Standalone Test Page — App Logic
 *  FLOW:
 *    ⓪ Load existing recognition data (outline + pads) from a calibration
 *    ① Cell lookup (AI)  ② IC resolve  ③ Config  ④ Topology/Balance
 *    → Generate KiCad project
 *  No photo upload / recognition needed — data comes from saved calibrations.
 * ────────────────────────────────────────────────── */

const $ = (sel, ctx = document) => ctx.querySelector(sel);
const $$ = (sel, ctx = document) => Array.from(ctx.querySelectorAll(sel));

// ── Global state (mirrors the shape used by the main page) ──
const STATE = {
  // Recognition data loaded from a saved calibration
  cal: { front: null, back: null },      // { id }
  extract: { front: null, back: null },  // { outline: [...] }
  pads: { front: null, back: null },     // { candidates: [...] }
  // Step 3: design parameters
  cellParams: null,        // AI cell lookup result
  icDevice: null,          // resolved IC device package
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

// ══════════════════════════════════════════════════════════
//  ⓪ Load existing recognition data (front + back)
// ══════════════════════════════════════════════════════════

async function loadCalibrationList() {
  const selF = $('#cal-select-front');
  const selB = $('#cal-select-back');
  try {
    const resp = await fetch('/api/calibrations/with-recognition');
    const data = await resp.json();
    const items = data.calibrations || [];
    if (!items.length) {
      setHTML(selF, '<option value="">— 暂无可用识别记录 —</option>');
      setHTML(selB, '<option value="">— 暂无 —</option>');
      return;
    }
    const toOption = it => {
      const labels = (it.labels || []).filter(Boolean).join('/');
      const desc = `${it.created || ''} ｜ ${it.outline_points}轮廓点 ｜ ${it.candidate_count}焊盘${labels ? ' [' + labels + ']' : ''}`;
      return `<option value="${it.calibration_id}">${it.calibration_id.slice(0, 12)}… — ${desc}</option>`;
    };
    const fronts = items.filter(it => it.side === 'front');
    const backs = items.filter(it => it.side === 'back');
    setHTML(selF, (fronts.length ? fronts.map(toOption).join('') : '<option value="">— 无正面记录 —</option>'));
    setHTML(selB, '<option value="">— 不加载背面 —</option>' + (backs.length ? backs.map(toOption).join('') : ''));
  } catch (err) {
    setHTML(selF, '<option value="">— 列表加载失败 —</option>');
    setHTML($('#recog-status'), `<span class="ng">${err.message}</span>`);
  }
}

async function loadSide(side) {
  const sel = $(`#cal-select-${side}`);
  const calId = sel.value;
  const statusEl = $('#recog-status');
  if (!calId) {
    setHTML(statusEl, `<span class="ng">请先选择一条${side === 'front' ? '正面' : '背面'}校准记录</span>`);
    return;
  }
  setHTML(statusEl, `<span class="spinner"></span> 正在加载${side === 'front' ? '正面' : '背面'}识别数据...`);
  try {
    const resp = await fetch(`/api/calibrations/${calId}/recognition`);
    const data = await resp.json();
    if (!resp.ok) throw new Error(data.error?.message || data.detail || '加载失败');

    // Populate STATE in the same shape the main page uses
    STATE.cal[side] = { id: calId };
    STATE.pads[side] = { candidates: data.candidates || [] };
    if (side === 'front') {
      STATE.extract.front = { outline: data.outline || [] };
      // Preview image (rectified photo) — only for front
      $('#recog-img').src = `/api/calibrations/${calId}/rectified.png`;
    }
    refreshRecogInfo();
    setHTML(statusEl, `<span class="ok">${side === 'front' ? '正面' : '背面'}识别数据加载成功</span>`);
    checkGenerateReady();
  } catch (err) {
    setHTML(statusEl, `<span class="ng">加载失败: ${err.message}</span>`);
  }
}

function refreshRecogInfo() {
  const fCands = STATE.pads.front?.candidates || [];
  const bCands = STATE.pads.back?.candidates || [];
  const fLabels = fCands.map(c => c.label).filter(Boolean);
  const bLabels = bCands.map(c => c.label).filter(Boolean);
  const outlinePts = STATE.extract.front?.outline?.length || 0;
  const chips = ls => ls.length ? ls.map(l => `<span class="pad-chip">${l}</span>`).join('') : '<span class="ng">无</span>';
  setHTML($('#recog-info'), `
    <b>轮廓顶点:</b> ${outlinePts} 个<br/>
    <b>正面焊盘:</b> ${fCands.length} 个 ${chips(fLabels)}<br/>
    <b>背面焊盘:</b> ${bCands.length} 个 ${chips(bLabels)}<br/>
    <b>合计端子:</b> ${fCands.length + bCands.length} 个
  `);
  show($('#recog-preview'));
  const total = fCands.length + bCands.length;
  setBadge('recog', total ? `已加载 ${total} 焊盘` : '未加载', total ? 'ready' : 'waiting');
}

// ══════════════════════════════════════════════════════════
//  ① Cell lookup (AI)
// ══════════════════════════════════════════════════════════

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

// ══════════════════════════════════════════════════════════
//  ② IC resolve (marking → real MPN)
// ══════════════════════════════════════════════════════════

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

// ══════════════════════════════════════════════════════════
//  ④ Port topology inference from loaded pads
// ══════════════════════════════════════════════════════════

function inferPortTopology() {
  const pads = [...(STATE.pads.front?.candidates || []), ...(STATE.pads.back?.candidates || [])];
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
  if (recognitionOk) inferPortTopology();
  if (ok) {
    setBadge('step3', '可生成', 'ready');
  } else {
    const missing = [];
    if (!recognitionOk) missing.push('识别数据');
    if (!cellOk) missing.push('电芯参数');
    if (!icOk) missing.push('IC解析');
    setBadge('step3', `缺少: ${missing.join(', ')}`, 'waiting');
  }
}

// ══════════════════════════════════════════════════════════
//  Generate KiCad project
// ══════════════════════════════════════════════════════════

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
  const terminals = [];
  let idx = 0;
  for (const side of ['front', 'back']) {
    const pads = STATE.pads[side]?.candidates || [];
    for (const pad of pads) {
      const label = (pad.label || '').toUpperCase().trim();
      const mapping = LABEL_ROLES[label];
      if (!mapping) continue;
      const region = pad.visible_region || (pad.matched_regions || [])[0] || {};
      const center = region.center || {};
      const poly = region.polygon || [];
      if (!center.x_mm && center.x_mm !== 0) continue;
      let w = 2.0, h = 2.0;
      if (poly.length >= 3) {
        const xs = poly.map(p => p.x_mm), ys = poly.map(p => p.y_mm);
        w = Math.max(Math.max(...xs) - Math.min(...xs), 0.5);
        h = Math.max(Math.max(...ys) - Math.min(...ys), 0.5);
      }
      const sourceRegion = {
        type: 'solder_pad',
        visual_class: 'pad',
        shape: 'rect',
        center: { x_mm: center.x_mm, y_mm: center.y_mm },
        bbox: { x_mm: center.x_mm - w / 2, y_mm: center.y_mm - h / 2, width_mm: w, height_mm: h },
        polygon: poly.map(p => ({ x_mm: p.x_mm, y_mm: p.y_mm })),
        source: 'vlm',
      };
      terminals.push({
        id: `T${++idx}_${label.replace(/[^A-Z0-9]/g, '')}`,
        position: { x_mm: center.x_mm, y_mm: center.y_mm },
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

  const outlinePts = STATE.extract.front?.outline || [];
  const terminals = buildTerminals();
  const topology = getSelectedTopology();
  const count = parseInt($('#cell-count').value) || 1;
  const connection = $('#cell-connection').value || 'series';
  const batteryType = deriveBatteryType();

  try {
    const spec = {
      name: `抄板_${ic}_${count}${connection === 'series' ? 'S' : 'P'}_${new Date().toISOString().slice(0, 10)}`,
      protection_ic: ic,
      battery: { count, connection, battery_type: batteryType },
      mos_count: 2,
      mos_mpn: ($('#mos-model')?.value || '').trim() || null,
      outline: { points: outlinePts, source: 'photo', confirmed: true },
      terminals,
      photo_capture: {
        front_calibration_id: fId || null,
        back_calibration_id: bId || null,
      },
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
    setHTML($('#gen-status'), statusHtml);
    show($('#gen-actions'));

    if (projId) {
      const mResp = await fetch(`/api/projects/${projId}/manufacturing`, { method: 'POST' });
      if (mResp.ok) {
        setHTML($('#gen-downloads'), `
          <a class="btn btn-outline" href="/api/projects/${projId}/artifacts/pcb.kicad_sch" download>
            下载 KiCad 原理图
          </a>
          <a class="btn btn-outline" href="/api/projects/${projId}/artifacts/pcb.kicad_pcb" download>
            下载 KiCad PCB
          </a>
        `);
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

// ══════════════════════════════════════════════════════════
//  Initialize
// ══════════════════════════════════════════════════════════

async function init() {
  $('#btn-load-front').addEventListener('click', () => loadSide('front'));
  $('#btn-load-back').addEventListener('click', () => loadSide('back'));
  $('#btn-refresh-list').addEventListener('click', loadCalibrationList);
  $('#btn-cell-lookup').addEventListener('click', lookupCell);
  $('#btn-ic-resolve').addEventListener('click', resolveIc);
  $('#btn-generate').addEventListener('click', generateProject);
  await loadCalibrationList();
}

document.addEventListener('DOMContentLoaded', init);
