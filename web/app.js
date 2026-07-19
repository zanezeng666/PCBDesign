const state = {outline: [], terminals: [], candidates: [], calibrations: {front: null, back: null}, images: {front: null, back: null}, annotatedImages: {front: null, back: null}, widthMm: 0, heightMm: 0, alignmentError: null};
const $ = id => document.getElementById(id);
const canvas = $('boardCanvas');
const ctx = canvas.getContext('2d');

$('calibrateFront').onclick = () => calibrateSide('front');
$('calibrateBack').onclick = () => calibrateSide('back');
$('activeSide').onchange = draw;
$('showAnnotation').onchange = draw;
$('backTransform').onchange = () => { updateAlignment(); draw(); };
$('detectTerminals').onclick = detectCurrentSide;
$('acceptAllCandidates').onclick = () => {
  const side = $('activeSide').value;
  state.candidates.filter(candidate => candidate.side === side && candidate.region_resolved).forEach(acceptCandidate);
  state.candidates = state.candidates.filter(candidate => candidate.side !== side || !candidate.region_resolved);
  draw(); list(); listCandidates();
};
$('role').onchange = () => {
  if (['temperature', 'identification', 'auxiliary'].includes($('role').value)) $('polarity').value = '';
};

async function calibrateSide(side) {
  const file = $(side === 'front' ? 'frontPhoto' : 'backPhoto').files[0];
  if (!file) return show(`请选择${side === 'front' ? '正面' : '反面'}照片`);
  const body = new FormData();
  body.append('file', file);
  const knownSize = $('calibrationMode').value === 'known_size';
  let endpoint = '/api/vision/calibrate';
  if (knownSize) {
    endpoint = '/api/vision/calibrate-known-size';
    body.append('width_mm', $('boardWidth').value);
    body.append('height_mm', $('boardHeight').value);
  } else body.append('marker_size_mm', $('markerSize').value);
  show(`正在标定${side === 'front' ? '正面' : '反面'}…`);
  try {
    const result = await api(endpoint, {method: 'POST', body});
    state.calibrations[side] = result;
    state.candidates = state.candidates.filter(candidate => candidate.side !== side);
    if (side === 'front') { state.outline = result.outline; state.widthMm = result.width_mm; state.heightMm = result.height_mm; }
    const image = new Image();
    image.onload = () => { state.images[side] = image; if ($('activeSide').value === side) draw(); updateAlignment(); };
    image.src = 'data:image/png;base64,' + result.rectified_png_base64;
    $('activeSide').value = side;
    const perspective = result.perspective_method === 'detected_board_corners' ? '四角透视已校正' : result.perspective_method ? '使用旋转矩形回退' : '';
    const refinement = result.refinement_applied ? '；二次裁剪已生效' : '；二次裁剪未生效（可能已对齐）';
    const warning = result.method === 'known_size_auto' ? '；自动板框必须人工确认' : '';
    show(`${side === 'front' ? '正面' : '反面'}标定完成：${result.width_mm} × ${result.height_mm} mm，置信度 ${(result.confidence * 100).toFixed(1)}%${perspective ? `，${perspective}` : ''}${refinement}${warning}`);
  } catch (error) { show(error.message); }
}

async function detectCurrentSide() {
  const side = $('activeSide').value, calibration = state.calibrations[side];
  if (!calibration) return show(`请先标定${side === 'front' ? '正面' : '反面'}照片`);
  const body = new FormData();
  body.append('calibration_id', calibration.calibration_id);
  body.append('side', side);
  show(`正在识别${side === 'front' ? '正面' : '反面'}丝印与焊盘…`);
  try {
    const result = await api('/api/vision/detect-terminals', {method: 'POST', body});
    state.candidates = state.candidates.filter(candidate => candidate.side !== side).concat(result.candidates);
    if (result.annotated_png_base64) {
      const annotated = new Image();
      annotated.onload = () => { state.annotatedImages[side] = annotated; draw(); };
      annotated.src = 'data:image/png;base64,' + result.annotated_png_base64;
    }
    draw(); listCandidates();
    show(`识别完成：找到 ${result.candidate_count} 个待确认候选（qwen3.7-plus VLM 识别）`);
  } catch (error) { show(error.message); }
}

function acceptCandidate(candidate) {
  if (!candidate.region_resolved || !candidate.visible_region) return show(`${candidate.label} 尚未匹配到焊盘区域或孔位，不能采纳`);
  const baseId = candidate.id || 'PAD';
  let id = baseId, suffix = 2;
  while (state.terminals.some(terminal => terminal.id === id)) id = `${baseId}_${suffix++}`;
  state.terminals.push({id, position: visibleToCanonical(candidate.visible_region.center, candidate.side), roles: candidate.roles, polarity: candidate.polarity, side: candidate.side, shape: candidate.shape, width_mm: candidate.width_mm, height_mm: candidate.height_mm, source_region: candidate.visible_region});
}

function listCandidates() {
  $('candidateList').innerHTML = state.candidates.map((candidate, index) => {
    const region = candidate.visible_region;
    const multiRegions = candidate.matched_regions || [];
    const multiHint = multiRegions.length > 1 ? `（双排 ${multiRegions.length} 个焊盘）` : '';
    const areaText = region ? `${region.type === 'hole' ? '孔位' : '焊盘'}区域 ${region.bbox.width_mm} × ${region.bbox.height_mm} mm${multiHint}，距离文字 ${candidate.match_distance_mm} mm` : '未匹配到银白色区域或孔位';
    const qualityTag = candidate.match_quality === 'manual_review' ? ' <span style="color:#e00">[需确认]</span>' : '';
    return `<li><span class="candidate-label">${candidate.label}</span> ${candidate.side === 'front' ? '正面' : '反面'} · ${Math.round(candidate.confidence * 100)}%${qualityTag}<br><span class="candidate-detail">${areaText}</span> <button class="tiny accept" data-accept="${index}" ${region ? '' : 'disabled'}>确认区域并采纳</button><button class="tiny" data-dismiss="${index}">忽略</button></li>`;
  }).join('');
  document.querySelectorAll('[data-accept]').forEach(button => button.onclick = () => { const index = +button.dataset.accept; acceptCandidate(state.candidates[index]); state.candidates.splice(index, 1); draw(); list(); listCandidates(); });
  document.querySelectorAll('[data-dismiss]').forEach(button => button.onclick = () => { state.candidates.splice(+button.dataset.dismiss, 1); draw(); listCandidates(); });
}

canvas.onclick = event => {
  const side = $('activeSide').value;
  if (!state.images[side] || $('editOutline').checked) return;
  const size = +$('padSize').value;
  const polarity = $('polarity').value || null;
  const position = visibleToCanonical(eventPoint(event), side);
  state.terminals.push({id: 'T' + (state.terminals.length + 1), position, roles: [$('role').value], polarity, side, shape: 'circle', width_mm: size, height_mm: size});
  draw(); list();
};

$('mergeRole').onclick = () => {
  const side = $('activeSide').value, role = $('role').value, polarity = $('polarity').value || null;
  const candidates = state.terminals.filter(t => t.side === side && t.polarity === polarity);
  if (!candidates.length) return show('当前板面没有相同极性的触点可合并');
  const terminal = candidates[candidates.length - 1];
  if (!terminal.roles.includes(role)) terminal.roles.push(role);
  draw(); list();
};

function eventPoint(event) {
  const rect = canvas.getBoundingClientRect(), calibration = state.calibrations[$('activeSide').value];
  return {x_mm: ((event.clientX - rect.left) / rect.width) * calibration.width_mm, y_mm: ((event.clientY - rect.top) / rect.height) * calibration.height_mm};
}
function roundPoint(point) { return {x_mm: +point.x_mm.toFixed(3), y_mm: +point.y_mm.toFixed(3)}; }
function visibleToCanonical(point, side) {
  if (side === 'front') return roundPoint(point);
  const calibration = state.calibrations.back;
  let x = point.x_mm * state.widthMm / calibration.width_mm, y = point.y_mm * state.heightMm / calibration.height_mm, transform = $('backTransform').value;
  if (transform === 'mirror_x' || transform === 'rotate_180') x = state.widthMm - x;
  if (transform === 'mirror_y' || transform === 'rotate_180') y = state.heightMm - y;
  return roundPoint({x_mm: x, y_mm: y});
}
function canonicalToVisible(point, side) {
  if (side === 'front') return point;
  let x = point.x_mm, y = point.y_mm, transform = $('backTransform').value;
  if (transform === 'mirror_x' || transform === 'rotate_180') x = state.widthMm - x;
  if (transform === 'mirror_y' || transform === 'rotate_180') y = state.heightMm - y;
  return {x_mm: x * state.calibrations.back.width_mm / state.widthMm, y_mm: y * state.calibrations.back.height_mm / state.heightMm};
}
function draw() {
  const side = $('activeSide').value, image = state.images[side], calibration = state.calibrations[side];
  if (!image || !calibration) { ctx.clearRect(0, 0, canvas.width, canvas.height); return; }
  canvas.width = image.width; canvas.height = image.height;
  const annotated = $('showAnnotation').checked ? state.annotatedImages[side] : null;
  ctx.drawImage(annotated || image, 0, 0);
  const displayed = state.outline.map(point => canonicalToVisible(point, side));
  ctx.strokeStyle = '#ff3344'; ctx.lineWidth = 3; ctx.beginPath();
  displayed.forEach((point, index) => { const x = point.x_mm / calibration.width_mm * canvas.width, y = point.y_mm / calibration.height_mm * canvas.height; index ? ctx.lineTo(x, y) : ctx.moveTo(x, y); });
  ctx.closePath(); ctx.stroke();
  state.terminals.filter(t => t.side === side).forEach(terminal => {
    const point = terminal.source_region?.center || canonicalToVisible(terminal.position, side), x = point.x_mm / calibration.width_mm * canvas.width, y = point.y_mm / calibration.height_mm * canvas.height;
    ctx.fillStyle = terminal.polarity === 'positive' ? '#ffcc2288' : terminal.polarity === 'negative' ? '#32c7ff88' : '#d946ef88';
    if (terminal.source_region) drawRegion(terminal.source_region, calibration, ctx.fillStyle, '#111827', false); else { ctx.beginPath(); ctx.arc(x, y, 9, 0, Math.PI * 2); ctx.fill(); }
    ctx.fillStyle = '#111'; ctx.font = '18px sans-serif'; ctx.fillText(terminal.id, x + 12, y);
  });
  state.candidates.filter(candidate => candidate.side === side).forEach(candidate => {
    const point = candidate.visible_region?.center || candidate.visible_position, x = point.x_mm / calibration.width_mm * canvas.width, y = point.y_mm / calibration.height_mm * canvas.height;
    const regions = candidate.matched_regions || (candidate.visible_region ? [candidate.visible_region] : []);
    regions.forEach(region => drawRegion(region, calibration, '#ff5fa244', '#ff5fa2', true));
    if (!regions.length) { ctx.save(); ctx.strokeStyle = '#ff5fa2'; ctx.lineWidth = 4; ctx.setLineDash([10, 7]); ctx.beginPath(); ctx.arc(x, y, 15, 0, Math.PI * 2); ctx.stroke(); ctx.restore(); }
    ctx.fillStyle = '#ff5fa2'; ctx.font = 'bold 22px sans-serif'; ctx.fillText(`${candidate.label}?`, x + 20, y + 7);
  });
}

function drawRegion(region, calibration, fillStyle, strokeStyle, dashed) {
  const points = region.polygon || [];
  if (!points.length) return;
  ctx.save(); ctx.fillStyle = fillStyle; ctx.strokeStyle = strokeStyle; ctx.lineWidth = 4; if (dashed) ctx.setLineDash([10, 7]);
  ctx.beginPath(); points.forEach((point, index) => { const x = point.x_mm / calibration.width_mm * canvas.width, y = point.y_mm / calibration.height_mm * canvas.height; index ? ctx.lineTo(x, y) : ctx.moveTo(x, y); });
  ctx.closePath(); ctx.fill(); ctx.stroke(); ctx.restore();
}
function updateAlignment() {
  const front = state.calibrations.front, back = state.calibrations.back;
  if (!front || !back) { state.alignmentError = null; return; }
  const transformed = back.outline.map(point => visibleToCanonical(point, 'back'));
  const directed = (a, b) => a.reduce((sum, p) => sum + Math.min(...b.map(q => Math.hypot(p.x_mm - q.x_mm, p.y_mm - q.y_mm))), 0) / a.length;
  state.alignmentError = (directed(state.outline, transformed) + directed(transformed, state.outline)) / 2;
  const level = state.alignmentError <= .2 ? '良好' : state.alignmentError <= .5 ? '需仔细确认' : '不合格';
  show(`双面轮廓对齐误差 ${state.alignmentError.toFixed(3)} mm（${level}）`);
}
function list() {
  $('terminalList').innerHTML = state.terminals.map((terminal, index) => `<li>${terminal.id}: ${terminal.side === 'front' ? '正面' : '反面'} · ${terminal.roles.join('+')} / ${terminal.polarity || 'signal'} · 区域 ${terminal.width_mm} × ${terminal.height_mm} mm，中心 ${terminal.position.x_mm}, ${terminal.position.y_mm} mm <button class="tiny" data-remove="${index}">删除</button></li>`).join('');
  document.querySelectorAll('[data-remove]').forEach(button => button.onclick = () => { state.terminals.splice(+button.dataset.remove, 1); draw(); list(); });
}

$('create').onclick = async () => {
  try {
    if (!$('outlineConfirmed').checked) throw new Error('请先确认轮廓、反面映射和真实尺寸');
    if (!state.calibrations.front) throw new Error('必须先标定正面照片');
    if (state.terminals.some(t => t.side === 'back') && !state.calibrations.back) throw new Error('存在反面触点，必须标定反面照片');
    if (state.alignmentError !== null && state.alignmentError > .5) throw new Error('正反面轮廓对齐误差超过 0.5 mm');
    const [vmin, vnom, vmax] = $('voltage').value.split(',').map(Number), [continuous, peak] = $('current').value.split(',').map(Number);
    const spec = {name: $('name').value, protection_ic: $('ic').value, battery: {count: +$('count').value, connection: $('connection').value, cell_min_v: vmin, cell_nominal_v: vnom, cell_max_v: vmax}, limits: {continuous_current_a: continuous, peak_current_a: peak, peak_duration_s: +$('peakDuration').value, ambient_temp_c: +$('ambient').value, max_temp_rise_c: +$('tempRise').value, overcurrent_trip_a: +$('overcurrentTrip').value}, outline: {points: state.outline, source: state.calibrations.front.method || 'photo', confirmed: true}, terminals: state.terminals, photo_capture: {front_calibration_id: state.calibrations.front.calibration_id, back_calibration_id: state.calibrations.back?.calibration_id || null, back_transform: $('backTransform').value, alignment_error_mm: state.alignmentError}};
    const project = await api('/api/projects', {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(spec)}), preview = await api(`/api/projects/${project.id}/preview`, {method: 'POST'});
    $('result').textContent = JSON.stringify({project, preview}, null, 2);
  } catch (error) { $('result').textContent = error.message; }
};

async function api(url, options = {}) {
  const response = await fetch(url, options), data = await response.json();
  if (!response.ok) throw new Error((data.error?.message || '请求失败') + '\n' + JSON.stringify(data.error?.details || {}, null, 2));
  return data;
}
function show(text) { $('calibrationStatus').textContent = text; }
