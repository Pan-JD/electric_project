'use strict';

const $ = id => document.getElementById(id);
const state = {
  meta: null, label: '', issue: '', query: '', cursors: [''], page: 0,
  next: '', request: null, current: null, comparison: null, difference: null,
  relationKind: '', relationOffset: 0, detailRequest: 0, relationRequest: 0,
  thresholds: {}, population: 'reported', score: '', poll: null,
  scoreMode: 'before', scoreChange: '', beforeScore: '', afterScore: '',
  quality: {},
};
const format = value => Number(value || 0).toLocaleString('zh-CN');
const metricFormat = value => Number.isFinite(value) ? value.toLocaleString('zh-CN', {maximumFractionDigits: 2}) : '未记录';

function thresholdsActive() {
  return ['gray_low', 'gray_high', 'laplacian_max', 'tenengrad_max'].some(key => key in state.thresholds);
}

function updateGrayInputs() {
  const mode = $('grayMode').value;
  $('grayLowGroup').hidden = mode === 'below';
  $('grayHighGroup').hidden = mode === 'above';
  $('grayLow').disabled = mode === 'below';
  $('grayHigh').disabled = mode === 'above';
  $('grayLowLabel').textContent = mode === 'above' ? '阈值' : '下限';
  $('grayHighLabel').textContent = mode === 'below' ? '阈值' : '上限';
  $('grayLow').placeholder = mode === 'above' ? '例如 230' : '不限';
  $('grayHigh').placeholder = mode === 'below' ? '例如 30' : '不限';
}

function readThresholds() {
  const mode = $('grayMode').value;
  const result = {metric_scope: $('metricScope').value, gray_mode: mode};
  for (const [id, key] of [['grayLow', 'gray_low'], ['grayHigh', 'gray_high'],
    ['laplacianMax', 'laplacian_max'], ['tenengradMax', 'tenengrad_max']]) {
    if ((id === 'grayLow' && mode === 'below') || (id === 'grayHigh' && mode === 'above')) continue;
    if ($(id).value !== '') result[key] = Number($(id).value);
  }
  return result;
}

function metricMatches(index, label, metrics) {
  if (!thresholdsActive()) return false;
  const f = state.thresholds;
  if ((index < 0 ? 'image' : 'target') !== f.metric_scope) return false;
  if (index >= 0 && state.label && label !== state.label) return false;
  if ('gray_low' in f || 'gray_high' in f) {
    const value = metrics.mean_gray;
    if (!Number.isFinite(value)) return false;
    const ok = {below: () => value < f.gray_high, above: () => value > f.gray_low,
      inside: () => value >= f.gray_low && value <= f.gray_high,
      outside: () => value < f.gray_low || value > f.gray_high};
    if (!ok[f.gray_mode]()) return false;
  }
  if ('laplacian_max' in f && !(Number.isFinite(metrics.laplacian_variance) && metrics.laplacian_variance < f.laplacian_max)) return false;
  if ('tenengrad_max' in f && !(Number.isFinite(metrics.tenengrad) && metrics.tenengrad < f.tenengrad_max)) return false;
  return true;
}

function element(tag, text = '', className = '') {
  const node = document.createElement(tag);
  if (text) node.textContent = text;
  if (className) node.className = className;
  return node;
}

async function api(path, values = {}, signal) {
  const response = await fetch(`${path}?${new URLSearchParams(values)}`, {signal});
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || `请求失败 ${response.status}`);
  return data;
}

function title(key) {
  return state.meta.issues.find(item => item.key === key)?.title || key;
}

function classTitle(code) {
  const name = state.meta.class_names?.[code];
  return name ? `${name}（${code}）` : code;
}

function selectedStats() {
  return state.label ? state.meta.class_stats.filter(row => row['类别'] === state.label) : state.meta.stats;
}

const SCORE_ISSUE = 'label_score';
const ISSUE_GROUPS = [
  ['图像质量', ['exposure', 'watermark_occlusion', 'blur', 'low_resolution', 'small_target']],
  ['类内重复', ['exact_duplicate', 'near_duplicate']],
  ['标注质量', [SCORE_ISSUE, 'large_box', 'annotation_conflict', 'box_geometry', 'coordinate_frame']],
  ['文件完整性', ['unavailable', 'annotation_unavailable']],
];

function renderIssueFilters() {
  const remaining = new Map(state.meta.issues.filter(item => item.key !== 'semantic_label_error').map(item => [item.key, item.title]));
  const scoreEnabled = Boolean(state.meta.label_scores?.enabled);
  remaining.set(SCORE_ISSUE, '标注一致性评分（大模型）');
  for (const [caption, keys] of ISSUE_GROUPS) {
    const group = element('optgroup'); group.label = caption;
    for (const key of keys) {
      if (!remaining.has(key)) continue;
      const option = element('option', remaining.get(key)); option.value = key;
      if (key === SCORE_ISSUE && !scoreEnabled) {
        option.disabled = true; option.textContent += ' · 未加载';
      }
      group.append(option); remaining.delete(key);
    }
    if (group.children.length) $('issueFilter').append(group);
  }
  // Preserve any additional issue added to the server in future.
  for (const [key, caption] of remaining) {
    const option = element('option', caption); option.value = key; $('issueFilter').append(option);
  }
  for (const issue of state.meta.issues.filter(item => item.key !== 'semantic_label_error')) {
    const option = element('option', issue.title); option.value = issue.key; $('scoreIssueFilter').append(option);
  }
}

function updateIssueControls() {
  const scoring = $('issueFilter').value === SCORE_ISSUE;
  $('scoreControls').hidden = !scoring;
  if (scoring) $('population').value = 'all';
  const comparing = state.meta.score_comparison?.enabled;
  $('scoreModeLabel').hidden = !comparing;
  const paired = comparing && $('scoreMode').value === 'compare';
  $('scoreChangeControls').hidden = !paired;
  $('scoreFilter').parentElement.hidden = paired;
}

function updateThresholdSummary() {
  const count = (['grayLow', 'grayHigh'].some(id => !$(id).disabled && $(id).value !== '') ? 1 : 0)
    + ['laplacianMax', 'tenengradMax'].filter(id => $(id).value !== '').length;
  $('thresholdSummary').textContent = count ? `${count} 项条件` : '未设置';
  $('thresholdDropdown').classList.toggle('has-filters', count > 0);
}

function renderScoreFilter() {
  const comparison = state.meta.score_comparison;
  const meta = comparison?.enabled && state.scoreMode === 'after' ? comparison.after : state.meta.label_scores;
  $('scoreFilter').disabled = !meta?.enabled;
  if (!meta?.enabled) return;
  const counts = Object.fromEntries([0, 1, 2, 3, 4, 5].map(score => [score, 0]));
  for (const [label, values] of Object.entries(meta.by_class)) {
    if (state.label && state.label !== label) continue;
    for (const score of [0, 1, 2, 3, 4, 5]) counts[score] += values[score] || 0;
  }
  if (!state.label) Object.assign(counts, meta.score_counts);
  const options = [['scored', '全部已评分样本'], ['unscored', '未评分样本'],
    ...[0, 1, 2, 3, 4, 5].map(score => [String(score), `${score} 分（${format(counts[score])}）`])];
  $('scoreFilter').replaceChildren(...options.map(([value, text]) => {
    const option = element('option', text); option.value = value; return option;
  }));
  $('scoreFilter').value = state.score || $('scoreFilter').value || 'scored';
  const scored = state.label ? Object.values(counts).reduce((a, b) => a + b, 0) : meta.matched_samples;
  const scope = selectedStats().find(row => row['问题'] === '任一已检出问题（去重）');
  const total = Number(scope?.['图像总数'] || 0);
  $('scoreSummary').textContent = `当前范围已评分 ${format(scored)} / ${format(total)} 个样本。可按分数复核，也可叠加原扫描问题和指标条件。`;
  const skipped = meta.unmatched_rows + meta.ambiguous_rows + meta.invalid_rows + meta.conflicting_samples;
  if (skipped) $('scoreSummary').textContent += ` 导入时有 ${format(skipped)} 条未匹配、无效或冲突记录，详情见启动日志。`;
  if (comparison?.enabled && state.scoreMode === 'compare') $('scoreSummary').textContent = '按同一类别、同一图片与标注样本对齐；差值 = 改进后 − 改进前。单侧缺少评分显示“未评分”。';
}

function comparisonCaption(row) {
  const before = row.before?.score ?? '未评分', after = row.after?.score ?? '未评分';
  const change = row.delta === null ? '单侧评分' : row.delta === 0 ? '不变' : `${row.delta > 0 ? '+' : ''}${row.delta} 分`;
  return `改进前 ${before} → 改进后 ${after} · ${change}`;
}

function renderComparisonSummary() {
  const panel = $('scoreComparisonSummary'), meta = state.meta.score_comparison;
  panel.hidden = !meta?.enabled || $('issueFilter').value !== SCORE_ISSUE;
  panel.replaceChildren();
  if (panel.hidden) return;
  const data = state.label ? meta.by_class[state.label] : meta.overall;
  panel.append(element('h2', '知识库方案 · 评分变化'));
  const source = version => meta[version].csv_files.map(path => path.split(/[\\/]/).pop()).join('、');
  panel.append(element('p', `改进前：${source('before')}　→　改进后：${source('after')}`, 'comparison-source'));
  if (!data) { panel.append(element('p', '当前类别尚无评分。')); return; }
  panel.append(element('p', '统计范围：当前类别的全部评分样本（全部类别时按样本×类别计数）。升降比例和均分使用两版共同样本；不随下方问题或阈值筛选改变。', 'threshold-help'));
  const stats = element('div', '', 'score-change-stats');
  for (const [key, name] of [['paired','共同评分'],['down','降分'],['same','不变'],['up','升分'],['before_only','仅改进前'],['after_only','仅改进后']]) {
    const button = element('button', '', `change-stat change-${key}`); button.type = 'button';
    button.append(element('span', name), element('strong', format(data[key])));
    if (['up','down','same'].includes(key)) button.append(element('small', `${data.paired ? (100*data[key]/data.paired).toFixed(1) : '0.0'}%`));
    button.onclick = () => { $('scoreMode').value = 'compare'; $('scoreChange').value = key; $('beforeScore').value = ''; $('afterScore').value = ''; updateIssueControls(); filter(); };
    stats.append(button);
  }
  panel.append(stats, element('p', `共同样本均分：${metricFormat(data.before_mean)} → ${metricFormat(data.after_mean)}；平均变化 ${data.mean_delta > 0 ? '+' : ''}${metricFormat(data.mean_delta)} 分。结合图片和标注框复核评分是否更符合规范。`, 'threshold-help'));
  const details = element('details', '', 'transition-details');
  details.append(element('summary', '分数转移表 · 点击格子查看对应图片'));
  const table = element('table'), head = element('tr'); head.append(element('th', '前 ↓ / 后 →'));
  for (let i=0; i<=5; i++) head.append(element('th', `${i} 分`));
  table.append(head);
  data.matrix.forEach((counts, i) => {
    const tr = element('tr'); tr.append(element('th', `${i} 分`));
    counts.forEach((count, j) => {
      const cell = element('td'), button = element('button', format(count), i === j ? 'same-transition' : '');
      button.type = 'button'; button.disabled = !count; button.title = `${i} → ${j} 分：${count} 个样本`;
      button.onclick = () => { $('scoreMode').value='compare'; $('scoreChange').value='paired'; $('beforeScore').value=String(i); $('afterScore').value=String(j); updateIssueControls(); filter(); };
      cell.append(button); tr.append(cell);
    }); table.append(tr);
  });
  details.append(table); panel.append(details);
}

function renderStats() {
  renderScoreFilter();
  renderComparisonSummary();
  const rows = selectedStats();
  const union = rows.find(row => row['问题'] === '任一已检出问题（去重）');
  $('classFilter').title = state.label ? classTitle(state.label) : '全部类别';
  $('scope').textContent = `${state.label ? classTitle(state.label) : '全部类别合并后'}共 ${format(union?.['图像总数'])} 个统计样本，${format(union?.['问题图像数'])} 个触发问题规则`;
  $('stats').replaceChildren();
  for (const row of rows) {
    if (row['问题'] === '任一已检出问题（去重）' || row['问题'] === '语义错标/漏标' || !Number(row['问题图像数'])) continue;
    const node = element('div', '', 'stat');
    node.append(element('span', row['问题']), element('strong', `${format(row['问题图像数'])} · ${row['比例(%)']}%`));
    $('stats').append(node);
  }
  $('listTitle').textContent = `${state.label ? classTitle(state.label) : '全部类别'} · ${state.scoreChange ? '评分前后对比' : state.score ? '标注评分筛选' : thresholdsActive() ? '阈值筛选' : '图片复核'}`;
  $('listNote').textContent = '正在统计当前筛选结果…';
}

function missingImage(container, image) {
  image.remove();
  container.append(element('div', '图片暂不可读取，连接原磁盘后可查看；点击仍可查看尺寸、标注和诊断依据', 'image-error'));
}

function card(item) {
  const node = element('article', '', 'card');
  node.tabIndex = 0;
  node.setAttribute('role', 'button');
  node.setAttribute('aria-label', `复核 ${item.path}`);
  const picture = element('div', '', 'thumbnail');
  const image = element('img');
  image.loading = 'lazy';
  image.alt = item.path.split(/[\\/]/).pop();
  image.src = `/api/image?${new URLSearchParams({id: item.file_id, size: 480})}`;
  image.onerror = () => missingImage(picture, image);
  picture.append(image);
  const body = element('div', '', 'card-body');
  const name = element('div', image.alt, 'filename');
  name.title = item.path;
  body.append(name, element('p', `${item.width || '?'} × ${item.height || '?'} 像素`, 'dimensions'));
  for (const row of item.quality_scores || []) {
    body.append(element('p', `${classTitle(row.class)} · 图像 ${metricFormat(row.image)} / 综合 ${metricFormat(row.total)} · 覆盖率 ${metricFormat(row.coverage)}%`, 'matched-metrics'));
  }
  for (const region of item.metric_matches || []) {
    const caption = region.scope === 'image' ? '整图' : `目标 ${region.object_index + 1} · ${classTitle(region.class)}`;
    body.append(element('p', `${caption}：灰度 ${metricFormat(region.mean_gray)} / 方差 ${metricFormat(region.laplacian_variance)} / 梯度 ${metricFormat(region.tenengrad)}`, 'matched-metrics'));
  }
  const tags = element('div', '', 'tags');
  for (const row of item.score_comparison || []) {
    body.append(element('div', `${state.label ? '' : classTitle(row.class)+' · '}${comparisonCaption(row)}`, `score-pair delta-${row.delta === null ? 'missing' : row.delta > 0 ? 'up' : row.delta < 0 ? 'down' : 'same'}`));
  }
  for (const score of state.meta.score_comparison?.enabled ? [] : item.label_scores || []) {
    tags.append(element('span', `标注一致性 ${score.score}/5 · ${classTitle(score.class)}`, `tag label-score score-${score.score}`));
  }
  if (state.meta.label_scores?.enabled && !item.label_scores?.length && !item.score_comparison?.length) tags.append(element('span', '标注一致性：未评分', 'tag unscored'));
  if (item.metric_matches?.length) tags.append(element('span', '当前阈值命中', 'tag current-match'));
  for (const key of item.issues) tags.append(element('span', `原扫描 · ${title(key)}`, `tag${key === 'annotation_conflict' || key === 'coordinate_frame' ? ' warning' : ''}`));
  if (!item.issues.length) tags.append(element('span', '原扫描未触发规则', 'tag'));
  body.append(tags);
  node.append(picture, body);
  node.onclick = () => openDetail(item.uid);
  node.onkeydown = event => {
    if (event.key === 'Enter' || event.key === ' ') { event.preventDefault(); openDetail(item.uid); }
  };
  return node;
}

async function loadSamples() {
  clearTimeout(state.poll);
  state.request?.abort();
  const controller = new AbortController();
  state.request = controller;
  $('message').textContent = '正在查询…';
  $('previous').disabled = true;
  $('next').disabled = true;
  $('cards').replaceChildren();
  $('empty').hidden = true;
  renderStats();
  try {
    const result = await api('/api/samples', {
      class: state.label, issue: state.issue, query: state.query, after: state.cursors[state.page],
      population: state.population, score: state.score, ...state.thresholds, ...state.quality,
      score_version: state.scoreMode === 'after' ? 'after' : 'before', score_change: state.scoreChange,
      before_score: state.beforeScore, after_score: state.afterScore,
    }, controller.signal);
    if (state.request !== controller) return;
    if (result.pending) {
      const progress = result.metric_index;
      $('message').textContent = `首次建立阈值索引：${format(progress.processed)} / ${format(progress.total)} 个样本（${(100 * progress.processed / Math.max(1, progress.total)).toFixed(1)}%）。从已保存指标读取，完成后自动显示结果。`;
      $('listNote').textContent = '正在准备指标查询，后续启动直接复用索引';
      state.poll = setTimeout(loadSamples, 1500);
      return;
    }
    $('cards').replaceChildren(...result.items.map(card));
    $('empty').hidden = result.items.length > 0;
    $('page').textContent = `第 ${state.page + 1} 页`;
    $('previous').disabled = state.page === 0;
    $('next').disabled = !result.has_more;
    state.next = result.next;
    const proportion = result.scope_total ? (100 * result.total / result.scope_total).toFixed(2) : '0.00';
    const description = thresholdsActive() ? '当前阈值与其他筛选共同命中' : '当前筛选匹配';
    $('listNote').textContent = `${description} ${format(result.total)} 个样本，占${state.label ? '该类别' : '总体'} ${format(result.scope_total)} 个样本的 ${proportion}%；每页最多 24 张`;
    $('message').textContent = '';
  } catch (error) {
    if (error.name !== 'AbortError') $('message').textContent = error.message;
  }
}

function filter() {
  if (!$('filters').reportValidity()) return;
  state.label = $('classFilter').value;
  const scoring = $('issueFilter').value === SCORE_ISSUE;
  state.issue = scoring ? $('scoreIssueFilter').value : $('issueFilter').value;
  state.query = $('search').value.trim();
  state.population = $('population').value;
  state.scoreMode = state.meta.score_comparison?.enabled ? $('scoreMode').value : 'before';
  const comparing = scoring && state.scoreMode === 'compare';
  state.score = scoring && !comparing ? ($('scoreFilter').value || 'scored') : '';
  state.scoreChange = comparing ? $('scoreChange').value : '';
  state.beforeScore = comparing ? $('beforeScore').value : '';
  state.afterScore = comparing ? $('afterScore').value : '';
  state.thresholds = readThresholds();
  state.quality = $('qualityField').value ? {quality_field:$('qualityField').value,
    quality_min:$('qualityMin').value, quality_max:$('qualityMax').value,
    quality_coverage:$('qualityCoverage').value} : {};
  updateThresholdSummary();
  state.page = 0;
  state.cursors = [''];
  loadSamples();
}

function svgNode(tag, values) {
  const node = document.createElementNS('http://www.w3.org/2000/svg', tag);
  for (const [key, value] of Object.entries(values)) node.setAttribute(key, value);
  return node;
}

function imageFrame(record) {
  const selected = $('frame').value;
  return selected === 'scan' ? (record.scan_frame === 'display' ? 'display' : 'encoded') : selected;
}

function drawImage(target, record, differences) {
  target.replaceChildren();
  const frame = imageFrame(record);
  let width = record.raw_width || record.width || 1;
  let height = record.raw_height || record.height || 1;
  if (frame === 'display' && [5, 6, 7, 8].includes(record.orientation)) [width, height] = [height, width];
  const wrapper = element('div', '', 'picture');
  wrapper.style.setProperty('--ratio', `${width} / ${height}`);
  wrapper.style.maxWidth = `${560 * width / height}px`;
  const image = element('img');
  image.alt = '图片与标注框';
  image.src = `/api/image?${new URLSearchParams({id: record.file_id, frame, size: 1600})}`;
  image.onerror = () => missingImage(wrapper, image);
  wrapper.append(image);
  if ($('showBoxes').checked) {
    const overlay = svgNode('svg', {viewBox: `0 0 ${width} ${height}`});
    const triggered = thresholdsActive()
      ? new Set(record.boxes.filter(box => box.eligible && metricMatches(box.index, box.class, box.metrics || {})).map(box => box.index))
      : new Set(record.findings.map(item => item.index));
    for (const box of differences || record.boxes) {
      if (!box.coords || !box.coords.every(Number.isFinite)) continue;
      if ($('onlyClass').checked && state.label && box.class !== state.label) continue;
      const [x1, y1, x2, y2] = box.coords;
      const color = differences ? (box.difference ? '#ffae49' : '#22d5b8') : (triggered.has(box.index) ? '#ff6973' : '#22d5b8');
      overlay.append(svgNode('rect', {x: Math.min(x1, x2), y: Math.min(y1, y2),
        width: Math.abs(x2 - x1), height: Math.abs(y2 - y1), stroke: color,
        'stroke-dasharray': x2 <= x1 || y2 <= y1 ? '6 4' : ''}));
      const label = svgNode('text', {x: Math.max(0, Math.min(x1, x2)), y: Math.max(14, Math.min(y1, y2) - 5), fill: color});
      label.textContent = `${box.index + 1} ${box.class}`;
      overlay.append(label);
    }
    wrapper.append(overlay);
  }
  target.append(wrapper);
}

function fileInfo(target, record) {
  target.replaceChildren();
  const declared = record.declared_size ? record.declared_size.join(' × ') : '未记录';
  target.append(element('p', `图片 ${record.width || '?'} × ${record.height || '?'} 像素 · XML 尺寸 ${declared} · EXIF ${record.orientation ?? '未记录'} · ${record.boxes.length} 个目标`));
  target.append(element('p', `类别 ${record.classes.map(classTitle).join('、') || '无'} · 合并归档副本 ${record.occurrences} 份`));
  target.append(element('p', `图片 ${record.path}`, 'path'), element('p', `标注 ${record.label_path || '缺失'}`, 'path'));
  const links = element('div', '', 'links');
  for (const [text, route] of [['打开原图', '/api/original'], ['查看 XML / TXT', '/api/label']]) {
    if (route.endsWith('label') && !record.label_path) continue;
    const link = element('a', text);
    link.href = `${route}?id=${record.file_id}`;
    link.target = '_blank';
    link.rel = 'noopener';
    links.append(link);
  }
  target.append(links);
  if (record.paths.length > 1) {
    const details = element('details');
    details.append(element('summary', '查看归档路径示例'));
    for (const path of record.paths) details.append(element('p', path.path, 'path'));
    target.append(details);
  }
}

function renderLabelScores(record) {
  $('labelScoreSection').hidden = !state.meta.label_scores?.enabled;
  $('labelScoreDetail').replaceChildren();
  if (state.meta.score_comparison?.enabled) {
    for (const item of record.score_comparison || []) {
      const row = element('div', '', 'evidence-row label-score-row');
      row.append(element('b', `${classTitle(item.class)} · ${comparisonCaption(item)}`));
      for (const [key, name] of [['before', '改进前'], ['after', '改进后']]) {
        const score = item[key];
        row.append(element('p', score ? `${name}：${score.score}/5 · ${score.source_file} · 第 ${score.source_row} 行` : `${name}：未评分`));
        if (score) row.append(element('p', `图片 ${score.image_name} · 标注 ${score.label_name || '未提供'}`, 'threshold-help'));
      }
      $('labelScoreDetail').append(row);
    }
    if (!record.score_comparison?.length) $('labelScoreDetail').append(element('p', '该样本在当前类别两版均未评分。'));
    return;
  }
  for (const item of record.label_scores || []) {
    const row = element('div', '', `evidence-row label-score-row score-${item.score}`);
    row.append(element('b', `${item.score} / 5 分 · ${classTitle(item.class)}`),
      element('p', `图片 ${item.image_name} · 标注 ${item.label_name || '未提供'}`),
      element('p', `来源 ${item.source_file.split(/[\\/]/).pop()}，第 ${item.source_row} 行`, 'threshold-help'));
    $('labelScoreDetail').append(row);
  }
  if (!record.label_scores?.length) $('labelScoreDetail').append(element('p', '该样本在当前类别尚未评分。', 'threshold-help'));
}

function renderInspection() {
  if (!state.current) return;
  drawImage($('primaryImage'), state.current, state.difference?.left);
  fileInfo($('primaryInfo'), state.current);
  renderLabelScores(state.current);
  renderMetricTable(state.current);
  $('thresholdNotice').hidden = !thresholdsActive();
  $('thresholdNotice').textContent = state.thresholds.metric_scope === 'image'
    ? '当前阈值筛选使用整图指标。下表突出显示命中的整图数值，目标框供定位。'
    : '红框为满足当前全部阈值的目标，青框为其余目标；下表突出显示对应指标。并排对照时仍以橙色和青色展示标注差异。';
  $('comparisonPanel').hidden = !state.comparison;
  if (state.comparison) {
    drawImage($('comparisonImage'), state.comparison, state.difference?.right);
    fileInfo($('comparisonInfo'), state.comparison);
  }
  const uncertain = !state.current.frame_ok || (state.comparison && !state.comparison.frame_ok);
  $('frameNotice').hidden = !uncertain && $('frame').value === 'scan';
  $('frameNotice').textContent = uncertain
    ? '需要复核图像方向与标注坐标。当前框沿用 XML 中的数值，可切换编码像素方向和 EXIF 显示方向，检查哪种方向与框位置一致。XML 尺寸与 EXIF 值见图片下方。'
    : '当前显示所选图片方向，并沿用标注坐标叠框。';
}

function renderMetricTable(record) {
  const table = element('table');
  const head = element('tr');
  for (const text of ['区域', '平均灰度', '暗像素≤10', '亮像素≥245', '灰度P10 / P50 / P90', '对比度P90−P10', '拉普拉斯方差', '梯度能量', '清晰度计算尺寸', '当前筛选']) head.append(element('th', text));
  table.append(head);
  const regions = [[-1, '', record.metrics || {}], ...record.boxes.filter(box => box.eligible && box.metrics && Object.keys(box.metrics).length)
    .map(box => [box.index, box.class, box.metrics])];
  for (const [index, label, metrics] of regions) {
    const matched = metricMatches(index, label, metrics);
    const row = element('tr', '', matched ? 'metric-match-row' : '');
    const size = metrics.measurement_width && metrics.measurement_height ? `${metrics.measurement_width} × ${metrics.measurement_height}` : '未记录';
    for (const text of [index < 0 ? '整图' : `目标 ${index + 1} · ${classTitle(label)}`, metricFormat(metrics.mean_gray),
      Number.isFinite(metrics.dark_fraction_le10) ? `${metricFormat(100*metrics.dark_fraction_le10)}%` : '未记录',
      Number.isFinite(metrics.bright_fraction_ge245) ? `${metricFormat(100*metrics.bright_fraction_ge245)}%` : '未记录',
      [metrics.p10,metrics.p50,metrics.p90].map(metricFormat).join(' / '),metricFormat(metrics.contrast_p90_p10),
      metricFormat(metrics.laplacian_variance), metricFormat(metrics.tenengrad), size, matched ? '命中' : '—']) row.append(element('td', text));
    table.append(row);
  }
  $('qualityMetrics').replaceChildren(table);
  const scales = state.meta.metric_scales;
  $('metricScaleNote').textContent = `灰度0～255；P50是中位数，P90−P10表示主要灰度范围。暗/亮占比统计接近纯黑/纯白的像素。灰度使用区域原始像素。清晰度按整图最长边至多 ${scales.image_metric_side || '未记录'}、目标区域最长边至多 ${scales.roi_metric_side || '未记录'} 像素计算，小图保留原尺寸。数值显示到两位小数，筛选使用保存的完整精度。`;
}

function differenceTable(boxes, caption) {
  const group = element('div');
  group.append(element('h3', caption));
  const table = element('table');
  const head = element('tr');
  for (const text of ['序号', '类别', '坐标 xmin, ymin, xmax, ymax', '对照结果']) head.append(element('th', text));
  table.append(head);
  for (const box of boxes) {
    const row = element('tr', '', box.difference ? 'diff-row' : '');
    for (const text of [box.index + 1, classTitle(box.class), box.coords?.join(', ') || '坐标无效', box.difference ? '本版本独有' : '两边共有']) row.append(element('td', String(text)));
    table.append(row);
  }
  group.append(table);
  return group;
}

async function compare(uid, node) {
  const currentUid = state.current.uid;
  const request = ++state.detailRequest;
  $('relationNote').textContent = '正在加载对照标注…';
  try {
    const [other, difference] = await Promise.all([
      api('/api/detail', {uid, class: state.label}),
      api('/api/difference', {left: currentUid, right: uid}),
    ]);
    if (request !== state.detailRequest) return;
    state.comparison = other;
    state.difference = difference;
    $('comparisonTitle').textContent = state.relationKind === 'annotation' ? '另一份标注版本' : '重复对照样本';
    document.querySelectorAll('.related button').forEach(button => button.classList.remove('selected'));
    node?.classList.add('selected');
    renderInspection();
    const leftCount = difference.left.filter(box => box.difference).length;
    const rightCount = difference.right.filter(box => box.difference).length;
    const relationship = difference.same_image ? '两边图片文件内容相同' : '两边图片文件内容不同';
    $('relationNote').textContent = `${relationship}，${difference.same_annotation ? '类别与框坐标组合相同' : '类别或框坐标组合存在差异'}。`;
    $('differenceSection').hidden = false;
    $('differenceNote').textContent = `当前版本独有 ${leftCount} 个框，对照版本独有 ${rightCount} 个框。共有框按类别与坐标逐项匹配，橙色显示差异。`;
    const tables = element('div', '', 'difference-tables');
    tables.append(differenceTable(difference.left, '当前样本'), differenceTable(difference.right, '对照样本'));
    $('differenceTables').replaceChildren(tables);
  } catch (error) {
    if (request === state.detailRequest) $('relationNote').textContent = error.message;
  }
}

async function loadRelated(kind, append = false) {
  if (!state.current) return;
  const request = ++state.relationRequest;
  const uid = state.current.uid;
  if (!append) {
    state.detailRequest++;
    state.relationKind = kind;
    state.relationOffset = 0;
    state.comparison = null;
    state.difference = null;
    $('differenceSection').hidden = true;
    $('related').replaceChildren();
    renderInspection();
  }
  document.querySelectorAll('.relation-buttons button').forEach(button => button.classList.toggle('active', button.dataset.kind === kind));
  $('relationNote').textContent = '正在查询对照组…';
  $('moreRelated').hidden = true;
  try {
    const result = await api('/api/related', {uid, kind, class: state.label, offset: state.relationOffset});
    if (request !== state.relationRequest || uid !== state.current.uid) return;
    let first;
    for (const item of result.items) {
      const button = element('button');
      button.append(element('b', item.path.split(/[\\/]/).pop()), element('span', `${item.width} × ${item.height} 像素`));
      button.title = item.path;
      button.onclick = () => compare(item.uid, button);
      $('related').append(button);
      if (!first) first = [item.uid, button];
    }
    state.relationOffset += result.items.length;
    $('moreRelated').hidden = !result.has_more;
    $('relationNote').textContent = result.items.length ? '点击对照文件，查看两边图片与标注。' : '当前类别和问题下暂无对照记录';
    if (!append && first) await compare(...first);
  } catch (error) {
    if (request === state.relationRequest) $('relationNote').textContent = error.message;
  }
}

async function openDetail(uid) {
  state.current = null;
  state.comparison = null;
  state.difference = null;
  state.relationRequest++;
  const request = ++state.detailRequest;
  $('detailTitle').textContent = '正在加载图片与标注…';
  for (const id of ['primaryImage', 'primaryInfo', 'evidence', 'related', 'relationButtons', 'differenceTables', 'qualityMetrics']) $(id).replaceChildren();
  $('comparisonPanel').hidden = true;
  $('differenceSection').hidden = true;
  $('frameNotice').hidden = true;
  $('thresholdNotice').hidden = true;
  $('labelScoreSection').hidden = true;
  $('totalScoreSection').hidden = true;
  $('moreRelated').hidden = true;
  $('relationNote').textContent = '';
  $('frame').value = 'scan';
  if (!$('reviewDialog').open) $('reviewDialog').showModal();
  try {
    const record = await api('/api/detail', {uid, class: state.label});
    if (request !== state.detailRequest) return;
    state.current = record;
    $('totalScoreSection').hidden = !state.meta.quality_scores?.enabled;
    $('totalScoreDetail').replaceChildren();
    for (const row of record.quality_scores || []) {
      const block = element('div', '', 'evidence-row');
      block.append(element('b', `${classTitle(row.class)} · 评分覆盖率 ${metricFormat(row.coverage)}%`));
      for (const [key,caption] of Object.entries(state.meta.quality_scores.fields)) {
        block.append(element('p', `${caption}：${row[key] == null ? '待评分' : metricFormat(row[key])+' / 5'}`));
      }
      block.append(element('p', `评分样本ID：${row.sample_id} · CSV第 ${row.source_row} 行`, 'threshold-help'));
      $('totalScoreDetail').append(block);
    }
    if (!record.quality_scores?.length) $('totalScoreDetail').append(element('p', '当前样本尚未关联综合评分。'));
    $('detailTitle').textContent = record.path.split(/[\\/]/).pop();
    renderInspection();
    for (const item of record.findings) {
      const row = element('div', '', 'evidence-row');
      row.append(element('b', item.title), document.createTextNode(item.text));
      $('evidence').append(row);
    }
    if (!record.findings.length) $('evidence').append(element('p', '该样本在原扫描中未触发问题规则。', 'threshold-help'));
    const keys = new Set(record.findings.map(item => item.issue));
    const relations = [['annotation', 'annotation_conflict', '对照标注版本'],
      ['exact', 'exact_duplicate', '展开精确重复'], ['near', 'near_duplicate', '展开近重复']];
    for (const [kind, issue, text] of relations) {
      if (!keys.has(issue)) continue;
      const button = element('button', text);
      button.dataset.kind = kind;
      button.onclick = () => loadRelated(kind);
      $('relationButtons').append(button);
    }
    if (keys.has('annotation_conflict')) await loadRelated('annotation');
    else if (keys.has('exact_duplicate')) await loadRelated('exact');
    else if (keys.has('near_duplicate')) await loadRelated('near');
  } catch (error) {
    if (request === state.detailRequest) $('detailTitle').textContent = error.message;
  }
}

async function start() {
  try {
    state.meta = await api('/api/meta');
    const quality = state.meta.quality_scores;
    $('qualityScoreControls').hidden = !quality?.enabled;
    if (quality?.enabled) {
      for (const [key,caption] of Object.entries(quality.fields)) {
        const option=element('option',caption);option.value=key;$('qualityField').append(option);
      }
      $('qualityField').value='total';
      state.quality={quality_field:'total',quality_min:'0',quality_max:'5',quality_coverage:'0'};
      state.population='all';$('population').value='all';
      $('qualityImportSummary').textContent = `已关联 ${format(quality.matched_samples)} 张。未匹配 ${format(quality.unmatched_class_rows)} 条，重名歧义 ${format(quality.ambiguous_class_rows)} 条，冲突 ${format(quality.conflicting_class_samples)} 条，无效 ${format(quality.invalid_rows)} 行。关联按类别＋原文件名进行；类别记录按样本×类别计数。`;
    }
    $('source').textContent = state.meta.results;
    $('classCount').textContent = `${state.meta.classes.length} 类`;
    for (const label of state.meta.classes) {
      const option = element('option', classTitle(label)); option.value = label; option.title = classTitle(label); $('classFilter').append(option);
    }
    renderIssueFilters();
    if (state.meta.score_comparison?.enabled) {
      $('issueFilter').value = SCORE_ISSUE;
      $('classFilter').value = state.meta.label_scores.default_class || '';
      state.label = $('classFilter').value;
      state.scoreMode = 'compare'; state.scoreChange = 'paired'; state.population = 'all';
      updateIssueControls();
    }
    const summary = state.meta.summary;
    const metrics = [['输入图片路径', summary.file_paths, '扫描发现的归档图片文件'],
      ['按归档规则合并的样本', summary.sample_units, `按文件主名、图片内容及标注合并 ${format(summary.collapsed_classification_copies)} 份副本`],
      ['分类数量', state.meta.classes.length, `按 YOLO ${state.meta.imgsz.join(' × ')} 输入评估目标大小`],
      ['原扫描触发任一问题', summary.issue_union_count, `${(100 * summary.issue_union_count / summary.sample_units).toFixed(2)}% · 包含重复与标注复核项`]];
    for (const [label, value, note] of metrics) {
      const block = element('div', '', 'metric');
      block.append(element('small', label), element('strong', format(value)), element('p', note));
      $('overview').append(block);
    }
    await loadSamples();
  } catch (error) { $('message').textContent = error.message; $('source').textContent = '读取结果失败'; }
}

$('filters').onsubmit = event => { event.preventDefault(); filter(); };
$('classFilter').onchange = filter;
$('issueFilter').onchange = () => { updateIssueControls(); filter(); };
$('scoreIssueFilter').onchange = filter;
$('scoreMode').onchange = () => { state.score = ''; $('scoreFilter').value = 'scored'; updateIssueControls(); filter(); };
for (const id of ['scoreChange', 'beforeScore', 'afterScore']) $(id).onchange = filter;
$('population').onchange = filter;
$('qualityField').onchange = () => { if ($('qualityField').value) $('population').value='all'; filter(); };
$('scoreFilter').onchange = () => {
  if ($('scoreFilter').value) $('population').value = 'all';
  filter();
};
$('grayMode').onchange = () => { updateGrayInputs(); updateThresholdSummary(); };
for (const id of ['grayLow', 'grayHigh', 'laplacianMax', 'tenengradMax']) $(id).oninput = updateThresholdSummary;
$('resetThresholds').onclick = () => {
  for (const id of ['grayLow', 'grayHigh', 'laplacianMax', 'tenengradMax']) $(id).value = '';
  filter();
};
$('previous').onclick = () => { if (state.page > 0) { state.page--; loadSamples(); } };
$('next').onclick = () => { if (state.next) { state.page++; state.cursors[state.page] = state.next; loadSamples(); } };
$('closeDialog').onclick = () => $('reviewDialog').close();
$('reviewDialog').onclose = () => { state.detailRequest++; state.relationRequest++; };
for (const id of ['showBoxes', 'onlyClass', 'frame']) $(id).onchange = renderInspection;
$('moreRelated').onclick = () => loadRelated(state.relationKind, true);
updateGrayInputs();
start();
