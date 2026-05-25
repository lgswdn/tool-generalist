const cardsEl = document.querySelector("#cards");
const detailPane = document.querySelector("#detailPane");
const countLabel = document.querySelector("#countLabel");
const searchInput = document.querySelector("#searchInput");
const stageFilter = document.querySelector("#stageFilter");
const refreshButton = document.querySelector("#refreshButton");
const showAllConfigsInput = document.querySelector("#showAllConfigs");

let experiments = [];
let selectedId = "";
let selectedExperiment = null;
let showAllConfigs = false;
let showAllArtifacts = false;

const stageLabels = {
  config: "配置入口",
  contact_gen: "接触数据",
  pretrain: "预训练",
  rl: "强化学习",
};

async function fetchJson(url) {
  const response = await fetch(url, { cache: "no-store" });
  if (!response.ok) {
    const text = await response.text();
    throw new Error(text || `${response.status} ${response.statusText}`);
  }
  return response.json();
}

function valueText(value) {
  if (value === null || value === undefined || value === "") {
    return "未设置";
  }
  if (Array.isArray(value) || typeof value === "object") {
    return JSON.stringify(value);
  }
  return String(value);
}

function escapeHtml(value) {
  return valueText(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

function stagePills(exp) {
  const stages = new Map((exp.stages || []).map((stage) => [stage.key, stage]));
  return ["contact_gen", "pretrain", "rl"]
    .map((stage) => {
      const item = stages.get(stage);
      const isEnabled = Boolean(item?.enabled);
      const statusText = item?.statusText || (isEnabled ? "启动" : "跳过");
      const statusClass = item?.status === "reused" ? "reused" : isEnabled ? "" : "off";
      const title = reuseChainText(item) || item?.reason || "";
      return `<span class="stage-pill ${statusClass}" title="${escapeHtml(title)}">${stageLabels[stage]} ${escapeHtml(statusText)}</span>`;
    })
    .join("");
}

function statusClass(stage) {
  if (stage.status === "reused") {
    return "reused";
  }
  return stage.enabled ? "" : "off";
}

function statusText(stage) {
  return stage.statusText || (stage.enabled ? "启动" : "跳过");
}

function reuseChainText(stage) {
  const chain = stage?.reuseChain || [];
  if (!chain.length) {
    return "";
  }
  return chain.map((item) => item.config || item.ref).filter(Boolean).join(" -> ");
}

function renderReuseNote(stage) {
  const chain = reuseChainText(stage);
  if (stage.status !== "reused" || !chain) {
    return "";
  }
  return `<p class="reuse-note">复用链: ${escapeHtml(chain)}</p>`;
}

function resultPills(summary = {}) {
  const byCategory = summary.byCategory || {};
  const total = summary.total || 0;
  if (!total) {
    return `<span class="stage-pill off">无结果</span>`;
  }
  return Object.entries(byCategory)
    .map(([category, count]) => `<span class="stage-pill result">${escapeHtml(category)} ${escapeHtml(count)}</span>`)
    .join("");
}

function formatPercent(value) {
  const number = Number(value);
  if (!Number.isFinite(number)) {
    return "未设置";
  }
  return `${(number * 100).toFixed(number >= 0.995 || number <= 0.005 ? 0 : 1)}%`;
}

function evalKindLabel(kind) {
  const labels = {
    multi_tool: "多工具 Eval",
    single_tool: "单工具 Eval",
  };
  return labels[kind] || kind || "Eval";
}

function evalPills(summary = {}) {
  if (!(summary.total > 0)) {
    return "";
  }
  return `<span class="stage-pill eval">Eval ${escapeHtml(summary.total)} / 最新 ${escapeHtml(formatPercent(summary.latestSuccessRate))}</span>`;
}

function matchesFilters(exp) {
  if (!showAllConfigs && !(exp.artifactSummary?.total > 0) && !(exp.evalSummary?.total > 0)) {
    return false;
  }

  const query = searchInput.value.trim().toLowerCase();
  const stage = stageFilter.value;
  const haystack = [
    exp.name,
    exp.file,
    exp.description,
    exp.model,
    exp.runName,
    exp.wandbProject,
    exp.pathsYaml,
  ]
    .filter(Boolean)
    .join(" ")
    .toLowerCase();

  if (query && !haystack.includes(query)) {
    return false;
  }
  if (stage !== "all" && !(exp.enabledStages || []).includes(stage)) {
    return false;
  }
  return true;
}

function getVisibleExperiments() {
  return experiments.filter(matchesFilters);
}

function renderCards() {
  const visible = getVisibleExperiments();
  const defaultHiddenCount = experiments.filter((exp) => !(exp.artifactSummary?.total > 0)).length;
  countLabel.textContent = `${visible.length} / ${experiments.length} 个配置`;
  if (showAllConfigsInput) {
    showAllConfigsInput.checked = showAllConfigs;
    showAllConfigsInput.title = defaultHiddenCount
      ? `${defaultHiddenCount} 个配置没有默认可见的 RL 结果`
      : "所有配置都有默认可见的 RL 结果";
  }
  if (!visible.length) {
    cardsEl.innerHTML = `<div class="empty-list">没有符合当前条件的配置。</div>`;
    return;
  }
  cardsEl.innerHTML = visible
    .map(
      (exp) => `
        <button class="card ${exp.id === selectedId ? "active" : ""}" type="button" data-id="${escapeHtml(exp.id)}">
          <div class="card-title">
            <h2>${escapeHtml(exp.name)}</h2>
            <span class="file-badge" title="${escapeHtml(exp.file)}">${escapeHtml(exp.file)}</span>
          </div>
          <p class="card-description">${escapeHtml(exp.description || "未提供说明")}</p>
          <div class="meta-grid">
            <span class="meta-item">
              <span class="meta-label">模型</span>
              <span class="meta-value" title="${escapeHtml(exp.model)}">${escapeHtml(exp.model)}</span>
            </span>
            <span class="meta-item">
              <span class="meta-label">GPU</span>
              <span class="meta-value">${escapeHtml(exp.numGpus)}</span>
            </span>
            <span class="meta-item">
              <span class="meta-label">Run</span>
              <span class="meta-value" title="${escapeHtml(exp.runName)}">${escapeHtml(exp.runName)}</span>
            </span>
            <span class="meta-item">
              <span class="meta-label">覆盖项</span>
              <span class="meta-value">${escapeHtml(exp.assignmentCount)}</span>
            </span>
            <span class="meta-item">
              <span class="meta-label">结果</span>
              <span class="meta-value">${escapeHtml(exp.artifactSummary?.total || 0)}</span>
            </span>
            <span class="meta-item">
              <span class="meta-label">Eval</span>
              <span class="meta-value">${escapeHtml(exp.evalSummary?.total || 0)}</span>
            </span>
          </div>
          <div class="stage-row">${stagePills(exp)}${resultPills(exp.artifactSummary)}${evalPills(exp.evalSummary)}</div>
        </button>
      `,
    )
    .join("");
}

function renderSummaryTiles(exp) {
  const tiles = [
    ["文件", exp.file],
    ["模型", exp.model],
    ["GPU", exp.numGpus],
    ["路径配置", exp.pathsYaml],
    ["结果数", exp.artifactSummary?.total || 0],
    ["Eval", exp.evalSummary?.total ? `${exp.evalSummary.total} / ${formatPercent(exp.evalSummary.latestSuccessRate)}` : "无"],
  ];
  return tiles
    .map(
      ([label, value]) => `
        <div class="summary-tile">
          <span>${label}</span>
          <strong title="${escapeHtml(value)}">${escapeHtml(value)}</strong>
        </div>
      `,
    )
    .join("");
}

function renderSequence(exp) {
  return `
    <div class="sequence">
      <div class="sequence-track">
        ${(exp.stages || [])
          .map(
            (stage, index) => `
              <div class="sequence-step">
                <div class="sequence-node ${stage.enabled ? "" : "off"}">${index + 1}</div>
                <div class="sequence-card">
                  <h4>${escapeHtml(stage.title)}</h4>
                  <p>${escapeHtml(stage.summary)}</p>
                  <span class="status-pill ${statusClass(stage)}" title="${escapeHtml(reuseChainText(stage))}">${escapeHtml(statusText(stage))}</span>
                  ${renderReuseNote(stage)}
                </div>
              </div>
            `,
          )
          .join("")}
      </div>
    </div>
  `;
}

function renderParams(params) {
  const entries = Object.entries(params || {});
  if (!entries.length) {
    return `<p class="card-description">没有显式关键参数。</p>`;
  }
  return `
    <div class="param-list">
      ${entries
        .map(
          ([key, value]) => `
            <div class="param-row">
              <span class="param-key">${escapeHtml(key)}</span>
              <span class="param-value">${escapeHtml(value)}</span>
            </div>
          `,
        )
        .join("")}
    </div>
  `;
}

function encoderSourceTypeLabel(type) {
  const labels = {
    local_pretrain: "本配置预训练",
    pretrain_reuse: "复用预训练 Encoder",
    checkpoint: "直接使用 Checkpoint",
    none: "未配置 Encoder",
  };
  return labels[type] || type || "未知";
}

function contactSourceLabel(source) {
  const labels = {
    planned_contact_artifact: "Contact artifact",
    inferred_existing_sibling: "推断已有 sibling artifact",
    dataset_manifest: "dataset_manifest",
  };
  return labels[source] || source || "未知";
}

function renderEncoderSource(exp) {
  const source = exp.encoderSource;
  if (!source) {
    return `<p class="card-description">没有 encoder/contact 来源信息。</p>`;
  }
  const contact = source.contactData || {};
  const encoderArtifact = source.encoderArtifact || {};
  const chain = (source.reuseChain || []).map((item) => item.config || item.ref).filter(Boolean).join(" -> ");
  const params = Object.fromEntries(Object.entries({
    "来源类型": encoderSourceTypeLabel(source.type),
    "Encoder cfg": source.encoderConfig,
    "复用链": chain || null,
    "Encoder artifact": encoderArtifact.path,
    "直接 checkpoint": source.directCheckpoint,
    "Pretrain resume checkpoint": source.pretrainResumeCheckpoint,
    "Contact 来源": contactSourceLabel(contact.source),
    "Contact 实际路径": contact.path,
    "Contact expected 路径": contact.expectedPath && contact.expectedPath !== contact.path ? contact.expectedPath : null,
    "Contact 路径存在": contact.path ? contact.exists : null,
    "Dataset manifest": contact.datasetManifest,
    "错误": source.error,
  }).filter(([, value]) => value !== null && value !== undefined && value !== ""));
  return renderParams(params);
}

function renderStageCards(exp) {
  return `
    <div class="stage-grid">
      ${(exp.stages || [])
        .map(
          (stage) => `
            <article class="stage-card">
              <div class="stage-card-header">
                <h4>${escapeHtml(stage.title)}</h4>
                <span class="status-pill ${statusClass(stage)}" title="${escapeHtml(reuseChainText(stage))}">${escapeHtml(statusText(stage))}</span>
              </div>
              ${renderReuseNote(stage)}
              ${renderParams(stage.params)}
            </article>
          `,
        )
        .join("")}
    </div>
  `;
}

function renderAssignments(exp) {
  return `
    <details>
      <summary>查看配置文件中的赋值语句</summary>
      <pre>${escapeHtml(
        (exp.assignments || [])
          .map((item) => `${item.line}: EXP_CFG.${item.path} = ${item.raw}`)
          .join("\n"),
      )}</pre>
    </details>
  `;
}

function renderFullConfig(exp) {
  return `
    <details>
      <summary>查看完整展开后的 JSON 配置</summary>
      <pre>${escapeHtml(JSON.stringify(exp.fullConfig, null, 2))}</pre>
    </details>
  `;
}

function categoryLabel(category) {
  const labels = {
    experiment: "启动记录",
    rl: "RL",
    encoder: "Encoder",
    contact: "Contact",
  };
  return labels[category] || category || "未知";
}

function formatBytes(bytes) {
  if (!bytes) {
    return "0 B";
  }
  const units = ["B", "KB", "MB", "GB", "TB"];
  let value = Number(bytes);
  let index = 0;
  while (value >= 1024 && index < units.length - 1) {
    value /= 1024;
    index += 1;
  }
  return `${value.toFixed(value >= 10 || index === 0 ? 0 : 1)} ${units[index]}`;
}

function formatEnvLayout(item) {
  const gpuCount = Number(item.numGpus);
  const envsPerGpu = Number(item.numEnv);
  if (Number.isFinite(gpuCount) && gpuCount > 0 && Number.isFinite(envsPerGpu) && envsPerGpu > 0) {
    return `${gpuCount} * ${envsPerGpu} = ${gpuCount * envsPerGpu}`;
  }
  return item.numEnv;
}

function renderArtifactSummary(summary = {}) {
  const categories = Object.entries(summary.byCategory || {});
  const statuses = Object.entries(summary.byStatus || {});
  return `
    <div class="artifact-overview">
      <div class="summary-tile">
        <span>总数</span>
        <strong>${escapeHtml(summary.total || 0)}</strong>
      </div>
      <div class="summary-tile">
        <span>阶段</span>
        <strong title="${escapeHtml(categories.map(([key, count]) => `${key}: ${count}`).join(", "))}">
          ${escapeHtml(categories.map(([key, count]) => `${categoryLabel(key)} ${count}`).join(" / ") || "无")}
        </strong>
      </div>
      <div class="summary-tile">
        <span>状态</span>
        <strong title="${escapeHtml(statuses.map(([key, count]) => `${key}: ${count}`).join(", "))}">
          ${escapeHtml(statuses.map(([key, count]) => `${key} ${count}`).join(" / ") || "无")}
        </strong>
      </div>
      <div class="summary-tile">
        <span>最新</span>
        <strong>${escapeHtml(summary.latestCreatedAt)}</strong>
      </div>
    </div>
  `;
}

function getRlArtifacts(exp) {
  const artifacts = exp.artifacts || [];
  return artifacts.filter((item) => item.category === "rl");
}

function getVisibleArtifacts(exp) {
  const artifacts = getRlArtifacts(exp);
  return showAllArtifacts ? artifacts : artifacts.filter((item) => !item.hiddenByDefault);
}

function summarizeArtifactItems(artifacts) {
  const byCategory = {};
  const byStatus = {};
  let latest = null;
  for (const item of artifacts) {
    const category = item.category || "unknown";
    const status = item.status || "unknown";
    byCategory[category] = (byCategory[category] || 0) + 1;
    byStatus[status] = (byStatus[status] || 0) + 1;
    if (!latest || valueText(item.createdAt) > valueText(latest.createdAt)) {
      latest = item;
    }
  }
  return {
    total: artifacts.length,
    byCategory,
    byStatus,
    latestCreatedAt: latest?.createdAt || null,
    latestStatus: latest?.status || null,
    latestStage: latest?.stage || null,
  };
}

function renderArtifactControls(exp) {
  const hiddenCount = getRlArtifacts(exp).filter((item) => item.hiddenByDefault).length;
  return `
    <div class="section-title-row">
      <h3>实验结果</h3>
      <label class="toggle-control">
        <input id="showAllArtifacts" type="checkbox" ${showAllArtifacts ? "checked" : ""} />
        <span>显示全部 RL${hiddenCount ? `（含 ${hiddenCount} 个无有效 checkpoint 的 RL）` : ""}</span>
      </label>
    </div>
  `;
}

function renderArtifactCards(exp) {
  const artifacts = getVisibleArtifacts(exp);
  if (!artifacts.length) {
    const rlCount = getRlArtifacts(exp).length;
    const hiddenCount = getRlArtifacts(exp).filter((item) => item.hiddenByDefault).length;
    if (hiddenCount && !showAllArtifacts) {
      return `<p class="card-description">当前 ${rlCount} 个 RL 都没有第二个 checkpoint，已默认隐藏。</p>`;
    }
    return `<p class="card-description">没有在 artifacts 中找到由这个 config 启动的 RL 结果。</p>`;
  }
  return `
    <div class="artifact-list">
      ${artifacts
        .map(
          (item) => `
            <article class="artifact-card">
              <div class="artifact-card-head">
                <div>
                  <span class="artifact-kind">${escapeHtml(categoryLabel(item.category))}</span>
                  <h4 title="${escapeHtml(item.artifactName)}">${escapeHtml(item.artifactName)}</h4>
                </div>
                <span class="status-pill ${item.status === "running" ? "" : "off"}">${escapeHtml(item.status || "unknown")}</span>
              </div>
              <div class="artifact-meta">
                <span><b>时间</b>${escapeHtml(item.createdAt)}</span>
                <span><b>阶段</b>${escapeHtml(item.stage)}</span>
                <span><b>Run</b>${escapeHtml(item.runName)}</span>
                <span><b>模型</b>${escapeHtml(item.model)}</span>
                <span><b>Contact</b>${escapeHtml(item.contactGen)}</span>
                <span><b>Env</b>${escapeHtml(formatEnvLayout(item))}</span>
                <span><b>Checkpoint</b>${escapeHtml(item.latestCheckpoint ?? (item.hasBestCheckpoint ? "best" : "无"))}</span>
                <span><b>文件</b>${escapeHtml(item.fileCount)} / ${escapeHtml(formatBytes(item.totalBytes))}</span>
              </div>
              ${item.isTrivialCheckpointRun ? `<p class="artifact-note">没有第二个 checkpoint，默认隐藏</p>` : ""}
              ${item.resolvedEncoderCheckpoint ? `<p class="artifact-path">Encoder: ${escapeHtml(item.resolvedEncoderCheckpoint)}</p>` : ""}
              ${item.error ? `<p class="artifact-error">${escapeHtml(item.error)}</p>` : ""}
              ${renderEvalPanels(item.evals)}
              <details>
                <summary>路径与哈希</summary>
                ${renderParams({
                  path: item.path,
                  manifest: item.manifestPath,
                  configHash: item.configHash,
                  gitCommit: item.gitCommit,
                  wandbProject: item.wandbProject,
                  maxIterations: item.maxIterations,
                  saveInterval: item.saveInterval,
                })}
              </details>
              ${renderWandbPanel(item)}
            </article>
          `,
        )
        .join("")}
    </div>
  `;
}

function successRateWidth(value) {
  const number = Number(value);
  if (!Number.isFinite(number)) {
    return 0;
  }
  return Math.max(0, Math.min(100, number * 100));
}

function evalRowMeta(row, summary) {
  const parts = [
    `成功 ${row.successes ?? 0} / Eval ${row.episodes ?? 0}`,
    formatPercent(row.success_rate),
  ];
  if (summary.kind === "multi_tool" && row.variants) {
    parts.push(`${row.variants} variants`);
  }
  return parts.join(" / ");
}

function renderEvalChart(summary) {
  const rows = summary.rows || [];
  if (!rows.length) {
    return `<p class="card-description">这个 eval summary 没有明细行。</p>`;
  }
  return `
    <div class="eval-chart" role="list" aria-label="${escapeHtml(summary.chartTitle)}">
      ${rows
        .map((row) => {
          const width = successRateWidth(row.success_rate).toFixed(2);
          return `
            <div class="eval-bar-row" role="listitem">
              <span class="eval-bar-label" title="${escapeHtml(row.name)}">${escapeHtml(row.name)}</span>
              <div class="eval-bar-track" aria-hidden="true">
                <span class="eval-bar-fill" style="width: ${width}%"></span>
              </div>
              <span class="eval-bar-value" title="${escapeHtml(evalRowMeta(row, summary))}">
                ${escapeHtml(evalRowMeta(row, summary))}
              </span>
            </div>
          `;
        })
        .join("")}
    </div>
  `;
}

function renderEvalPanels(evals = []) {
  if (!evals.length) {
    return "";
  }
  return `
    <div class="eval-panels">
      ${evals
        .map(
          (summary) => `
            <article class="eval-panel">
              <div class="eval-panel-head">
                <div>
                  <span class="artifact-kind">${escapeHtml(evalKindLabel(summary.kind))}</span>
                  <h4>${escapeHtml(summary.chartTitle)}</h4>
                </div>
                <span class="eval-rate">${escapeHtml(formatPercent(summary.successRate))}</span>
              </div>
              <div class="eval-metrics">
                <span><b>总成功率</b>${escapeHtml(formatPercent(summary.successRate))}</span>
                <span><b>成功 / Eval</b>${escapeHtml(summary.successes)} / ${escapeHtml(summary.episodes)}</span>
                <span><b>${escapeHtml(summary.itemLabel)}</b>${escapeHtml(summary.itemCount)}${summary.rawItemCount !== summary.itemCount ? ` / raw ${escapeHtml(summary.rawItemCount)}` : ""}</span>
                <span><b>Checkpoint</b>${escapeHtml((summary.checkpoint || "").split("/").pop())}</span>
              </div>
              ${summary.tool ? `<p class="artifact-path">Tool: ${escapeHtml(summary.tool)}</p>` : ""}
              ${renderEvalChart(summary)}
              <details>
                <summary>Eval 文件与参数</summary>
                ${renderParams({
                  file: summary.file,
                  path: summary.path,
                  task: summary.task,
                  modifiedAt: summary.modifiedAt,
                  worldSize: summary.worldSize,
                  numEnvsPerRank: summary.numEnvsPerRank,
                  episodesPerTool: summary.episodesPerTool,
                  episodesPerObject: summary.episodesPerObject,
                  randomizeObjects: summary.randomizeObjects,
                  objectRandomSeed: summary.objectRandomSeed,
                })}
              </details>
            </article>
          `,
        )
        .join("")}
    </div>
  `;
}

function renderWandbPanel(item) {
  if (item.category !== "rl") {
    return "";
  }
  if (!item.wandb?.panelUrl) {
    return "";
  }
  return `
    <div class="wandb-panel">
      <div class="wandb-panel-head">
        <span>${escapeHtml(item.runName || item.artifactName)}</span>
        <a href="${escapeHtml(item.wandb.runUrl || item.wandb.panelUrl)}" target="_blank" rel="noreferrer">打开 W&B</a>
      </div>
      <iframe
        title="W&B run metrics"
        loading="lazy"
        referrerpolicy="no-referrer-when-downgrade"
        src="${escapeHtml(item.wandb.runUrl || item.wandb.panelUrl)}"
      ></iframe>
    </div>
  `;
}

function renderExperimentDetail(exp) {
  const visibleArtifacts = getVisibleArtifacts(exp);
  detailPane.innerHTML = `
    ${exp.loadError ? `<div class="error-banner">${escapeHtml(exp.loadError)}</div>` : ""}
    <div class="detail-header">
      <h2>${escapeHtml(exp.name)}</h2>
      <p class="detail-subtitle">${escapeHtml(exp.description || "未提供说明")}</p>
      <div class="summary-strip">${renderSummaryTiles(exp)}</div>
    </div>
    <section class="section">
      <h3>过程顺序</h3>
      ${renderSequence(exp)}
    </section>
    <section class="section">
      <h3>Encoder / Contact 来源</h3>
      ${renderEncoderSource(exp)}
    </section>
    <section class="section">
      <h3>步骤参数</h3>
      ${renderStageCards(exp)}
    </section>
    <section class="section">
      ${renderArtifactControls(exp)}
      ${renderArtifactSummary(summarizeArtifactItems(visibleArtifacts))}
      ${renderArtifactCards(exp)}
    </section>
    <section class="section">
      <h3>配置详情</h3>
      ${renderAssignments(exp)}
      ${renderFullConfig(exp)}
    </section>
  `;
}

async function selectExperiment(id, updateHash = true) {
  selectedId = id;
  renderCards();
  detailPane.innerHTML = `<div class="empty-state"><h2>加载中</h2></div>`;

  try {
    const exp = await fetchJson(`/api/experiments/${encodeURIComponent(id)}`);
    if (updateHash) {
      history.replaceState(null, "", `#${encodeURIComponent(id)}`);
    }
    selectedExperiment = exp;
    renderExperimentDetail(exp);
  } catch (error) {
    detailPane.innerHTML = `<div class="error-banner">${escapeHtml(error.message)}</div>`;
  }
}

async function loadExperiments(forceRefresh = false) {
  countLabel.textContent = "加载中";
  cardsEl.innerHTML = "";
  try {
    const data = await fetchJson(forceRefresh ? "/api/experiments?refresh=1" : "/api/experiments");
    experiments = data.experiments || [];
    renderCards();
    const hashId = decodeURIComponent(location.hash.replace(/^#/, ""));
    const visibleExperiments = getVisibleExperiments();
    const initial =
      visibleExperiments.find((item) => item.id === hashId)?.id || visibleExperiments[0]?.id;
    if (initial) {
      await selectExperiment(initial, Boolean(hashId));
    } else {
      selectedId = "";
      selectedExperiment = null;
      detailPane.innerHTML = `<div class="empty-state"><h2>没有可显示的配置</h2><p>打开“显示全部配置”可以查看没有默认可见 RL 结果的配置。</p></div>`;
    }
  } catch (error) {
    countLabel.textContent = "加载失败";
    cardsEl.innerHTML = `<div class="error-banner">${escapeHtml(error.message)}</div>`;
  }
}

cardsEl.addEventListener("click", (event) => {
  const card = event.target.closest(".card");
  if (!card) {
    return;
  }
  selectExperiment(card.dataset.id);
});

searchInput.addEventListener("input", renderCards);
stageFilter.addEventListener("change", renderCards);
refreshButton.addEventListener("click", () => loadExperiments(true));
showAllConfigsInput.addEventListener("change", () => {
  showAllConfigs = showAllConfigsInput.checked;
  const visibleExperiments = getVisibleExperiments();
  renderCards();
  if (!visibleExperiments.some((item) => item.id === selectedId)) {
    const next = visibleExperiments[0]?.id;
    if (next) {
      selectExperiment(next);
    } else {
      selectedId = "";
      selectedExperiment = null;
      detailPane.innerHTML = `<div class="empty-state"><h2>没有可显示的配置</h2><p>调整搜索条件或打开“显示全部配置”。</p></div>`;
    }
  }
});
detailPane.addEventListener("change", (event) => {
  if (event.target?.id !== "showAllArtifacts") {
    return;
  }
  showAllArtifacts = event.target.checked;
  if (selectedExperiment) {
    renderExperimentDetail(selectedExperiment);
  }
});
window.addEventListener("hashchange", () => {
  const id = decodeURIComponent(location.hash.replace(/^#/, ""));
  if (id && id !== selectedId) {
    selectExperiment(id, false);
  }
});

loadExperiments();
