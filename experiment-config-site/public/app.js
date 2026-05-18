const cardsEl = document.querySelector("#cards");
const detailPane = document.querySelector("#detailPane");
const countLabel = document.querySelector("#countLabel");
const searchInput = document.querySelector("#searchInput");
const stageFilter = document.querySelector("#stageFilter");
const refreshButton = document.querySelector("#refreshButton");

let experiments = [];
let selectedId = "";

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
  const enabled = new Set(exp.enabledStages || []);
  return ["contact_gen", "pretrain", "rl"]
    .map((stage) => {
      const isEnabled = enabled.has(stage);
      return `<span class="stage-pill ${isEnabled ? "" : "off"}">${stageLabels[stage]} ${isEnabled ? "ON" : "OFF"}</span>`;
    })
    .join("");
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

function matchesFilters(exp) {
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

function renderCards() {
  const visible = experiments.filter(matchesFilters);
  countLabel.textContent = `${visible.length} / ${experiments.length} 个实验`;
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
          </div>
          <div class="stage-row">${stagePills(exp)}${resultPills(exp.artifactSummary)}</div>
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
                  <span class="status-pill ${stage.enabled ? "" : "off"}">${stage.enabled ? "启用" : "跳过"}</span>
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

function renderStageCards(exp) {
  return `
    <div class="stage-grid">
      ${(exp.stages || [])
        .map(
          (stage) => `
            <article class="stage-card">
              <div class="stage-card-header">
                <h4>${escapeHtml(stage.title)}</h4>
                <span class="status-pill ${stage.enabled ? "" : "off"}">${stage.enabled ? "启用" : "跳过"}</span>
              </div>
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

function renderArtifactCards(exp) {
  const artifacts = exp.artifacts || [];
  if (!artifacts.length) {
    return `<p class="card-description">没有在 artifacts 中找到由这个 config 启动的实验结果。</p>`;
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
                <span><b>Env</b>${escapeHtml(item.numEnv)}</span>
                <span><b>Checkpoint</b>${escapeHtml(item.latestCheckpoint ?? (item.hasBestCheckpoint ? "best" : "无"))}</span>
                <span><b>文件</b>${escapeHtml(item.fileCount)} / ${escapeHtml(formatBytes(item.totalBytes))}</span>
              </div>
              ${item.resolvedEncoderCheckpoint ? `<p class="artifact-path">Encoder: ${escapeHtml(item.resolvedEncoderCheckpoint)}</p>` : ""}
              ${item.error ? `<p class="artifact-error">${escapeHtml(item.error)}</p>` : ""}
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
            </article>
          `,
        )
        .join("")}
    </div>
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
        <h3>步骤参数</h3>
        ${renderStageCards(exp)}
      </section>
      <section class="section">
        <h3>实验结果</h3>
        ${renderArtifactSummary(exp.artifactSummary)}
        ${renderArtifactCards(exp)}
      </section>
      <section class="section">
        <h3>配置详情</h3>
        ${renderAssignments(exp)}
        ${renderFullConfig(exp)}
      </section>
    `;
  } catch (error) {
    detailPane.innerHTML = `<div class="error-banner">${escapeHtml(error.message)}</div>`;
  }
}

async function loadExperiments() {
  countLabel.textContent = "加载中";
  cardsEl.innerHTML = "";
  try {
    const data = await fetchJson("/api/experiments");
    experiments = data.experiments || [];
    renderCards();
    const hashId = decodeURIComponent(location.hash.replace(/^#/, ""));
    const initial = experiments.find((item) => item.id === hashId)?.id || experiments[0]?.id;
    if (initial) {
      await selectExperiment(initial, Boolean(hashId));
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
refreshButton.addEventListener("click", loadExperiments);
window.addEventListener("hashchange", () => {
  const id = decodeURIComponent(location.hash.replace(/^#/, ""));
  if (id && id !== selectedId) {
    selectExperiment(id, false);
  }
});

loadExperiments();
