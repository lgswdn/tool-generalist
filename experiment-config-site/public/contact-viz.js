const groupSelect = document.querySelector("#groupSelect");
const prevButton = document.querySelector("#prevButton");
const nextButton = document.querySelector("#nextButton");
const preImage = document.querySelector("#preImage");
const postImage = document.querySelector("#postImage");
const loadingState = document.querySelector("#loadingState");
const groupCounter = document.querySelector("#groupCounter");
const groupTitle = document.querySelector("#groupTitle");
const autoToggle = document.querySelector("#autoToggle");
const currentFrameLabel = document.querySelector("#currentFrameLabel");
const intervalInput = document.querySelector("#intervalInput");
const intervalLabel = document.querySelector("#intervalLabel");
const showPreButton = document.querySelector("#showPreButton");
const showPostButton = document.querySelector("#showPostButton");
const groupSlider = document.querySelector("#groupSlider");
const groupIndexLabel = document.querySelector("#groupIndexLabel");
const preFileLabel = document.querySelector("#preFileLabel");
const postFileLabel = document.querySelector("#postFileLabel");

let groups = [];
let currentIndex = 0;
let currentFrame = "pre";
let timerId = null;

async function fetchJson(url) {
  const response = await fetch(url, { cache: "no-store" });
  if (!response.ok) {
    const text = await response.text();
    throw new Error(text || `${response.status} ${response.statusText}`);
  }
  return response.json();
}

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

function clampIndex(index) {
  if (!groups.length) {
    return 0;
  }
  return Math.max(0, Math.min(groups.length - 1, index));
}

function updateTimer() {
  if (timerId) {
    clearInterval(timerId);
    timerId = null;
  }
  if (!autoToggle.checked || !groups.length) {
    return;
  }
  timerId = setInterval(() => {
    setFrame(currentFrame === "pre" ? "post" : "pre");
  }, Number(intervalInput.value));
}

function setFrame(frame) {
  currentFrame = frame;
  preImage.classList.toggle("active", frame === "pre");
  postImage.classList.toggle("active", frame === "post");
  showPreButton.classList.toggle("active", frame === "pre");
  showPostButton.classList.toggle("active", frame === "post");
  currentFrameLabel.textContent = frame === "pre" ? "Pre" : "Post";
}

function setIndex(index, updateHash = true) {
  if (!groups.length) {
    return;
  }
  currentIndex = clampIndex(index);
  const group = groups[currentIndex];

  groupSelect.value = group.id;
  groupSlider.value = String(currentIndex);
  groupCounter.textContent = `${currentIndex + 1} / ${groups.length}`;
  groupTitle.textContent = group.label;
  groupIndexLabel.textContent = String(group.index).padStart(3, "0");
  preFileLabel.textContent = group.pre.file;
  postFileLabel.textContent = group.post.file;

  loadingState.hidden = false;
  preImage.src = group.pre.url;
  postImage.src = group.post.url;
  preImage.alt = `${group.label} pre`;
  postImage.alt = `${group.label} post`;
  setFrame(currentFrame);
  syncLoadedState();

  if (updateHash) {
    history.replaceState(null, "", `#${encodeURIComponent(group.id)}`);
  }
}

function changeIndex(delta) {
  if (!groups.length) {
    return;
  }
  setIndex((currentIndex + delta + groups.length) % groups.length);
}

function preloadNeighbors() {
  if (!groups.length) {
    return;
  }
  for (const offset of [-1, 1]) {
    const group = groups[(currentIndex + offset + groups.length) % groups.length];
    for (const url of [group.pre.url, group.post.url]) {
      const img = new Image();
      img.src = url;
    }
  }
}

function syncLoadedState() {
  loadingState.hidden = preImage.complete && postImage.complete;
  if (loadingState.hidden) {
    preloadNeighbors();
  }
}

function renderOptions() {
  groupSelect.innerHTML = groups
    .map((group) => `<option value="${escapeHtml(group.id)}">${escapeHtml(group.label)}</option>`)
    .join("");
  groupSlider.max = String(Math.max(0, groups.length - 1));
}

async function init() {
  try {
    const data = await fetchJson("/api/contact-viz");
    groups = data.groups || [];
    if (!groups.length) {
      groupTitle.textContent = "没有找到成对图片";
      loadingState.textContent = "没有找到成对图片";
      return;
    }

    renderOptions();
    const hashId = decodeURIComponent(location.hash.replace(/^#/, ""));
    const hashIndex = groups.findIndex((group) => group.id === hashId);
    setIndex(hashIndex >= 0 ? hashIndex : 0, hashIndex >= 0);
    setFrame("pre");
    updateTimer();
  } catch (error) {
    groupTitle.textContent = "加载失败";
    loadingState.textContent = error.message;
  }
}

preImage.addEventListener("load", syncLoadedState);
postImage.addEventListener("load", syncLoadedState);

groupSelect.addEventListener("change", () => {
  setIndex(groups.findIndex((group) => group.id === groupSelect.value));
});
groupSlider.addEventListener("input", () => {
  setIndex(Number(groupSlider.value));
});
prevButton.addEventListener("click", () => changeIndex(-1));
nextButton.addEventListener("click", () => changeIndex(1));
autoToggle.addEventListener("change", updateTimer);
intervalInput.addEventListener("input", () => {
  intervalLabel.textContent = `${intervalInput.value} ms`;
  updateTimer();
});
showPreButton.addEventListener("click", () => setFrame("pre"));
showPostButton.addEventListener("click", () => setFrame("post"));
window.addEventListener("hashchange", () => {
  const hashId = decodeURIComponent(location.hash.replace(/^#/, ""));
  const nextIndex = groups.findIndex((group) => group.id === hashId);
  if (nextIndex >= 0 && nextIndex !== currentIndex) {
    setIndex(nextIndex, false);
  }
});

init();
