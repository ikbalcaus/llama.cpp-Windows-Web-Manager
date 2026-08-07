"use strict";

const root = document.getElementById("app");
const api = JSON.parse(root.dataset.api);
const modelList = document.getElementById("model-list");
const statusPill = document.getElementById("status-pill");
const statusLabel = document.getElementById("status-label");
const logsNode = document.getElementById("logs");
const contextSizeSlider = document.getElementById("context-size-slider");
const contextSizeOutput = document.getElementById("context-size-output");
contextSizeSlider.dataset.userModified = "false";

const metricNodes = {
  cpu_percent: document.getElementById("cpu-value"),
  ram_percent: document.getElementById("ram-value"),
  gpu_percent: document.getElementById("gpu-value"),
  vram_percent: document.getElementById("vram-value"),
};
const chartPaths = {
  cpu_percent: document.getElementById("cpu-path"),
  ram_percent: document.getElementById("ram-path"),
  gpu_percent: document.getElementById("gpu-path"),
  vram_percent: document.getElementById("vram-path"),
};

const metricHistory = [];
const maxHistoryPoints = 30;
const metricHistoryStorageKey = "llama-manager.metric-history.v1";
const persistedMetricKeys = [
  "cpu_percent",
  "ram_percent",
  "gpu_percent",
  "vram_percent",
  "ram_used_bytes",
  "ram_total_bytes",
  "vram_used_bytes",
  "vram_total_bytes",
];
const contextSizes = [16384, 32768, 65536, 131072];
let models = [];
let latestStatus = {
  running: false,
  selected_model: null,
  load_mmproj: null,
  web_search: null,
};
let busy = false;
let logsPointerActive = false;
let preserveLogSelectionUntil = 0;

function formatBytes(bytes) {
  if (bytes === null || bytes === undefined) return "—";
  const units = ["B", "KB", "MB", "GB", "TB"];
  let value = bytes;
  let index = 0;
  while (value >= 1024 && index < units.length - 1) {
    value /= 1024;
    index += 1;
  }
  return `${value.toFixed(index > 1 ? 2 : 0)} ${units[index]}`;
}

async function requestJson(url, options = {}) {
  const response = await fetch(url, {
    credentials: "same-origin",
    headers: { "Content-Type": "application/json", ...(options.headers || {}) },
    ...options,
  });
  const payload = await response.json().catch(() => ({ error: "Invalid server response" }));
  if (!response.ok) {
    throw new Error(payload.error || `Request failed (${response.status})`);
  }
  return payload;
}

function setNotice(message, isError = false) {
  if (message && isError) {
    statusPill.classList.add("error");
    statusLabel.textContent = message;
  }
}

function setBusy(value) {
  busy = value;
  updateControls();
}

function isLoaded(filename) {
  return Boolean(latestStatus.running && latestStatus.selected_model === filename);
}

function updateControls() {
  document.querySelectorAll("[data-load-model]").forEach((button) => {
    const loaded = isLoaded(button.dataset.loadModel);
    button.textContent = loaded ? "Reload" : "Load";
    button.disabled = busy;
  });
  document.querySelectorAll("[data-unload-model]").forEach((button) => {
    button.disabled = busy || !isLoaded(button.dataset.unloadModel);
  });
  document.querySelectorAll("[data-load-mmproj]").forEach((checkbox) => {
    checkbox.disabled = busy;
  });
  document.querySelectorAll("[data-web-search]").forEach((checkbox) => {
    checkbox.disabled = busy;
  });
  document.querySelectorAll(".model-card").forEach((card) => {
    card.classList.toggle("loaded", isLoaded(card.dataset.model));
  });
  document.getElementById("refresh-button").disabled = busy;
  document.getElementById("clear-graphs-button").disabled = busy;
  document.getElementById("clear-logs-button").disabled = busy;
  contextSizeSlider.disabled = busy;
}

function syncMmprojControls() {
  if (
    !latestStatus.running
    || typeof latestStatus.load_mmproj !== "boolean"
  ) {
    return;
  }
  document.querySelectorAll("[data-load-mmproj]").forEach((checkbox) => {
    if (
      checkbox.dataset.loadMmproj === latestStatus.selected_model
      && checkbox.dataset.userModified !== "true"
    ) {
      checkbox.checked = latestStatus.load_mmproj;
    }
  });
}

function syncWebSearchControls() {
  if (
    !latestStatus.running
    || typeof latestStatus.web_search !== "boolean"
  ) {
    return;
  }
  document.querySelectorAll("[data-web-search]").forEach((checkbox) => {
    if (
      checkbox.dataset.webSearch === latestStatus.selected_model
      && checkbox.dataset.userModified !== "true"
    ) {
      checkbox.checked = latestStatus.web_search;
    }
  });
}

function syncContextSizeSlider() {
  if (
    !latestStatus.running
    || contextSizeSlider.dataset.userModified === "true"
  ) {
    return;
  }
  const activeIndex = contextSizes.indexOf(latestStatus.context_size);
  if (activeIndex >= 0) {
    contextSizeSlider.value = String(activeIndex);
  }
}

function renderStatus(status) {
  latestStatus = status;
  statusPill.classList.remove("error");
  statusPill.classList.toggle("online", status.running);
  statusLabel.textContent = status.running ? "Server online" : "Server stopped";
  syncContextSizeSlider();
  updateContextSizeOutput();
  syncMmprojControls();
  syncWebSearchControls();
  updateControls();
}

async function loadStatus() {
  try {
    renderStatus(await requestJson(api.status));
  } catch (error) {
    setNotice(error.message, true);
  }
}

function renderModels() {
  modelList.replaceChildren();
  if (!models.length) {
    const empty = document.createElement("p");
    empty.className = "empty";
    empty.textContent = "No loadable .gguf files found in the models folder.";
    modelList.append(empty);
    return;
  }

  models.forEach((model) => {
    const card = document.createElement("article");
    card.className = "model-card";
    card.dataset.model = model.filename;

    const details = document.createElement("div");
    details.className = "model-details";
    const name = document.createElement("strong");
    name.textContent = model.name;
    const meta = document.createElement("span");
    const features = [];
    if (model.has_mmproj) features.push("vision");
    if (model.uses_mtp) features.push("MTP");
    const quant = model.quantization ? ` · ${model.quantization.toUpperCase()}` : "";
    const suffix = features.length ? ` · ${features.join(" · ")}` : "";
    meta.textContent = `${model.filename} · ${formatBytes(model.size_bytes)}${quant}${suffix}`;
    details.append(name, meta);

    const actions = document.createElement("div");
    actions.className = "model-actions";
    const featureToggles = document.createElement("div");
    featureToggles.className = "feature-toggles";
    actions.append(featureToggles);

    let mmprojToggle = null;
    if (model.has_mmproj) {
      actions.classList.add("has-mmproj");
      const mmprojControl = document.createElement("label");
      mmprojControl.className = "mmproj-toggle";
      mmprojToggle = document.createElement("input");
      mmprojToggle.type = "checkbox";
      mmprojToggle.checked = (
        isLoaded(model.filename)
        && typeof latestStatus.load_mmproj === "boolean"
      )
        ? latestStatus.load_mmproj
        : true;
      mmprojToggle.dataset.loadMmproj = model.filename;
      mmprojToggle.dataset.userModified = "false";
      mmprojToggle.addEventListener("change", () => {
        mmprojToggle.dataset.userModified = "true";
      });
      mmprojToggle.setAttribute(
        "aria-label",
        `Load MMPROJ for ${model.name}`,
      );
      const switchTrack = document.createElement("span");
      switchTrack.className = "switch-track";
      switchTrack.setAttribute("aria-hidden", "true");
      const switchLabel = document.createElement("span");
      switchLabel.className = "switch-label";
      switchLabel.textContent = "MMPROJ";
      mmprojControl.title = "Load MMPROJ";
      mmprojControl.append(mmprojToggle, switchLabel, switchTrack);
      featureToggles.append(mmprojControl);
    }

    const webSearchControl = document.createElement("label");
    webSearchControl.className = "mmproj-toggle web-search-toggle";
    const webSearchToggle = document.createElement("input");
    webSearchToggle.type = "checkbox";
    webSearchToggle.checked = (
      isLoaded(model.filename)
      && typeof latestStatus.web_search === "boolean"
    )
      ? latestStatus.web_search
      : true;
    webSearchToggle.dataset.webSearch = model.filename;
    webSearchToggle.dataset.userModified = "false";
    webSearchToggle.addEventListener("change", () => {
      webSearchToggle.dataset.userModified = "true";
    });
    webSearchToggle.setAttribute(
      "aria-label",
      `Enable Web Search for ${model.name}`,
    );
    const webSearchTrack = document.createElement("span");
    webSearchTrack.className = "switch-track";
    webSearchTrack.setAttribute("aria-hidden", "true");
    const webSearchLabel = document.createElement("span");
    webSearchLabel.className = "switch-label";
    webSearchLabel.textContent = "WEB SEARCH";
    webSearchControl.title = "Enable Web Search";
    webSearchControl.append(webSearchToggle, webSearchLabel, webSearchTrack);
    featureToggles.append(webSearchControl);

    const unload = document.createElement("button");
    unload.type = "button";
    unload.className = "button danger";
    unload.dataset.unloadModel = model.filename;
    unload.textContent = "Unload";
    unload.addEventListener("click", () => {
      performAction(api.stop, {}, "");
    });

    const load = document.createElement("button");
    load.type = "button";
    load.className = "button";
    load.dataset.loadModel = model.filename;
    load.textContent = "Load";
    load.addEventListener("click", () => {
      performAction(
        api.start,
        {
          model: model.filename,
          context_size: contextSizes[Number(contextSizeSlider.value)],
          load_mmproj: mmprojToggle ? mmprojToggle.checked : false,
          web_search: webSearchToggle.checked,
        },
        "",
      );
    });

    actions.append(load, unload);
    card.append(details, actions);
    modelList.append(card);
  });
  updateControls();
}

async function loadModels() {
  try {
    const payload = await requestJson(api.models);
    models = payload.models;
    renderModels();
  } catch (error) {
    setNotice(error.message, true);
  }
}

async function performAction(url, body, successMessage) {
  setBusy(true);
  setNotice("");
  try {
    const status = await requestJson(url, { method: "POST", body: JSON.stringify(body) });
    renderStatus(status);
    setNotice(successMessage || "");
    await loadLogs();
  } catch (error) {
    setNotice(error.message, true);
    await loadStatus();
  } finally {
    setBusy(false);
  }
}

function valueLabel(value) {
  return value === null || value === undefined ? "N/A" : `${value.toFixed(1)}%`;
}

function normalizeMetricSample(sample) {
  const normalized = {};
  persistedMetricKeys.forEach((key) => {
    const value = sample && sample[key];
    normalized[key] = typeof value === "number" && Number.isFinite(value)
      ? value
      : null;
  });
  return normalized;
}

function restoreMetricHistory() {
  try {
    const saved = JSON.parse(localStorage.getItem(metricHistoryStorageKey) || "[]");
    if (!Array.isArray(saved)) return;
    saved.slice(-maxHistoryPoints).forEach((sample) => {
      if (sample && typeof sample === "object" && !Array.isArray(sample)) {
        metricHistory.push(normalizeMetricSample(sample));
      }
    });
  } catch (error) {
    try {
      localStorage.removeItem(metricHistoryStorageKey);
    } catch (storageError) {
      // Ignore browsers that disable localStorage entirely.
    }
  }
}

function persistMetricHistory() {
  try {
    localStorage.setItem(metricHistoryStorageKey, JSON.stringify(metricHistory));
  } catch (error) {
    // Monitoring continues in memory if browser storage is unavailable.
  }
}

function pathForMetric(key) {
  const left = 12;
  const right = 446;
  const top = 12;
  const bottom = 192;
  const denominator = Math.max(1, maxHistoryPoints - 1);
  let path = "";
  let drawing = false;
  metricHistory.forEach((sample, index) => {
    const value = sample[key];
    if (value === null || value === undefined) {
      drawing = false;
      return;
    }
    const historyOffset = maxHistoryPoints - metricHistory.length;
    const x = left + ((historyOffset + index) / denominator) * (right - left);
    const y = bottom - (Math.max(0, Math.min(100, value)) / 100) * (bottom - top);
    path += `${drawing ? " L" : "M"} ${x.toFixed(2)} ${y.toFixed(2)}`;
    drawing = true;
  });
  return path;
}

function renderMetricHistory() {
  const sample = metricHistory[metricHistory.length - 1];
  if (!sample) return;
  Object.keys(metricNodes).forEach((key) => {
    metricNodes[key].textContent = valueLabel(sample[key]);
    chartPaths[key].setAttribute("d", pathForMetric(key));
  });
  const ramDetail = sample.ram_used_bytes !== null && sample.ram_total_bytes
    ? `${formatBytes(sample.ram_used_bytes)} / ${formatBytes(sample.ram_total_bytes)} · ${valueLabel(sample.ram_percent)}`
    : valueLabel(sample.ram_percent);
  const vramDetail = sample.vram_used_bytes !== null && sample.vram_total_bytes
    ? `${formatBytes(sample.vram_used_bytes)} / ${formatBytes(sample.vram_total_bytes)} · ${valueLabel(sample.vram_percent)}`
    : valueLabel(sample.vram_percent);
  metricNodes.ram_percent.textContent = ramDetail;
  metricNodes.vram_percent.textContent = vramDetail;
}

function renderMetrics(sample) {
  metricHistory.push(normalizeMetricSample(sample));
  if (metricHistory.length > maxHistoryPoints) metricHistory.shift();
  persistMetricHistory();
  renderMetricHistory();
}

async function loadMetrics() {
  try {
    renderMetrics(await requestJson(api.metrics));
  } catch (error) {
    setNotice("Resource monitor unavailable", true);
  }
}

function hasLogTextSelection() {
  const selection = window.getSelection();
  if (!selection || selection.isCollapsed) return false;
  return (
    logsNode.contains(selection.anchorNode)
    || logsNode.contains(selection.focusNode)
  );
}

function shouldPreserveLogSelection() {
  return (
    logsPointerActive
    || Date.now() < preserveLogSelectionUntil
    || hasLogTextSelection()
  );
}

async function loadLogs() {
  try {
    const payload = await requestJson(`${api.logs}?limit=120`);
    if (shouldPreserveLogSelection()) return;

    const distanceFromBottom =
      logsNode.scrollHeight - logsNode.scrollTop - logsNode.clientHeight;
    const shouldFollowLatest = distanceFromBottom <= 24;
    const previousScrollTop = logsNode.scrollTop;

    logsNode.replaceChildren();
    if (!payload.logs.length) {
      const empty = document.createElement("p");
      empty.className = "log-empty";
      empty.textContent = "No output logs yet.";
      logsNode.append(empty);
      return;
    }
    payload.logs.forEach((entry) => {
      const line = document.createElement("div");
      line.className = "log-line";
      const time = document.createElement("time");
      time.textContent = entry.timestamp.slice(11, 19);
      const source = document.createElement("span");
      source.className = "log-source";
      source.textContent = entry.source;
      const message = document.createElement("span");
      message.className = "log-message";
      message.textContent = entry.message;
      line.append(time, source, message);
      logsNode.append(line);
    });
    logsNode.scrollTop = shouldFollowLatest
      ? logsNode.scrollHeight
      : previousScrollTop;
  } catch (error) {
    setNotice("Console unavailable", true);
  }
}

logsNode.addEventListener("pointerdown", () => {
  logsPointerActive = true;
});
window.addEventListener("pointerup", () => {
  logsPointerActive = false;
  preserveLogSelectionUntil = Date.now() + 500;
});
window.addEventListener("pointercancel", () => {
  logsPointerActive = false;
  preserveLogSelectionUntil = Date.now() + 500;
});

function updateContextSizeOutput() {
  const selectedContextSize = contextSizes[Number(contextSizeSlider.value)];
  const activeContextSize = latestStatus.running
    ? latestStatus.context_size
    : null;
  contextSizeOutput.value = activeContextSize
    ? `${activeContextSize / 1024}K in use`
    : `${selectedContextSize / 1024}K`;
}

contextSizeSlider.addEventListener("input", () => {
  contextSizeSlider.dataset.userModified = "true";
  updateContextSizeOutput();
});
updateContextSizeOutput();

document.getElementById("refresh-button").addEventListener("click", async () => {
  await Promise.all([loadModels(), loadStatus(), loadLogs()]);
});
document.getElementById("clear-graphs-button").addEventListener("click", () => {
  metricHistory.length = 0;
  try {
    localStorage.removeItem(metricHistoryStorageKey);
  } catch (error) {
    // The in-memory history is still cleared when storage is unavailable.
  }
  Object.values(chartPaths).forEach((path) => path.setAttribute("d", ""));
  Object.values(metricNodes).forEach((node) => {
    node.textContent = "—";
  });
});
document.getElementById("clear-logs-button").addEventListener("click", async () => {
  try {
    await requestJson(api.clear_logs, { method: "POST", body: "{}" });
    await loadLogs();
  } catch (error) {
    setNotice(error.message, true);
  }
});

restoreMetricHistory();
renderMetricHistory();
Promise.all([loadModels(), loadStatus(), loadMetrics(), loadLogs()]);
window.setInterval(loadStatus, 3000);
window.setInterval(loadMetrics, 1100);
window.setInterval(loadLogs, 3000);
