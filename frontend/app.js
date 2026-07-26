"use strict";

const root = document.getElementById("app");
const api = JSON.parse(root.dataset.api);
const modelList = document.getElementById("model-list");
const statusPill = document.getElementById("status-pill");
const statusLabel = document.getElementById("status-label");
const logsNode = document.getElementById("logs");
const contextSizeSlider = document.getElementById("context-size-slider");
const contextSizeOutput = document.getElementById("context-size-output");

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
const contextSizes = [16384, 32768, 65536, 131072];
let models = [];
let latestStatus = { running: false, selected_model: null };
let busy = false;

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
  document.querySelectorAll(".model-card").forEach((card) => {
    card.classList.toggle("loaded", isLoaded(card.dataset.model));
  });
  document.getElementById("refresh-button").disabled = busy;
  document.getElementById("clear-graphs-button").disabled = busy;
  document.getElementById("clear-logs-button").disabled = busy;
  contextSizeSlider.disabled = busy;
}

function renderStatus(status) {
  latestStatus = status;
  statusPill.classList.remove("error");
  statusPill.classList.toggle("online", status.running);
  statusLabel.textContent = status.running ? "Server online" : "Server stopped";
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
    const suffix = features.length ? ` · ${features.join(" · ")}` : "";
    meta.textContent = `${model.filename} · ${formatBytes(model.size_bytes)}${suffix}`;
    details.append(name, meta);

    const actions = document.createElement("div");
    actions.className = "model-actions";

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

function renderMetrics(sample) {
  metricHistory.push(sample);
  if (metricHistory.length > maxHistoryPoints) metricHistory.shift();
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

async function loadMetrics() {
  try {
    renderMetrics(await requestJson(api.metrics));
  } catch (error) {
    setNotice("Resource monitor unavailable", true);
  }
}

async function loadLogs() {
  try {
    const payload = await requestJson(`${api.logs}?limit=120`);
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

function updateContextSizeOutput() {
  const contextSize = contextSizes[Number(contextSizeSlider.value)];
  contextSizeOutput.value = `${contextSize / 1024}K`;
}

contextSizeSlider.addEventListener("input", updateContextSizeOutput);
updateContextSizeOutput();

document.getElementById("refresh-button").addEventListener("click", async () => {
  await Promise.all([loadModels(), loadStatus(), loadLogs()]);
});
document.getElementById("clear-graphs-button").addEventListener("click", () => {
  metricHistory.length = 0;
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

Promise.all([loadModels(), loadStatus(), loadMetrics(), loadLogs()]);
window.setInterval(loadStatus, 3000);
window.setInterval(loadMetrics, 1000);
window.setInterval(loadLogs, 3000);
