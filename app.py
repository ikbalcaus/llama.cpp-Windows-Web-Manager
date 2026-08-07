from __future__ import annotations

import atexit
import csv
import io
import ipaddress
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import webbrowser
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import psutil
from dotenv import load_dotenv
from flask import Flask, Response, jsonify, render_template, request, url_for
from werkzeug.middleware.proxy_fix import ProxyFix

from web_search import (
    DEFAULT_AGENTIC_MAX_TURNS,
    DEFAULT_FETCH_MAX_CHARS,
    DEFAULT_MAX_RESULTS,
    DEFAULT_SEARXNG_URL,
    DEFAULT_TIMEOUT_SECONDS,
    build_web_search_arguments,
    validate_web_search_configuration,
)


PROJECT_DIR = Path(__file__).resolve().parent
load_dotenv(PROJECT_DIR / ".env")

FRONTEND_DIR = Path(
    os.environ.get("FRONTEND_DIR", str(PROJECT_DIR / "frontend"))
).expanduser().resolve()
MODELS_DIR = Path(
    os.environ.get("MODELS_DIR", str(PROJECT_DIR / "models"))
).expanduser().resolve()
LLAMA_SERVER_PATH = Path(
    os.environ.get("LLAMA_SERVER_PATH", str(PROJECT_DIR / "llama-server.exe"))
).expanduser().resolve()
CADDY_PATH = Path(
    os.environ.get("CADDY_PATH", str(PROJECT_DIR / "caddy.exe"))
).expanduser().resolve()
CADDYFILE_PATH = Path(
    os.environ.get("CADDYFILE_PATH", str(PROJECT_DIR / "Caddyfile"))
).expanduser().resolve()
LLAMA_HOST = os.environ["LLAMA_HOST"]
LLAMA_PORT = int(os.environ["LLAMA_PORT"])
FLASK_HOST = os.environ["FLASK_HOST"]
FLASK_PORT = int(os.environ["FLASK_PORT"])
CADDY_PORT = int(os.environ["CADDY_PORT"])
ENABLE_NGROK = os.environ.get("ENABLE_NGROK", "false").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
NGROK_DOMAIN = os.environ.get("NGROK_DOMAIN", "")
LLAMA_DEFAULT_CONTEXT_SIZE = int(os.environ["LLAMA_DEFAULT_CONTEXT_SIZE"])
LLAMA_ALIAS = os.environ["LLAMA_ALIAS"]
LLAMA_MTP_SPEC_TYPE = os.environ["LLAMA_MTP_SPEC_TYPE"]
LLAMA_MTP_DRAFT_N_MAX = int(os.environ["LLAMA_MTP_DRAFT_N_MAX"])
LLAMA_NO_MMPROJ_OFFLOAD = (
    os.environ["LLAMA_NO_MMPROJ_OFFLOAD"].strip().lower() in {"1", "true", "yes", "on"}
)
WEB_LOG_LINES = int(os.environ["WEB_LOG_LINES"])
TRAY_ENABLED = os.environ["TRAY_ENABLED"].strip().lower() in {"1", "true", "yes", "on"}
QUANTIZATION_RE = re.compile(r"(?i)^q[0-9]+$")
TRAY_TOOLTIP = os.environ["TRAY_TOOLTIP"].strip()
SEARXNG_URL = os.environ.get("SEARXNG_URL", DEFAULT_SEARXNG_URL).strip()
WEB_SEARCH_MAX_RESULTS = int(
    os.environ.get("WEB_SEARCH_MAX_RESULTS", str(DEFAULT_MAX_RESULTS))
)
WEB_SEARCH_TIMEOUT = int(
    os.environ.get("WEB_SEARCH_TIMEOUT", str(DEFAULT_TIMEOUT_SECONDS))
)
WEB_FETCH_MAX_CHARS = int(
    os.environ.get("WEB_FETCH_MAX_CHARS", str(DEFAULT_FETCH_MAX_CHARS))
)
WEB_SEARCH_MAX_TURNS = int(
    os.environ.get("WEB_SEARCH_MAX_TURNS", str(DEFAULT_AGENTIC_MAX_TURNS))
)

ALLOWED_CONTEXT_SIZES = (16 * 1024, 32 * 1024, 64 * 1024, 128 * 1024)
class ForwardedPrefixMiddleware:
    """Apply Caddy's trusted prefix so Flask generates prefix-aware URLs."""

    def __init__(self, app: Any) -> None:
        self.app = app

    def __call__(self, environ: dict[str, Any], start_response: Any) -> Iterable[bytes]:
        prefix = environ.get("HTTP_X_FORWARDED_PREFIX", "")
        if prefix:
            normalized = "/" + prefix.strip("/")
            environ["SCRIPT_NAME"] = normalized
        return self.app(environ, start_response)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalized_path(path: str | Path) -> str:
    return os.path.normcase(str(Path(path).resolve()))


def validate_context_size(value: Any) -> int:
    if value is None:
        value = LLAMA_DEFAULT_CONTEXT_SIZE
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError("context_size must be an integer")
    if value not in ALLOWED_CONTEXT_SIZES:
        allowed = ", ".join(str(size) for size in ALLOWED_CONTEXT_SIZES)
        raise ValueError(f"context_size must be one of: {allowed}")
    return value


def validate_load_mmproj(value: Any) -> bool:
    if value is None:
        return True
    if not isinstance(value, bool):
        raise ValueError("load_mmproj must be a boolean")
    return value


def validate_web_search(value: Any) -> bool:
    if value is None:
        return True
    if not isinstance(value, bool):
        raise ValueError("web_search must be a boolean")
    return value


class LlamaServerManager:
    def __init__(self, executable: Path, models_dir: Path) -> None:
        self.executable = executable
        self.models_dir = models_dir
        self._process: subprocess.Popen[str] | None = None
        self._selected_model: str | None = None
        self._context_size: int | None = None
        self._load_mmproj: bool | None = None
        self._web_search: bool | None = None
        self._started_at: str | None = None
        self._logs: deque[dict[str, str]] = deque(maxlen=WEB_LOG_LINES)
        self._lock = threading.RLock()

    @staticmethod
    def _extract_quantization(stem: str) -> str | None:
        for token in reversed(stem.split("-")):
            if QUANTIZATION_RE.fullmatch(token):
                return token
        return None

    @staticmethod
    def _mmproj_stem(stem: str) -> str:
        quantization = LlamaServerManager._extract_quantization(stem)
        if quantization is None:
            return stem
        return "-".join(
            token
            for token in stem.split("-")
            if token.lower() != quantization.lower()
        )

    @classmethod
    def _display_name(cls, stem: str, quantization: str | None) -> str:
        parts = [
            token
            for token in stem.split("-")
            if token.lower() != "mtp"
            and (quantization is None or token.lower() != quantization.lower())
        ]
        name = "-".join(parts)
        return name[:1].upper() + name[1:]

    def scan_models(self) -> list[dict[str, Any]]:
        self.models_dir.mkdir(parents=True, exist_ok=True)
        models: list[dict[str, Any]] = []
        for path in sorted(self.models_dir.glob("*.gguf"), key=lambda item: item.name.lower()):
            if not path.is_file() or "mmproj" in path.name.lower():
                continue
            mmproj_path = path.with_name(f"{self._mmproj_stem(path.stem)}-mmproj.gguf")
            quantization = self._extract_quantization(path.stem)
            display_name = self._display_name(path.stem, quantization)
            models.append(
                {
                    "name": display_name,
                    "filename": path.name,
                    "size_bytes": path.stat().st_size,
                    "quantization": quantization,
                    "has_mmproj": mmproj_path.is_file(),
                    "uses_mtp": "mtp" in path.name.lower(),
                }
            )
        return models

    def _validated_model_path(self, filename: Any) -> Path:
        if not isinstance(filename, str) or not filename or len(filename) > 255:
            raise ValueError("model must be a non-empty GGUF filename")
        if Path(filename).name != filename or not filename.lower().endswith(".gguf"):
            raise ValueError("model must be a filename from the models folder")
        available = {item["filename"]: item for item in self.scan_models()}
        if filename not in available:
            raise ValueError("selected model is not available")
        candidate = (self.models_dir / filename).resolve()
        if candidate.parent != self.models_dir.resolve() or not candidate.is_file():
            raise ValueError("invalid model path")
        return candidate

    def build_command(
        self,
        filename: Any,
        context_size: Any = None,
        load_mmproj: Any = None,
        web_search: Any = None,
    ) -> list[str]:
        model_path = self._validated_model_path(filename)
        selected_context_size = validate_context_size(context_size)
        should_load_mmproj = validate_load_mmproj(load_mmproj)
        should_enable_web_search = validate_web_search(web_search)
        if not self.executable.is_file():
            raise FileNotFoundError(f"llama-server executable not found: {self.executable}")
        command = [
            str(self.executable),
            "-m",
            str(model_path),
        ]
        mmproj_path = model_path.with_name(f"{self._mmproj_stem(model_path.stem)}-mmproj.gguf")
        if should_load_mmproj and mmproj_path.is_file():
            command.extend(["--mmproj", str(mmproj_path)])
            if LLAMA_NO_MMPROJ_OFFLOAD:
                command.append("--no-mmproj-offload")
        command.extend(
            [
                "-c",
                str(selected_context_size),
            ]
        )
        command.extend(["--reasoning", "off", "--reasoning-budget", "0"])
        command.extend(["--alias", LLAMA_ALIAS])
        command.extend(["--host", LLAMA_HOST, "--port", str(LLAMA_PORT)])
        if "mtp" in model_path.name.lower():
            command.extend(
                [
                    "--spec-type",
                    LLAMA_MTP_SPEC_TYPE,
                    "--spec-draft-n-max",
                    str(LLAMA_MTP_DRAFT_N_MAX),
                ]
            )
        web_search_arguments = build_web_search_arguments(
            should_enable_web_search,
            SEARXNG_URL,
            WEB_SEARCH_MAX_RESULTS,
            WEB_SEARCH_TIMEOUT,
            WEB_FETCH_MAX_CHARS,
            WEB_SEARCH_MAX_TURNS,
        )
        if "--jinja" in command and web_search_arguments[:1] == ["--jinja"]:
            web_search_arguments = web_search_arguments[1:]
        command.extend(web_search_arguments)
        return command

    def _matching_processes(self) -> list[psutil.Process]:
        expected = normalized_path(self.executable)
        matches: list[psutil.Process] = []
        for process in psutil.process_iter(["pid", "exe"]):
            try:
                executable = process.info.get("exe")
                if executable and normalized_path(executable) == expected and process.is_running():
                    matches.append(process)
            except (psutil.AccessDenied, psutil.NoSuchProcess, OSError):
                continue
        return matches

    def _model_from_process(self, process: psutil.Process) -> str | None:
        try:
            arguments = process.cmdline()
        except (psutil.AccessDenied, psutil.NoSuchProcess):
            return None
        for index, argument in enumerate(arguments[:-1]):
            if argument in {"--model", "-m"}:
                candidate = Path(arguments[index + 1])
                if candidate.suffix.lower() == ".gguf":
                    return candidate.name
        return None

    def _context_size_from_process(self, process: psutil.Process) -> int | None:
        try:
            arguments = process.cmdline()
        except (psutil.AccessDenied, psutil.NoSuchProcess):
            return None
        for index, argument in enumerate(arguments[:-1]):
            if argument not in {"-c", "--ctx-size"}:
                continue
            raw_context_size = arguments[index + 1]
            if raw_context_size.isascii() and raw_context_size.isdecimal():
                return int(raw_context_size)
        return None

    def _load_mmproj_from_process(self, process: psutil.Process) -> bool | None:
        try:
            arguments = process.cmdline()
        except (psutil.AccessDenied, psutil.NoSuchProcess):
            return None
        return "--mmproj" in arguments

    def _web_search_from_process(self, process: psutil.Process) -> bool | None:
        try:
            arguments = process.cmdline()
        except (psutil.AccessDenied, psutil.NoSuchProcess):
            return None
        return "--mcp-servers-json" in arguments

    def _append_log(self, source: str, message: str) -> None:
        cleaned = message.rstrip("\r\n")
        if not cleaned:
            return
        with self._lock:
            self._logs.append(
                {"timestamp": utc_now(), "source": source, "message": cleaned}
            )

    def _capture_output(self, process: subprocess.Popen[str]) -> None:
        if process.stdout is None:
            return
        try:
            for line in iter(process.stdout.readline, ""):
                self._append_log("llama-server", line)
        except (OSError, ValueError):
            pass
        finally:
            try:
                process.stdout.close()
            except OSError:
                pass
            return_code = process.poll()
            if return_code is not None:
                self._append_log(
                    "manager", f"llama-server exited with code {return_code}"
                )

    def status(self) -> dict[str, Any]:
        with self._lock:
            matches = self._matching_processes()
            process = matches[0] if matches else None
            if process is None:
                self._process = None
                self._context_size = None
                self._load_mmproj = None
                self._web_search = None
                return {
                    "running": False,
                    "pid": None,
                    "selected_model": self._selected_model,
                    "context_size": None,
                    "load_mmproj": None,
                    "web_search": None,
                    "started_at": self._started_at,
                    "process_count": 0,
                    "owned": False,
                }
            owned = self._process is not None and self._process.pid == process.pid
            selected = self._selected_model or self._model_from_process(process)
            context_size = (
                self._context_size
                if owned
                else self._context_size_from_process(process)
            )
            load_mmproj = (
                self._load_mmproj
                if owned
                else self._load_mmproj_from_process(process)
            )
            web_search = (
                self._web_search
                if owned
                else self._web_search_from_process(process)
            )
            return {
                "running": True,
                "pid": process.pid,
                "selected_model": selected,
                "context_size": context_size,
                "load_mmproj": load_mmproj,
                "web_search": web_search,
                "started_at": self._started_at if owned else None,
                "process_count": len(matches),
                "owned": owned,
            }

    def start(
        self,
        filename: Any,
        context_size: Any = None,
        load_mmproj: Any = None,
        web_search: Any = None,
    ) -> dict[str, Any]:
        with self._lock:
            selected_context_size = validate_context_size(context_size)
            selected_load_mmproj = validate_load_mmproj(load_mmproj)
            selected_web_search = validate_web_search(web_search)
            command = self.build_command(
                filename,
                selected_context_size,
                selected_load_mmproj,
                selected_web_search,
            )
            matches = self._matching_processes()
            if matches:
                self.stop()

            command_message = subprocess.list2cmdline(command)
            self._append_log("manager", command_message)

            creation_flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
            try:
                process = subprocess.Popen(
                    command,
                    cwd=str(PROJECT_DIR),
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    bufsize=1,
                    shell=False,
                    creationflags=creation_flags,
                )
            except OSError as exc:
                raise RuntimeError(f"could not start llama-server: {exc}") from exc

            self._process = process
            self._selected_model = Path(str(filename)).name
            self._context_size = selected_context_size
            self._load_mmproj = selected_load_mmproj
            self._web_search = selected_web_search
            self._started_at = utc_now()
            threading.Thread(
                target=self._capture_output,
                args=(process,),
                name="llama-output-reader",
                daemon=True,
            ).start()
            return self.status()

    @staticmethod
    def _terminate_process_tree(process: psutil.Process, timeout: float = 8.0) -> None:
        try:
            children = process.children(recursive=True)
        except (psutil.AccessDenied, psutil.NoSuchProcess):
            children = []
        targets = [process] + children
        for target in reversed(targets):
            try:
                target.terminate()
            except (psutil.AccessDenied, psutil.NoSuchProcess):
                continue
        _, alive = psutil.wait_procs(targets, timeout=timeout)
        for target in alive:
            try:
                target.kill()
            except (psutil.AccessDenied, psutil.NoSuchProcess):
                continue
        if alive:
            psutil.wait_procs(alive, timeout=2.0)

    def stop(self) -> dict[str, Any]:
        with self._lock:
            matches = self._matching_processes()
            if not matches:
                self._process = None
                self._selected_model = None
                self._context_size = None
                self._load_mmproj = None
                self._web_search = None
                self._started_at = None
                return self.status()
            for process in matches:
                self._terminate_process_tree(process)
            self._process = None
            self._selected_model = None
            self._context_size = None
            self._load_mmproj = None
            self._web_search = None
            self._started_at = None
            return self.status()

    def restart(
        self,
        filename: Any | None,
        context_size: Any = None,
        load_mmproj: Any = None,
        web_search: Any = None,
    ) -> dict[str, Any]:
        with self._lock:
            selected = filename or self.status().get("selected_model")
            if not selected:
                raise ValueError("select a model before restarting")
            self._validated_model_path(selected)
            self.stop()
            return self.start(selected, context_size, load_mmproj, web_search)

    def command_preview(
        self,
        filename: Any,
        context_size: Any = None,
        load_mmproj: Any = None,
        web_search: Any = None,
    ) -> str:
        return subprocess.list2cmdline(
            self.build_command(filename, context_size, load_mmproj, web_search)
        )

    def logs(self, limit: int = 120) -> list[dict[str, str]]:
        if not isinstance(limit, int) or isinstance(limit, bool) or not 1 <= limit <= WEB_LOG_LINES:
            raise ValueError(f"limit must be between 1 and {WEB_LOG_LINES}")
        with self._lock:
            return list(self._logs)[-limit:]

    def clear_logs(self) -> None:
        with self._lock:
            self._logs.clear()

    def shutdown(self) -> None:
        with self._lock:
            if self._process is None or self._process.poll() is not None:
                return
        try:
            self.stop()
        except Exception:
            pass


class ResourceMonitor:
    """Sample system resources without blocking request threads."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._nvidia_smi = shutil.which("nvidia-smi")
        self._vram_total = self._windows_vram_total()
        self._latest: dict[str, Any] = {
            "timestamp": utc_now(),
            "cpu_percent": 0.0,
            "ram_percent": psutil.virtual_memory().percent,
            "ram_used_bytes": psutil.virtual_memory().used,
            "ram_total_bytes": psutil.virtual_memory().total,
            "gpu_percent": None,
            "vram_percent": None,
            "vram_used_bytes": None,
            "vram_total_bytes": self._vram_total or None,
            "gpu_source": "unavailable",
        }
        psutil.cpu_percent(interval=None)

    @staticmethod
    def _windows_vram_total() -> int:
        if os.name != "nt":
            return 0
        try:
            import winreg

            root_path = r"SYSTEM\CurrentControlSet\Control\Video"
            total = 0
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, root_path) as root:
                for index in range(winreg.QueryInfoKey(root)[0]):
                    adapter_name = winreg.EnumKey(root, index)
                    try:
                        with winreg.OpenKey(root, rf"{adapter_name}\0000") as adapter:
                            value, _ = winreg.QueryValueEx(
                                adapter, "HardwareInformation.qwMemorySize"
                            )
                            if isinstance(value, int) and value > 0:
                                total += value
                    except OSError:
                        continue
            return total
        except (ImportError, OSError):
            return 0

    def _sample_nvidia(self) -> tuple[float | None, int | None, int | None]:
        if not self._nvidia_smi:
            return None, None, None
        command = [
            self._nvidia_smi,
            "--query-gpu=utilization.gpu,memory.used,memory.total",
            "--format=csv,noheader,nounits",
        ]
        flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=3,
                check=True,
                shell=False,
                creationflags=flags,
            )
            rows = [
                [float(part.strip()) for part in line.split(",")]
                for line in result.stdout.splitlines()
                if line.strip()
            ]
            if not rows:
                return None, None, None
            used = int(sum(row[1] for row in rows) * 1024 * 1024)
            total = int(sum(row[2] for row in rows) * 1024 * 1024)
            return max(row[0] for row in rows), used, total
        except (OSError, subprocess.SubprocessError, ValueError, IndexError):
            return None, None, None

    def _sample_windows_gpu(self) -> tuple[float | None, int | None]:
        if os.name != "nt" or not shutil.which("typeperf"):
            return None, None
        flags = subprocess.CREATE_NO_WINDOW
        command = [
            "typeperf",
            r"\GPU Engine(*)\Utilization Percentage",
            r"\GPU Adapter Memory(*)\Dedicated Usage",
            "-sc",
            "1",
        ]
        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=4,
                check=True,
                shell=False,
                creationflags=flags,
            )
            rows = list(csv.reader(io.StringIO(result.stdout)))
            rows = [row for row in rows if row and len(row) > 1]
            if len(rows) < 2:
                return None, None
            headers = rows[0]
            matching_rows = [row for row in rows[1:] if len(row) == len(headers)]
            if not matching_rows:
                return None, None
            values = matching_rows[-1]
            gpu_engines: dict[str, float] = {}
            vram_values: list[float] = []
            for header, raw_value in zip(headers[1:], values[1:]):
                try:
                    value = float(raw_value)
                except ValueError:
                    continue
                if "GPU Engine(" in header and "Utilization Percentage" in header:
                    match = re.search(
                        r"GPU Engine\(pid_\d+_(.+?)\)\\Utilization Percentage",
                        header,
                        re.IGNORECASE,
                    )
                    engine = match.group(1) if match else header
                    gpu_engines[engine] = gpu_engines.get(engine, 0.0) + value
                elif "GPU Adapter Memory(" in header and "Dedicated Usage" in header:
                    vram_values.append(value)
            return (max(gpu_engines.values()) if gpu_engines else None), (
                int(sum(vram_values)) if vram_values else None
            )
        except (OSError, subprocess.SubprocessError):
            return None, None

    def _sample(self) -> dict[str, Any]:
        memory = psutil.virtual_memory()
        gpu_percent, vram_used, vram_total = self._sample_nvidia()
        source = "nvidia-smi"
        if gpu_percent is None:
            gpu_percent, vram_used = self._sample_windows_gpu()
            vram_total = self._vram_total or None
            source = "Windows performance counters" if gpu_percent is not None else "unavailable"
        vram_percent = None
        if vram_used is not None and vram_total:
            vram_percent = min(100.0, (vram_used / vram_total) * 100.0)
        return {
            "timestamp": utc_now(),
            "cpu_percent": round(psutil.cpu_percent(interval=None), 1),
            "ram_percent": round(memory.percent, 1),
            "ram_used_bytes": memory.used,
            "ram_total_bytes": memory.total,
            "gpu_percent": round(min(100.0, gpu_percent), 1) if gpu_percent is not None else None,
            "vram_percent": round(vram_percent, 1) if vram_percent is not None else None,
            "vram_used_bytes": vram_used,
            "vram_total_bytes": vram_total,
            "gpu_source": source,
        }

    def _run(self) -> None:
        while not self._stop_event.is_set():
            sample = self._sample()
            with self._lock:
                self._latest = sample
            self._stop_event.wait(0.5)

    def latest(self) -> dict[str, Any]:
        with self._lock:
            if self._thread is None:
                self._thread = threading.Thread(
                    target=self._run, name="resource-monitor", daemon=True
                )
                self._thread.start()
            sample = dict(self._latest)
        memory = psutil.virtual_memory()
        sample.update(
            {
                "timestamp": utc_now(),
                "cpu_percent": round(psutil.cpu_percent(interval=None), 1),
                "ram_percent": round(memory.percent, 1),
                "ram_used_bytes": memory.used,
                "ram_total_bytes": memory.total,
            }
        )
        return sample

    def shutdown(self) -> None:
        self._stop_event.set()


manager = LlamaServerManager(LLAMA_SERVER_PATH, MODELS_DIR)
resource_monitor = ResourceMonitor()

app = Flask(
    __name__,
    static_folder=str(FRONTEND_DIR),
    static_url_path="/static",
    template_folder=str(FRONTEND_DIR),
)
app.config.update(
    JSON_SORT_KEYS=False,
    MAX_CONTENT_LENGTH=32 * 1024,
)
app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1)  # type: ignore[method-assign]
app.wsgi_app = ForwardedPrefixMiddleware(app.wsgi_app)  # type: ignore[method-assign]


def require_json_object() -> dict[str, Any]:
    if not request.is_json:
        raise ValueError("request body must be JSON")
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        raise ValueError("request body must be a JSON object")
    return payload


@app.after_request
def security_headers(response: Response) -> Response:
    response.headers["Cache-Control"] = "no-store"
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "same-origin"
    return response


@app.get("/")
def index() -> str:
    return render_template(
        "index.html",
        default_context_size=LLAMA_DEFAULT_CONTEXT_SIZE,
        default_context_index=ALLOWED_CONTEXT_SIZES.index(
            LLAMA_DEFAULT_CONTEXT_SIZE
        ),
        api_urls={
            "models": url_for("api_models"),
            "status": url_for("api_status"),
            "start": url_for("api_start"),
            "stop": url_for("api_stop"),
            "restart": url_for("api_restart"),
            "metrics": url_for("api_metrics"),
            "logs": url_for("api_logs"),
            "clear_logs": url_for("api_clear_logs"),
        },
    )


@app.get("/health")
def health() -> Response:
    return jsonify({"ok": True, "service": "llama-manager"})


@app.get("/api/models")
def api_models() -> Response:
    return jsonify({"models": manager.scan_models()})


@app.get("/api/status")
def api_status() -> Response:
    return jsonify(manager.status())


@app.post("/api/start")
def api_start() -> tuple[Response, int] | Response:
    try:
        payload = require_json_object()
        status = manager.start(
            payload.get("model"),
            payload.get("context_size"),
            payload.get("load_mmproj"),
            payload.get("web_search"),
        )
        return jsonify(status)
    except (ValueError, FileNotFoundError) as exc:
        return jsonify({"error": str(exc)}), 400
    except RuntimeError as exc:
        return jsonify({"error": str(exc)}), 409


@app.post("/api/stop")
def api_stop() -> Response:
    return jsonify(manager.stop())


@app.post("/api/restart")
def api_restart() -> tuple[Response, int] | Response:
    try:
        payload = require_json_object()
        status = manager.restart(
            payload.get("model"),
            payload.get("context_size"),
            payload.get("load_mmproj"),
            payload.get("web_search"),
        )
        return jsonify(status)
    except (ValueError, FileNotFoundError) as exc:
        return jsonify({"error": str(exc)}), 400
    except RuntimeError as exc:
        return jsonify({"error": str(exc)}), 409


@app.get("/api/metrics")
def api_metrics() -> Response:
    return jsonify(resource_monitor.latest())


@app.get("/api/logs")
def api_logs() -> tuple[Response, int] | Response:
    try:
        raw_limit = request.args.get("limit", "120")
        if not raw_limit.isascii() or not raw_limit.isdecimal():
            raise ValueError("limit must be an integer")
        return jsonify({"logs": manager.logs(int(raw_limit))})
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400


@app.post("/api/logs/clear")
def api_clear_logs() -> Response:
    manager.clear_logs()
    return jsonify({"ok": True})


@app.get("/api/command")
def api_command() -> tuple[Response, int] | Response:
    try:
        raw_context_size = request.args.get("context_size")
        raw_load_mmproj = request.args.get("load_mmproj")
        raw_web_search = request.args.get("web_search")
        context_size: Any = None
        if raw_context_size is not None:
            context_size = (
                int(raw_context_size)
                if raw_context_size.isascii() and raw_context_size.isdecimal()
                else raw_context_size
            )
        load_mmproj: Any = None
        if raw_load_mmproj is not None:
            lowered = raw_load_mmproj.strip().lower()
            if lowered in {"1", "true", "yes", "on"}:
                load_mmproj = True
            elif lowered in {"0", "false", "no", "off"}:
                load_mmproj = False
            else:
                load_mmproj = raw_load_mmproj
        web_search: Any = None
        if raw_web_search is not None:
            lowered = raw_web_search.strip().lower()
            if lowered in {"1", "true", "yes", "on"}:
                web_search = True
            elif lowered in {"0", "false", "no", "off"}:
                web_search = False
            else:
                web_search = raw_web_search
        return jsonify(
            {
                "command": manager.command_preview(
                    request.args.get("model"),
                    context_size,
                    load_mmproj,
                    web_search,
                )
            }
        )
    except (ValueError, FileNotFoundError) as exc:
        return jsonify({"error": str(exc)}), 400


@app.errorhandler(404)
def not_found(_: Exception) -> tuple[Response, int]:
    if request.path.startswith("/api/"):
        return jsonify({"error": "not found"}), 404
    return jsonify({"error": "not found"}), 404


@app.errorhandler(413)
def too_large(_: Exception) -> tuple[Response, int]:
    return jsonify({"error": "request body is too large"}), 413


@app.errorhandler(Exception)
def unhandled_error(exc: Exception) -> tuple[Response, int]:
    return jsonify({"error": "internal server error"}), 500


def resolve_ngrok_path() -> Path:
    override = os.environ.get("NGROK_PATH", "").strip()
    if override:
        return Path(override).expanduser().resolve()
    local_path = (PROJECT_DIR / "ngrok.exe").resolve()
    if local_path.is_file():
        return local_path
    discovered = shutil.which("ngrok")
    return Path(discovered).resolve() if discovered else local_path


def validate_configuration() -> None:
    validate_context_size(LLAMA_DEFAULT_CONTEXT_SIZE)
    validate_web_search_configuration(
        SEARXNG_URL,
        WEB_SEARCH_MAX_RESULTS,
        WEB_SEARCH_TIMEOUT,
        WEB_FETCH_MAX_CHARS,
        WEB_SEARCH_MAX_TURNS,
    )
    for name, port in (
        ("LLAMA_PORT", LLAMA_PORT),
        ("FLASK_PORT", FLASK_PORT),
        ("CADDY_PORT", CADDY_PORT),
    ):
        if not 1 <= port <= 65535:
            raise ValueError(f"{name} must be between 1 and 65535")
    if ENABLE_NGROK:
        labels = NGROK_DOMAIN.strip().lower().split(".")
        if (
            not labels
            or any(not label or len(label) > 63 for label in labels)
            or any(label.startswith("-") or label.endswith("-") for label in labels)
            or any(
                not all(character.isalnum() or character == "-" for character in label)
                for label in labels
            )
        ):
            raise ValueError("NGROK_DOMAIN must be a valid hostname")
    required = {
        "LLAMA_SERVER_PATH": LLAMA_SERVER_PATH,
        "CADDY_PATH": CADDY_PATH,
        "CADDYFILE_PATH": CADDYFILE_PATH,
    }
    if ENABLE_NGROK:
        required["NGROK_PATH"] = resolve_ngrok_path()
    missing = [
        f"{name}={path}"
        for name, path in required.items()
        if not path.is_file()
    ]
    if missing:
        raise FileNotFoundError(
            "Missing required executable or configuration file: " + ", ".join(missing)
        )


@dataclass
class ServiceProcess:
    command: list[str]
    process: subprocess.Popen[Any] | None = None

    def stop_existing(self) -> None:
        expected_name = Path(self.command[0]).name.casefold()
        expected_arguments = [argument.casefold() for argument in self.command[1:]]
        for process in psutil.process_iter(["pid", "name", "cmdline"]):
            try:
                command_line = process.info.get("cmdline") or []
                if not command_line:
                    continue
                process_name = (process.info.get("name") or "").casefold()
                arguments = [argument.casefold() for argument in command_line[1:]]
                if process_name != expected_name or arguments != expected_arguments:
                    continue
                terminate_process_tree(process)
            except (psutil.AccessDenied, psutil.NoSuchProcess, OSError):
                continue

    def start(self, environment: dict[str, str]) -> None:
        flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
        self.process = subprocess.Popen(
            self.command,
            cwd=str(PROJECT_DIR),
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            shell=False,
            creationflags=flags,
        )


def terminate_process_tree(parent: psutil.Process) -> None:
    try:
        children = parent.children(recursive=True)
    except (psutil.AccessDenied, psutil.NoSuchProcess):
        return
    targets = [parent] + children
    for target in reversed(targets):
        try:
            target.terminate()
        except (psutil.AccessDenied, psutil.NoSuchProcess):
            continue
    _, alive = psutil.wait_procs(targets, timeout=8)
    for target in alive:
        try:
            target.kill()
        except (psutil.AccessDenied, psutil.NoSuchProcess):
            continue


def terminate_service(process: subprocess.Popen[Any] | None) -> None:
    if process is None or process.poll() is not None:
        return
    try:
        parent = psutil.Process(process.pid)
    except (psutil.AccessDenied, psutil.NoSuchProcess):
        return
    terminate_process_tree(parent)


def detect_lan_ipv4() -> str | None:
    if os.name != "nt":
        return None

    try:
        result = subprocess.run(
            ["ipconfig"],
            capture_output=True,
            text=True,
            check=False,
            creationflags=subprocess.CREATE_NO_WINDOW,
        )
    except OSError:
        return None

    address_pattern = re.compile(
        r"IPv4 Address[^:]*:\s*(\d{1,3}(?:\.\d{1,3}){3})",
        re.IGNORECASE,
    )
    candidates: list[str] = []
    for adapter in re.split(r"(?:\r?\n){2,}", result.stdout):
        match = address_pattern.search(adapter)
        if match is None:
            continue
        address = match.group(1)
        try:
            parsed_address = ipaddress.ip_address(address)
        except ValueError:
            continue
        if (
            not parsed_address.is_private
            or parsed_address.is_loopback
            or parsed_address.is_link_local
        ):
            continue
        candidates.append(address)

        gateway_offset = adapter.lower().find("default gateway")
        if gateway_offset >= 0:
            gateway_text = adapter[gateway_offset:]
            gateway_addresses = re.findall(
                r"\d{1,3}(?:\.\d{1,3}){3}",
                gateway_text,
            )
            if gateway_addresses:
                return address

    return candidates[0] if candidates else None


def local_service_url(port: int) -> str:
    host = detect_lan_ipv4() or "127.0.0.1"
    return f"http://{host}:{port}/"


def add_startup_messages() -> None:
    lan_ipv4 = detect_lan_ipv4() or "127.0.0.1"
    local_base = f"http://{lan_ipv4}"
    local_messages = [
        f'llama.cpp UI started on "{local_base}:{LLAMA_PORT}"',
        (
            "llama.cpp Windows Web Manager started on "
            f'"{local_base}:{FLASK_PORT}"'
        ),
        f'Caddy started on "{local_base}:{CADDY_PORT}"',
    ]
    if ENABLE_NGROK:
        public_base = f"https://{NGROK_DOMAIN}"
        messages = [
            (
                f"llama.cpp UI started on \"{local_base}:{LLAMA_PORT}\" "
                f"and \"{public_base}/{LLAMA_PORT}\""
            ),
            (
                "llama.cpp Windows Web Manager started on \""
                f"{local_base}:{FLASK_PORT}\" and \"{public_base}/{FLASK_PORT}\""
            ),
            (
                f"Caddy started on \"{local_base}:{CADDY_PORT}\" "
                f"and \"{public_base}/{CADDY_PORT}\""
            ),
        ]
    else:
        messages = local_messages
    terminal_time = datetime.now().astimezone().strftime("%H:%M:%S")
    for message in messages:
        manager._append_log("startup", message)
        if sys.stdout is not None:
            print(f"{terminal_time} {message}", flush=True)


def create_tray_image() -> Any:
    from PIL import Image, ImageDraw

    image = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
    drawing = ImageDraw.Draw(image)
    drawing.rounded_rectangle((3, 3, 61, 61), radius=15, fill=(7, 16, 14, 255))
    drawing.polygon(
        ((18, 45), (30, 18), (38, 18), (30, 36), (46, 36), (42, 45)),
        fill=(250, 250, 250, 255),
    )
    return image


def normalize_tray_click_message(message: int) -> int | None:
    windows_left_button_up = 0x0202
    windows_left_button_double_click = 0x0203
    if message == windows_left_button_up:
        return None
    if message == windows_left_button_double_click:
        return windows_left_button_up
    return message


def create_tray_icon(shutdown: Any) -> Any:
    import pystray

    class DoubleClickTrayIcon(pystray.Icon):
        def _on_notify(self, wparam: int, lparam: int) -> Any:
            normalized = normalize_tray_click_message(lparam)
            if normalized is None:
                return None
            return super()._on_notify(wparam, normalized)

    def open_settings(_: Any = None, __: Any = None) -> None:
        webbrowser.open(local_service_url(FLASK_PORT), new=2)

    def open_llama_ui(_: Any = None, __: Any = None) -> None:
        webbrowser.open(local_service_url(LLAMA_PORT), new=2)

    def exit_application(icon: Any, _: Any = None) -> None:
        shutdown()
        icon.stop()

    menu = pystray.Menu(
        pystray.MenuItem("Open Settings", open_settings, default=True),
        pystray.MenuItem("Open llama.cpp UI", open_llama_ui),
        pystray.Menu.SEPARATOR,
        pystray.MenuItem("Exit", exit_application),
    )
    return DoubleClickTrayIcon(
        "llama-cpp-manager", create_tray_image(), TRAY_TOOLTIP, menu
    )


def build_service_processes() -> list[ServiceProcess]:
    services = [
        ServiceProcess(
            [
                str(CADDY_PATH),
                "run",
                "--config",
                str(CADDYFILE_PATH),
                "--adapter",
                "caddyfile",
            ]
        )
    ]
    if ENABLE_NGROK:
        services.append(
            ServiceProcess(
                [
                    str(resolve_ngrok_path()),
                    "http",
                    str(CADDY_PORT),
                    f"--url={NGROK_DOMAIN}",
                ]
            )
        )
    return services


def run() -> int:
    from waitress import create_server

    validate_configuration()
    environment = os.environ.copy()
    services = build_service_processes()
    for service in services:
        service.stop_existing()
    for service in services:
        service.start(environment)
    add_startup_messages()

    server = create_server(app, host=FLASK_HOST, port=FLASK_PORT, threads=8)
    stopping = threading.Event()
    tray_icon: Any | None = None

    def shutdown(_: int | None = None, __: Any = None) -> None:
        if stopping.is_set():
            return
        stopping.set()
        manager.shutdown()
        resource_monitor.shutdown()
        server.close()
        for service in reversed(services):
            terminate_service(service.process)
        if tray_icon is not None:
            tray_icon.stop()

    signal.signal(signal.SIGINT, shutdown)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, shutdown)

    def monitor_services() -> None:
        while not stopping.wait(0.5):
            if any(
                service.process is not None and service.process.poll() is not None
                for service in services
            ):
                shutdown()
                return

    threading.Thread(target=monitor_services, name="service-monitor", daemon=True).start()
    use_tray = os.name == "nt" and TRAY_ENABLED and "--no-tray" not in sys.argv[1:]
    try:
        if use_tray:
            threading.Thread(
                target=server.run,
                name="waitress-server",
                daemon=True,
            ).start()
            tray_icon = create_tray_icon(shutdown)
            tray_icon.run()
        else:
            server.run()
    finally:
        shutdown()
    return 0


atexit.register(manager.shutdown)
atexit.register(resource_monitor.shutdown)

if __name__ == "__main__":
    try:
        raise SystemExit(run())
    except (FileNotFoundError, OSError, ValueError) as error:
        print(f"Startup failed: {error}", file=sys.stderr, flush=True)
        raise SystemExit(1) from None
