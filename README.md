# llama.cpp Windows Web Manager

![llama.cpp Windows Web Manager](llama.cpp-windows-web-manager.png)

The complete runtime now lives in `app.py`. Running it starts:

- llama.cpp models on `0.0.0.0:8080` when loaded from the UI
- Flask/Waitress on `0.0.0.0:8081`
- Caddy on `0.0.0.0:8082`
- Cloudflare Tunnel (`cloudflared`) forwarding your public hostname to Caddy
  only when `ENABLE_CLOUDFLARE_TUNNEL=true`

When Cloudflare Tunnel is enabled, the public routes are:

```text
{CLOUDFLARE_TUNNEL_URL}/8080/  -> llama.cpp
{CLOUDFLARE_TUNNEL_URL}/8081/  -> management UI
```

## Windows setup

Open PowerShell in the project:

```powershell
Set-Location "C:\Users\%LOCALAPPDATA%\OneDrive\Documents\llama.cpp Windows Web Manager"
```

Create the environment and install dependencies:

```powershell
py -3.11 -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
python -m pip install -r .\requirements.txt
```

The checked-in `.env` contains the current local paths, ports, model options,
and optional Cloudflare Tunnel settings. The tunnel is disabled by default:

```dotenv
ENABLE_CLOUDFLARE_TUNNEL=false
```

To expose your localhost publicly for free, create a tunnel in the Cloudflare
Zero Trust dashboard (<https://one.dash.cloudflare.com> → Networks →
Tunnels). On the tunnel's overview page choose **Add a replica** to reveal the
installation command and copy the `eyJ...` token. Configure the tunnel's
public hostname to point to `http://localhost:{CADDY_PORT}` (8082 by default).
Then set these locally:

```dotenv
ENABLE_CLOUDFLARE_TUNNEL=true
CLOUDFLARE_TUNNEL_URL=https://your-hostname.example.com
CLOUDFLARE_TUNNEL_TOKEN=your_eyJ_token
```

`cloudflared.exe` is resolved from `CLOUDFLARE_TUNNEL_PATH`, the project
directory, or `PATH`. The token is passed to `cloudflared` through the
`TUNNEL_TOKEN` environment variable and never appears in its command line.

Start the entire stack with one command:

```powershell
.\.venv\Scripts\python.exe .\app.py
```

On Windows this starts the notification-area icon. Double-click it to open the
settings page. A single left-click does nothing. Right-click it to open Settings,
open the llama.cpp UI, or choose **Exit**. Exit cleanly stops Flask, Caddy,
Cloudflare Tunnel when enabled, and any llama-server process loaded by the UI.

To start minimized without a PowerShell window, use:

```powershell
Start-Process .\.venv\Scripts\pythonw.exe -ArgumentList ".\app.py" -WorkingDirectory "C:\Users\%LOCALAPPDATA%\OneDrive\Documents\llama.cpp Windows Web Manager"
```

You can use that command as the target of a desktop or Startup-folder shortcut.
After launching with `pythonw.exe`, always stop the application from the tray
icon's **Exit** command.

For console-only operation without a tray icon, use:

```powershell
.\.venv\Scripts\python.exe .\app.py --no-tray
```

In console-only mode, `Ctrl+C` performs the same clean shutdown.
Child executables use `CREATE_NO_WINDOW`. Caddy and enabled Cloudflare Tunnel
output is discarded;
llama-server output is kept only in bounded memory for the website console. It
is never printed by `app.py` and is never written to a log file. The three
startup URL announcements are printed only when a console is present.

## Layout

```text
app.py
Caddyfile
.env
frontend\
  index.html
  app.js
  style.css
  favicon.svg
  favicon-dark.svg
models\
  model-name-q4.gguf
  model-name-mtp.gguf
  model-name-mmproj.gguf
```

Files containing `mmproj` are hidden from the model list. A matching projection
file is attached automatically. The mmproj filename is derived from the model
without the quantization token, so a model named
`model-name-q4-mtp.gguf` looks for `model-name-mtp-mmproj.gguf`. Display names
remove both the internal `-mtp` marker and any `-q<number>` quantization token,
while the original filename remains available for applying MTP launch options.
The quantization is shown separately in the model list, in uppercase, after the
file size (for example `model-name-q4-mtp.gguf · 21.11 GB · Q4`).

MTP can come from two sources. When the main GGUF itself contains the MTP head
and no matching draft exists (a name like `model-name-q4-mtp.gguf` on its own),
the manager launches it with `--spec-type draft-mtp --spec-draft-n-max`. When
MTP is separated into its own file, the draft shares the same base name as the
main model but without the quantization token, for example main
`model-name-q4.gguf` paired with `model-name-mtp.gguf`. Because they share the
same base name, the `-mtp` file is recognized as a draft, hidden from the model
list, and the main card launches with `--model-draft model-name-mtp.gguf`
followed by the same `--spec-type` flags. In both cases the model card shows the
`MTP` feature badge.

## Configuration

`app.py` loads these values from `.env`:

- frontend and model directories
- llama-server, Caddy, and Caddyfile paths; the Cloudflare Tunnel path only
  when enabled
- llama.cpp, Flask, and Caddy hosts/ports
- default context size (`LLAMA_DEFAULT_CONTEXT_SIZE`), reasoning, alias, mmproj,
  and MTP options
- maximum in-memory website console lines (`WEB_LOG_LINES`)
- tray enablement and tooltip (`TRAY_ENABLED`, `TRAY_TOOLTIP`)
- Cloudflare Tunnel enablement (`ENABLE_CLOUDFLARE_TUNNEL`), public URL
  (`CLOUDFLARE_TUNNEL_URL`), and secret token (`CLOUDFLARE_TUNNEL_TOKEN`)

Caddy also reads the host and port variables from the environment passed by
`app.py`.

## API

| Method | Route | Purpose |
|---|---|---|
| `GET` | `/api/models` | List loadable models |
| `GET` | `/api/status` | Current model process state |
| `POST` | `/api/start` | Load or reload with `model`, `context_size`, and optional `load_mmproj` |
| `POST` | `/api/stop` | Unload the current model |
| `POST` | `/api/restart` | Restart with optional `context_size` and `load_mmproj` |
| `GET` | `/api/command?model=name.gguf&context_size=16384&load_mmproj=true` | Safe command preview |
| `GET` | `/api/metrics` | CPU, RAM, GPU, and VRAM sample |
| `GET` | `/api/logs` | Recent in-memory llama-server output |
| `POST` | `/api/logs/clear` | Clear the website console |

All process commands use argument lists, `shell=False`, validated model
filenames, and absolute paths.
