#!/usr/bin/env python3
"""dgml-app-sample-server.py — stdlib-only HTTP server for the DGML app sample.

Usage:
    python dgml-app-sample-server.py [--workspace /path/to/workspace] [--port 5173]

Workspace root resolution: --workspace > DGML_HOME env var > ./dgml-workspace
"""

import argparse
import json
import os
import platform
import queue
import re
import shlex
import subprocess
import sys
import threading
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

# ---------------------------------------------------------------------------
# Global workspace state
# ---------------------------------------------------------------------------

_workspace_root: str = ""
_workspace_lock = threading.Lock()


def get_workspace_root() -> str:
    with _workspace_lock:
        return _workspace_root


def set_workspace_root(root: str) -> None:
    global _workspace_root
    with _workspace_lock:
        _workspace_root = root


# ---------------------------------------------------------------------------
# Workspace helpers
# ---------------------------------------------------------------------------


def _read_dir_ids(d: Path) -> list[str]:
    try:
        return sorted(e.name for e in d.iterdir() if e.is_dir() and not e.name.startswith("."))
    except FileNotFoundError:
        return []


def build_index(root: str) -> dict[str, Any]:
    r = Path(root)
    docsets_dir = r / "docsets"
    files_dir = r / "files"
    docset_ids = _read_dir_ids(docsets_dir)
    file_ids = _read_dir_ids(files_dir)

    docsets: list[dict[str, Any]] = []
    for ds_id in docset_ids:
        ds_file_ids = _read_dir_ids(docsets_dir / ds_id / "files")
        ds_name = ds_id
        try:
            meta = json.loads((docsets_dir / ds_id / "docset.json").read_text("utf-8"))
            ds_name = meta.get("name") or ds_id
        except Exception:
            pass
        docsets.append({"id": ds_id, "name": ds_name, "fileIds": ds_file_ids})

    files: list[dict[str, Any]] = []
    for f_id in file_ids:
        meta_path = files_dir / f_id / "file.json"
        try:
            meta = json.loads(meta_path.read_text("utf-8"))
            filename = meta.get("original_filename", f_id)
            if not isinstance(filename, str):
                filename = f_id
            page_count = meta.get("page_count", 0)
            if not isinstance(page_count, int):
                page_count = 0
        except Exception:
            filename, page_count = f_id, 0
        files.append({"id": f_id, "filename": filename, "pageCount": page_count})

    return {"root": root, "docsets": docsets, "files": files}


def _find_xml_file(file_dir: Path) -> Path | None:
    """Locate the generated <stem>.dgml.xml (grounded in place, dg:origin)."""
    if not file_dir.exists():
        return None
    try:
        entries = [e.name for e in file_dir.iterdir() if e.is_file()]
    except Exception:
        return None
    name = next((e for e in entries if e.endswith(".dgml.xml")), None)
    return file_dir / name if name else None


# ---------------------------------------------------------------------------
# Directory picker
# ---------------------------------------------------------------------------


def pick_directory(start_path: str | None = None) -> str | None:
    sys_platform = platform.system()

    if sys_platform == "Windows":
        start_arg = (start_path or "").replace("\\", "\\\\").replace("'", "\\'")
        script = f"""
Add-Type -AssemblyName System.Windows.Forms
[System.Windows.Forms.Application]::EnableVisualStyles()
$owner = New-Object System.Windows.Forms.Form
$owner.TopMost = $true; $owner.ShowInTaskbar = $false
$owner.WindowState = 'Minimized'; $owner.Show()
$d = New-Object System.Windows.Forms.OpenFileDialog
$d.Title = 'Select folder'
$d.Filter = 'Folders|*.none'
$d.CheckFileExists = $false
$d.ValidateNames = $false
$d.FileName = 'Select Folder'
if ('{start_arg}') {{ $d.InitialDirectory = '{start_arg}' }}
if ($d.ShowDialog($owner) -eq 'OK') {{
    [System.IO.Path]::GetDirectoryName($d.FileName)
}} else {{ '' }}
$owner.Dispose()
""".strip()
        try:
            proc = subprocess.run(
                ["powershell", "-STA", "-NoProfile", "-Command", script],
                capture_output=True,
                text=True,
            )
            result = proc.stdout.strip()
            return result or None
        except Exception:
            return None

    elif sys_platform == "Darwin":
        hint = f' default location POSIX file "{start_path}"' if start_path else ""
        try:
            proc = subprocess.run(
                ["osascript", "-e", f'tell app "Finder" to POSIX path of (choose folder{hint})'],
                capture_output=True,
                text=True,
            )
            result = proc.stdout.strip().rstrip("/")
            return result or None
        except Exception:
            return None

    else:
        args = ["zenity", "--file-selection", "--directory"]
        if start_path:
            args += ["--filename", start_path]
        try:
            proc = subprocess.run(args, capture_output=True, text=True)
            result = proc.stdout.strip()
            return result or None
        except Exception:
            return None


# ---------------------------------------------------------------------------
# Pipeline state
# ---------------------------------------------------------------------------

_pipeline_running = False
_pipeline_current_proc: subprocess.Popen[bytes] | None = None
_pipeline_event_buffer: list[dict[str, Any]] = []
_pipeline_clients: list["queue.Queue[dict[str, Any] | None]"] = []
_pipeline_lock = threading.Lock()


def _broadcast(event: dict[str, Any]) -> None:
    with _pipeline_lock:
        _pipeline_event_buffer.append(event)
        dead = []
        for q in _pipeline_clients:
            try:
                q.put_nowait(event)
            except queue.Full:
                dead.append(q)
        for q in dead:
            _pipeline_clients.remove(q)


def _load_config_json(path: Path) -> dict[str, Any]:
    """Parse a workspace ``config.json`` that may carry full-line ``//`` comments
    (it is seeded verbatim from ``local_config.json``). Returns ``{}`` on any
    read/parse error so a fresh classification section can still be written."""
    try:
        raw = path.read_text("utf-8")
    except OSError:
        return {}
    raw = "\n".join("" if ln.lstrip().startswith("//") else ln for ln in raw.split("\n"))
    try:
        parsed = json.loads(raw)
    except Exception:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _api_key_env(model: str, api_key: str) -> dict[str, str]:
    if not api_key:
        return {}
    lower = model.lower()
    if lower.startswith("claude") or lower.startswith("anthropic/"):
        return {"ANTHROPIC_API_KEY": api_key}
    if (
        lower.startswith("openai/")
        or lower.startswith("gpt")
        or lower.startswith("o1")
        or lower.startswith("o3")
    ):
        return {"OPENAI_API_KEY": api_key}
    if lower.startswith("gemini/") or lower.startswith("google/"):
        return {"GEMINI_API_KEY": api_key}
    return {}


def _dgml_cmd() -> list[str]:
    override = os.environ.get("DGML_CMD", "")
    if override:
        return shlex.split(override)
    return ["uv", "run", "dgml"]


def _run_dgml(args: list[str], env: dict[str, str]) -> str:
    global _pipeline_current_proc
    cmd_base = _dgml_cmd()
    full_args = cmd_base + args
    _broadcast({"type": "log", "text": f"$ {' '.join(full_args)}"})

    proc = subprocess.Popen(
        full_args,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        stdin=subprocess.DEVNULL,
    )
    with _pipeline_lock:
        _pipeline_current_proc = proc

    stdout_lines: list[str] = []

    def _read(stream: Any, capture: bool) -> None:
        for raw in stream:
            line = raw.decode("utf-8", errors="replace").rstrip("\n")
            if capture:
                stdout_lines.append(line)
            if line:
                _broadcast({"type": "log", "text": line})
        stream.close()

    t_out = threading.Thread(target=_read, args=(proc.stdout, True), daemon=True)
    t_err = threading.Thread(target=_read, args=(proc.stderr, False), daemon=True)
    t_out.start()
    t_err.start()
    t_out.join()
    t_err.join()

    code = proc.wait()
    with _pipeline_lock:
        _pipeline_current_proc = None

    if code != 0:
        # Decode common Windows crash codes for a friendlier message.
        win_codes = {0xC0000005: "access violation (native crash — possible PyTorch/CUDA issue)"}
        hint = win_codes.get(code & 0xFFFFFFFF, "")
        detail = f" ({hint})" if hint else ""
        raise RuntimeError(f"dgml {args[0] if args else ''} exited with code {code}{detail}")
    return "\n".join(stdout_lines)


def _assign_unclustered_files(ws: list[str], env: dict[str, str], abs_workspace: str) -> None:
    """Give every file clustering left unassigned its own single-file DocSet.

    `docset generate` only processes DocSet members, so a file clustering
    skipped — because `dgml cluster` failed outright (missing extra/config)
    or because the file landed in `failed_file_ids` — would otherwise never
    get DGML output. `docset create`/`add-file` make no LLM call, so this
    fallback works even without a classification config or API key.
    """
    index = build_index(abs_workspace)
    assigned_ids = {fid for ds in index["docsets"] for fid in ds["fileIds"]}
    for f in index["files"]:
        if f["id"] in assigned_ids:
            continue
        create_out = _run_dgml([*ws, "docset", "create", "--name", str(f["filename"])], env)
        ds_id = json.loads(create_out)["id"]
        _run_dgml([*ws, "docset", "add-file", f["id"], "--docset", ds_id], env)


def _generate_docsets(ws: list[str], env: dict[str, str]) -> None:
    list_out = _run_dgml([*ws, "docset", "list"], env)
    docsets: list[dict[str, Any]] = []
    try:
        parsed = json.loads(list_out)
        if isinstance(parsed, list):
            docsets = parsed
        elif isinstance(parsed, dict):
            docsets = parsed.get("docsets", [])
    except Exception:
        _broadcast({"type": "log", "text": "Warning: could not parse docset list"})

    for ds in docsets:
        ds_id = str(ds.get("id", ""))
        ds_label = str(ds.get("name") or ds_id)
        _broadcast({"type": "step", "name": f"generate:{ds_label}", "status": "running"})
        _run_dgml([*ws, "docset", "generate", ds_id], env)
        _broadcast({"type": "step", "name": f"generate:{ds_label}", "status": "done"})


def _run_pipeline(body: dict[str, Any]) -> None:
    global _pipeline_running
    source_dir = str(body.get("sourceDir", ""))
    workspace_dir = str(body.get("workspaceDir", ""))
    model = str(body.get("model", ""))
    api_key = str(body.get("apiKey", ""))

    env: dict[str, str] = {**os.environ, **_api_key_env(model, api_key)}  # type: ignore[dict-item]
    abs_workspace = str(Path(workspace_dir).resolve())
    abs_source = str(Path(source_dir).resolve())
    ws = ["--workspace", abs_workspace, "--verbose"]

    try:
        _broadcast({"type": "step", "name": "init", "status": "running"})
        # `dgml init` seeds the shared local_config.json (idempotent); `workspace
        # create` builds the workspace and copies that config into it. The
        # required --organization is embedded in docset namespace URIs.
        _run_dgml([*ws, "init"], env)
        _run_dgml([*ws, "workspace", "create", "--organization", "dgml-app-sample"], env)
        _broadcast({"type": "step", "name": "init", "status": "done"})

        _broadcast({"type": "step", "name": "add", "status": "running"})
        _run_dgml([*ws, "file", "add", abs_source, "--recursive", "--on-conflict", "skip"], env)
        _broadcast({"type": "step", "name": "add", "status": "done"})

        if model and api_key:
            cfg_path = Path(abs_workspace) / "config.json"
            existing = _load_config_json(cfg_path)
            existing["classification"] = {"model": model, "api_key": api_key}
            # `docset generate` reads its models from the 'generation' section
            # (there are no --model flags); both model and label_model are
            # required, so name both. The sample has one model input, so it
            # doubles as the labeling model.
            existing["generation"] = {
                "model": model,
                "label_model": model,
                "api_key": api_key,
            }
            cfg_path.write_text(json.dumps(existing, indent=2), "utf-8")

        _broadcast({"type": "step", "name": "cluster", "status": "running"})
        try:
            _run_dgml([*ws, "cluster"], env)
            _broadcast({"type": "step", "name": "cluster", "status": "done"})
        except Exception as cluster_exc:
            _broadcast(
                {
                    "type": "log",
                    "text": (
                        f"Warning: clustering failed ({cluster_exc}). "
                        "Continuing without automatic docset grouping."
                    ),
                }
            )
            _broadcast({"type": "step", "name": "cluster", "status": "error"})

        _broadcast({"type": "step", "name": "assign_stragglers", "status": "running"})
        _assign_unclustered_files(ws, env, abs_workspace)
        _broadcast({"type": "step", "name": "assign_stragglers", "status": "done"})

        _generate_docsets(ws, env)

        set_workspace_root(abs_workspace)
        _broadcast({"type": "done", "workspaceDir": abs_workspace})

    except Exception as exc:
        _broadcast({"type": "error", "message": str(exc)})
    finally:
        with _pipeline_lock:
            _pipeline_running = False


def _run_resume(body: dict[str, Any]) -> None:
    global _pipeline_running
    workspace_dir = str(body.get("workspaceDir", ""))
    model = str(body.get("model", ""))
    api_key = str(body.get("apiKey", ""))

    env: dict[str, str] = {**os.environ, **_api_key_env(model, api_key)}  # type: ignore[dict-item]
    abs_workspace = str(Path(workspace_dir).resolve())
    ws = ["--workspace", abs_workspace, "--verbose"]

    try:
        if model and api_key:
            cfg_path = Path(abs_workspace) / "config.json"
            existing = _load_config_json(cfg_path)
            existing["classification"] = {"model": model, "api_key": api_key}
            # `docset generate` reads its models from the 'generation' section
            # (there are no --model flags); both model and label_model are
            # required, so name both. The sample has one model input, so it
            # doubles as the labeling model.
            existing["generation"] = {
                "model": model,
                "label_model": model,
                "api_key": api_key,
            }
            cfg_path.write_text(json.dumps(existing, indent=2), "utf-8")

        _broadcast({"type": "step", "name": "cluster", "status": "running"})
        try:
            _run_dgml([*ws, "cluster", "--skip-existing"], env)
            _broadcast({"type": "step", "name": "cluster", "status": "done"})
        except Exception as cluster_exc:
            _broadcast(
                {
                    "type": "log",
                    "text": (
                        f"Warning: clustering failed ({cluster_exc}). "
                        "Continuing without automatic docset grouping."
                    ),
                }
            )
            _broadcast({"type": "step", "name": "cluster", "status": "error"})

        _broadcast({"type": "step", "name": "assign_stragglers", "status": "running"})
        _assign_unclustered_files(ws, env, abs_workspace)
        _broadcast({"type": "step", "name": "assign_stragglers", "status": "done"})

        _generate_docsets(ws, env)

        set_workspace_root(abs_workspace)
        _broadcast({"type": "done", "workspaceDir": abs_workspace})

    except Exception as exc:
        _broadcast({"type": "error", "message": str(exc)})
    finally:
        with _pipeline_lock:
            _pipeline_running = False


# ---------------------------------------------------------------------------
# Chain / wallet / registry / stake / prove
#
# Unlike the pipeline (long-running, streamed via SSE), these are one-shot
# request/response calls: run `dgml`, parse its JSON stdout, hand it back.
# Every handler returns {"ok": true, "data": ...} or {"ok": false, "error": ...}
# so the client can branch on `ok` the same way regardless of which call it made.
# ---------------------------------------------------------------------------

_NVNM_ANCHOR_ADDRESS = "0x0000000000000000000000000000000000000A00"


class _DgmlCallError(Exception):
    pass


def _run_dgml_json(args: list[str]) -> Any:
    """Run `dgml <args>` synchronously and parse its JSON stdout.

    Raises `_DgmlCallError` with a human-readable message (the CLI's own
    error envelope message when present) on a non-zero exit or unparsable
    output. Never streamed/broadcast — these are single request/response
    calls, not pipeline steps.
    """
    proc = subprocess.run(
        [*_dgml_cmd(), "--workspace", get_workspace_root(), *args],
        capture_output=True,
        text=True,
        timeout=180,
    )
    if proc.returncode != 0:
        message = proc.stderr.strip() or proc.stdout.strip() or f"exited {proc.returncode}"
        try:
            envelope = json.loads(proc.stdout or proc.stderr)
            if isinstance(envelope, dict) and isinstance(envelope.get("error"), dict):
                message = envelope["error"].get("message", message)
        except (json.JSONDecodeError, ValueError):
            pass
        raise _DgmlCallError(message)
    try:
        return json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        raise _DgmlCallError(f"could not parse dgml output: {exc}") from exc


def _rpc_call(rpc_url: str, method: str, params: list[Any]) -> Any:
    """A minimal JSON-RPC POST, stdlib-only (no web3 dependency in this server)."""
    body = json.dumps({"jsonrpc": "2.0", "id": 1, "method": method, "params": params}).encode()
    req = urllib.request.Request(
        rpc_url, data=body, headers={"Content-Type": "application/json"}, method="POST"
    )
    with urllib.request.urlopen(req, timeout=15) as resp:
        payload = json.loads(resp.read())
    if isinstance(payload, dict) and payload.get("error"):
        raise _DgmlCallError(str(payload["error"].get("message", payload["error"])))
    return payload["result"]


def _verify_chain_connection(rpc_url: str, chain_id: int, anchor_address: str) -> None:
    """Confirm the RPC is reachable, reports the expected chain id, and has
    *something* deployed at the anchor address. Raises `_DgmlCallError` with a
    plain-English reason on any failure. This does not verify the deployed
    contract is a compatible anchor implementation — only that one exists."""
    try:
        reported_hex = _rpc_call(rpc_url, "eth_chainId", [])
        reported = int(reported_hex, 16)
    except Exception as exc:
        raise _DgmlCallError(f"could not reach {rpc_url}: {exc}") from exc
    if reported != chain_id:
        raise _DgmlCallError(f"RPC reports chain id {reported}, expected {chain_id}")
    try:
        code = _rpc_call(rpc_url, "eth_getCode", [anchor_address, "latest"])
    except Exception as exc:
        raise _DgmlCallError(f"could not read contract code: {exc}") from exc
    if not code or code == "0x":
        raise _DgmlCallError(f"no contract deployed at {anchor_address} on this chain")


def _keyring_cmd() -> list[str]:
    """Same venv invocation as `_dgml_cmd()`, pointed at `keyring` instead."""
    override = os.environ.get("DGML_CMD", "")
    if override:
        parts = shlex.split(override)
        if parts and parts[-1].endswith("dgml"):
            return [*parts[:-1], "keyring"]
    return ["uv", "run", "keyring"]


def _keyring_set_key(private_key: str) -> None:
    """Store the wallet private key via the `keyring` CLI, piped over stdin
    only — never argv, never logged. `keyring set` reads stdin directly when
    it isn't a TTY (see keyring.cli.CommandLineTool.pass_from_pipe)."""
    proc = subprocess.run(
        [*_keyring_cmd(), "set", "nvnm-wallet", "default"],
        input=private_key,
        capture_output=True,
        text=True,
        timeout=30,
    )
    if proc.returncode != 0:
        # Deliberately generic: never forward stderr, in case some keyring
        # backend ever echoes input back on failure.
        raise _DgmlCallError("failed to store the key in the OS keyring")


def _keyring_remove_key() -> None:
    proc = subprocess.run(
        [*_keyring_cmd(), "del", "nvnm-wallet", "default"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    if proc.returncode != 0:
        raise _DgmlCallError("failed to remove the key from the OS keyring (it may not be set)")


# ---------------------------------------------------------------------------
# LLM proxy
# ---------------------------------------------------------------------------


def _proxy_llm(
    api_url: str,
    api_key: str,
    body_bytes: bytes,
    extra_headers: dict[str, str],
) -> tuple[int, bytes, str]:
    """Forward an LLM request to the provider. Returns (status, body, content-type)."""
    req_headers: dict[str, str] = {"Content-Type": "application/json", **extra_headers}
    target_url = api_url

    lower_url = api_url.lower()
    if "openai.com" in lower_url:
        req_headers["Authorization"] = f"Bearer {api_key}"
    elif "googleapis.com" in lower_url:
        sep = "&" if "?" in target_url else "?"
        target_url = f"{target_url}{sep}key={api_key}"
    else:
        # Anthropic-compatible
        req_headers["x-api-key"] = api_key
        req_headers.setdefault("anthropic-version", "2023-06-01")

    req = urllib.request.Request(target_url, data=body_bytes, headers=req_headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            return resp.status, resp.read(), resp.headers.get("Content-Type", "application/json")
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read(), "application/json"
    except Exception as exc:
        err = json.dumps({"error": {"message": str(exc)}}).encode()
        return 502, err, "application/json"


# ---------------------------------------------------------------------------
# Static content types
# ---------------------------------------------------------------------------

_CONTENT_TYPES: dict[str, str] = {
    ".json": "application/json; charset=utf-8",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".pdf": "application/pdf",
    ".xml": "application/xml; charset=utf-8",
    ".jsonl": "application/x-ndjson; charset=utf-8",
}

HTML_PATH = Path(__file__).parent / "dgml-app-sample.html"

# ---------------------------------------------------------------------------
# HTTP handler
# ---------------------------------------------------------------------------


class _Handler(BaseHTTPRequestHandler):
    server_version = "dgml-app-sample"

    def log_message(self, fmt: str, *args: Any) -> None:  # silence default logger
        pass

    # -- helpers --

    def _cors(self) -> None:
        self.send_header("Access-Control-Allow-Origin", "*")

    def _send_json(self, data: Any, status: int = 200) -> None:
        body = json.dumps(data).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self._cors()
        self.end_headers()
        self.wfile.write(body)

    def _send_404(self) -> None:
        self.send_response(404)
        self.send_header("Content-Type", "text/plain")
        self.end_headers()
        self.wfile.write(b"not found")

    def _read_body(self) -> bytes:
        length = int(self.headers.get("Content-Length", 0) or 0)
        return self.rfile.read(length) if length > 0 else b""

    # -- routing --

    def do_OPTIONS(self) -> None:
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "*")
        self.send_header("Access-Control-Max-Age", "86400")
        self.end_headers()

    def do_GET(self) -> None:
        path = urlparse(self.path).path

        if path in ("/", ""):
            self._serve_html()
        elif path == "/workspace/_index":
            self._handle_workspace_index()
        elif path.startswith("/workspace/"):
            self._handle_workspace_static(path)
        elif path == "/pipeline/events":
            self._handle_pipeline_events()
        else:
            self._send_404()

    def do_POST(self) -> None:
        path = urlparse(self.path).path

        dispatch: dict[str, Any] = {
            "/workspace/switch": self._handle_workspace_switch,
            "/api/pick-dir": self._handle_pick_dir,
            "/pipeline/start": self._handle_pipeline_start,
            "/pipeline/resume": self._handle_pipeline_resume,
            "/pipeline/cancel": self._handle_pipeline_cancel,
            "/pipeline/workspace": self._handle_pipeline_workspace,
            "/proxy/v1/messages": self._handle_proxy,
            "/chain/list": self._handle_chain_list,
            "/chain/connect": self._handle_chain_connect,
            "/registry/list": self._handle_registry_list,
            "/registry/create": self._handle_registry_create,
            "/wallet/status": self._handle_wallet_status,
            "/wallet/set-key": self._handle_wallet_set_key,
            "/wallet/remove-key": self._handle_wallet_remove_key,
            "/node/resolve": self._handle_node_resolve,
            "/stake/node": self._handle_stake_node,
            "/stake/file": self._handle_stake_file,
            "/prove/node": self._handle_prove_node,
            "/prove/file": self._handle_prove_file,
        }
        handler = dispatch.get(path)
        if handler:
            handler()
        else:
            self._send_404()

    # -- GET handlers --

    def _serve_html(self) -> None:
        try:
            content = HTML_PATH.read_bytes()
        except FileNotFoundError:
            self.send_response(404)
            self.end_headers()
            self.wfile.write(b"HTML file not found alongside server script")
            return
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(content)))
        # This file changes across dev iterations; without this, some browsers'
        # heuristic caching can serve a stale copy after a restart/reload.
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(content)

    def _handle_workspace_index(self) -> None:
        root = get_workspace_root()
        if not root:
            self._send_json({"root": "", "docsets": [], "files": []})
            return
        try:
            self._send_json(build_index(root))
        except Exception as exc:
            self._send_json({"error": str(exc)}, 500)

    def _handle_workspace_static(self, path: str) -> None:
        root = get_workspace_root()
        if not root:
            self._send_404()
            return

        rel = path[len("/workspace/") :]  # strip leading /workspace/

        # /workspace/docsets/:docsetId/files/:fileId/_dgml
        m = re.match(r"^docsets/([^/]+)/files/([^/]+)/_dgml$", rel)
        if m:
            docset_id, file_id = m.group(1), m.group(2)
            file_dir = Path(root) / "docsets" / docset_id / "files" / file_id
            xml_path = _find_xml_file(file_dir)
            if not xml_path:
                self._send_404()
                return
            try:
                xml = xml_path.read_text("utf-8")
                self._send_json({"xml": xml})
            except Exception:
                self._send_404()
            return

        # Static file pass-through
        abs_path = (Path(root) / unquote(rel)).resolve()
        root_resolved = Path(root).resolve()
        if abs_path != root_resolved and not str(abs_path).startswith(str(root_resolved) + os.sep):
            self.send_response(403)
            self.end_headers()
            self.wfile.write(b"forbidden")
            return

        if not abs_path.is_file():
            self._send_404()
            return

        ext = abs_path.suffix.lower()
        content_type = _CONTENT_TYPES.get(ext, "application/octet-stream")
        size = abs_path.stat().st_size
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(size))
        self._cors()
        self.end_headers()
        with abs_path.open("rb") as f:
            while True:
                chunk = f.read(65536)
                if not chunk:
                    break
                self.wfile.write(chunk)

    def _handle_pipeline_events(self) -> None:
        global _pipeline_clients
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self._cors()
        self.end_headers()

        client_q: queue.Queue[dict[str, Any] | None] = queue.Queue(maxsize=500)

        # Replay buffered events so late-connecting clients catch up
        with _pipeline_lock:
            history = list(_pipeline_event_buffer)
        for event in history:
            try:
                self.wfile.write(f"data: {json.dumps(event)}\n\n".encode())
            except Exception:
                return
        try:
            self.wfile.flush()
        except Exception:
            return

        with _pipeline_lock:
            _pipeline_clients.append(client_q)

        try:
            while True:
                try:
                    event = client_q.get(timeout=15)
                    if event is None:
                        break
                    self.wfile.write(f"data: {json.dumps(event)}\n\n".encode())
                    self.wfile.flush()
                except queue.Empty:
                    self.wfile.write(b": keepalive\n\n")
                    self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass
        finally:
            with _pipeline_lock:
                if client_q in _pipeline_clients:
                    _pipeline_clients.remove(client_q)

    # -- POST handlers --

    def _handle_workspace_switch(self) -> None:
        body = self._read_body()
        try:
            data = json.loads(body)
            root = str(Path(data["root"]).resolve())
            set_workspace_root(root)
            self._send_json({"root": root})
        except Exception as exc:
            self._send_json({"error": str(exc)}, 400)

    def _handle_pick_dir(self) -> None:
        body = self._read_body()
        start_path: str | None = None
        try:
            start_path = json.loads(body).get("startPath")
        except Exception:
            pass
        result = pick_directory(start_path)
        self._send_json({"path": result})

    def _handle_pipeline_start(self) -> None:
        global _pipeline_running, _pipeline_event_buffer
        body = self._read_body()
        with _pipeline_lock:
            if _pipeline_running:
                self._send_json({"error": "Pipeline already running"}, 409)
                return
            _pipeline_running = True
            _pipeline_event_buffer = []
        try:
            data = json.loads(body)
        except Exception:
            with _pipeline_lock:
                _pipeline_running = False
            self._send_json({"error": "Invalid body"}, 400)
            return
        threading.Thread(target=_run_pipeline, args=(data,), daemon=True).start()
        self._send_json({"started": True})

    def _handle_pipeline_resume(self) -> None:
        global _pipeline_running, _pipeline_event_buffer
        body = self._read_body()
        with _pipeline_lock:
            if _pipeline_running:
                self._send_json({"error": "Pipeline already running"}, 409)
                return
            _pipeline_running = True
            _pipeline_event_buffer = []
        try:
            data = json.loads(body)
        except Exception:
            with _pipeline_lock:
                _pipeline_running = False
            self._send_json({"error": "Invalid body"}, 400)
            return
        threading.Thread(target=_run_resume, args=(data,), daemon=True).start()
        self._send_json({"started": True})

    def _handle_pipeline_cancel(self) -> None:
        global _pipeline_running, _pipeline_current_proc
        with _pipeline_lock:
            proc = _pipeline_current_proc
            _pipeline_running = False
            _pipeline_current_proc = None
        if proc:
            try:
                proc.terminate()
            except Exception:
                pass
        _broadcast({"type": "cancelled"})
        self._send_json({"cancelled": True})

    def _handle_pipeline_workspace(self) -> None:
        body = self._read_body()
        try:
            data = json.loads(body)
            root = str(Path(data["dir"]).resolve())
            set_workspace_root(root)
            self._send_json({"root": root})
        except Exception as exc:
            self._send_json({"error": str(exc)}, 400)

    # -- chain / wallet / registry / stake / prove --

    def _json_body(self) -> dict[str, Any]:
        raw = self._read_body()
        if not raw:
            return {}
        data = json.loads(raw)
        return data if isinstance(data, dict) else {}

    def _ok(self, data: Any) -> None:
        self._send_json({"ok": True, "data": data})

    def _err(self, error: str) -> None:
        self._send_json({"ok": False, "error": error})

    def _handle_chain_list(self) -> None:
        try:
            out = _run_dgml_json(["chain", "list"])
            self._ok(out)
        except _DgmlCallError as exc:
            self._err(str(exc))

    def _handle_chain_connect(self) -> None:
        try:
            body = self._json_body()
            name = str(body["name"])
            rpc_url = str(body["rpcUrl"])
            chain_id = int(body["chainId"])
            anchor_address = str(body.get("anchorAddress") or _NVNM_ANCHOR_ADDRESS)
        except (KeyError, ValueError, TypeError) as exc:
            self._err(f"invalid request: {exc}")
            return
        try:
            _verify_chain_connection(rpc_url, chain_id, anchor_address)
            args = [
                "chain",
                "add",
                "--name",
                name,
                "--rpc-url",
                rpc_url,
                "--chain-id",
                str(chain_id),
            ]
            if body.get("anchorAddress"):
                args += ["--anchor-address", str(body["anchorAddress"])]
            if body.get("explorer"):
                args += ["--explorer", str(body["explorer"])]
            if body.get("nativeToken"):
                args += ["--native-token", str(body["nativeToken"])]
            out = _run_dgml_json(args)
            self._ok(out)
        except _DgmlCallError as exc:
            self._err(str(exc))

    def _handle_registry_list(self) -> None:
        try:
            body = self._json_body()
            chain = str(body.get("chain", "nvnm-testnet"))
            args = ["registry", "list", "--chain", chain]
            if body.get("name"):
                args += ["--name", str(body["name"])]
            self._ok(_run_dgml_json(args))
        except _DgmlCallError as exc:
            self._err(str(exc))

    def _handle_registry_create(self) -> None:
        try:
            body = self._json_body()
            chain = str(body.get("chain", "nvnm-testnet"))
            name = str(body["name"])
            args = ["registry", "create", "--chain", chain, "--name", name]
            if body.get("description"):
                args += ["--description", str(body["description"])]
            if body.get("metadata"):
                args += ["--metadata", str(body["metadata"])]
            if body.get("dryRun"):
                args += ["--dry-run"]
            self._ok(_run_dgml_json(args))
        except (KeyError, _DgmlCallError) as exc:
            self._err(str(exc))

    def _handle_wallet_status(self) -> None:
        try:
            body = self._json_body()
            chain = str(body.get("chain", "nvnm-testnet"))
            args = ["wallet", "status", "--chain", chain]
            if body.get("address"):
                args += ["--address", str(body["address"])]
            self._ok(_run_dgml_json(args))
        except _DgmlCallError as exc:
            self._err(str(exc))

    def _handle_wallet_set_key(self) -> None:
        try:
            body = self._json_body()
            private_key = str(body["privateKey"]).strip()
            if not private_key:
                self._err("private key must not be empty")
                return
            _keyring_set_key(private_key)
            self._ok({"stored": True})
        except (KeyError, _DgmlCallError) as exc:
            self._err(str(exc))

    def _handle_wallet_remove_key(self) -> None:
        try:
            _keyring_remove_key()
            self._ok({"removed": True})
        except _DgmlCallError as exc:
            self._err(str(exc))

    def _handle_node_resolve(self) -> None:
        try:
            body = self._json_body()
            file_id = str(body["fileId"])
            docset_id = str(body["docsetId"])
            child_path = str(body.get("childPath", ""))
            args = [
                "node",
                "export",
                file_id,
                "--docset",
                docset_id,
                "--child-path",
                child_path,
            ]
            self._ok(_run_dgml_json(args))
        except (KeyError, _DgmlCallError) as exc:
            self._err(str(exc))

    def _handle_stake_node(self) -> None:
        try:
            body = self._json_body()
            file_id = str(body["fileId"])
            docset_id = str(body["docsetId"])
            xpath = str(body["xpath"])
            chain = str(body.get("chain", "nvnm-testnet"))
            registry = str(body["registry"])
            args = [
                "stake",
                "node",
                file_id,
                "--docset",
                docset_id,
                "--xpath",
                xpath,
                "--chain",
                chain,
                "--registry",
                registry,
            ]
            if body.get("dryRun"):
                args += ["--dry-run"]
            self._ok(_run_dgml_json(args))
        except (KeyError, _DgmlCallError) as exc:
            self._err(str(exc))

    def _handle_stake_file(self) -> None:
        try:
            body = self._json_body()
            file_id = str(body["fileId"])
            docset_id = body.get("docsetId")
            chain = str(body.get("chain", "nvnm-testnet"))
            registry = str(body["registry"])
            args = ["stake", "file", file_id, "--chain", chain, "--registry", registry]
            if docset_id:
                args += ["--docset", str(docset_id)]
            if body.get("dryRun"):
                args += ["--dry-run"]
            self._ok(_run_dgml_json(args))
        except (KeyError, _DgmlCallError) as exc:
            self._err(str(exc))

    def _handle_prove_node(self) -> None:
        self._handle_prove("node")

    def _handle_prove_file(self) -> None:
        self._handle_prove("file")

    def _handle_prove(self, kind: str) -> None:
        try:
            body = self._json_body()
            chain = str(body.get("chain", "nvnm-testnet"))
            args = ["prove", kind, "--chain", chain]
            if body.get("registry"):
                args += ["--registry", str(body["registry"])]
            if body.get("checksum"):
                args += ["--checksum", str(body["checksum"])]
            if body.get("recordJson"):
                args += ["--record-json", str(body["recordJson"])]
            self._ok(_run_dgml_json(args))
        except _DgmlCallError as exc:
            self._err(str(exc))

    def _handle_proxy(self) -> None:
        api_url = self.headers.get("x-dgml-api-url", "https://api.anthropic.com/v1/messages")
        api_key = self.headers.get("x-dgml-api-key", "")

        # Translate x-dgml-extra-* headers → real header names
        extra: dict[str, str] = {}
        for k, v in self.headers.items():
            lower_k = k.lower()
            if lower_k.startswith("x-dgml-extra-"):
                real_k = k[len("x-dgml-extra-") :]
                extra[real_k] = v

        body_bytes = self._read_body()
        status, resp_body, content_type = _proxy_llm(api_url, api_key, body_bytes, extra)

        self.send_response(status)
        self.send_header("Content-Type", content_type or "application/json")
        self.send_header("Content-Length", str(len(resp_body)))
        self._cors()
        self.end_headers()
        self.wfile.write(resp_body)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    global _workspace_root

    parser = argparse.ArgumentParser(description="dgml-app-sample HTTP server (port 5173)")
    parser.add_argument("--workspace", help="Workspace root directory")
    parser.add_argument("--port", type=int, default=5173, help="Port (default: 5173)")
    args = parser.parse_args()

    raw_root = args.workspace or os.environ.get("DGML_HOME", "") or "dgml-workspace"
    _workspace_root = str(Path(raw_root).resolve()) if raw_root else ""

    class _QuietServer(ThreadingHTTPServer):
        def handle_error(self, request: object, client_address: object) -> None:
            import errno as _errno
            import traceback

            exc = sys.exc_info()[1]
            # Suppress harmless connection-abort noise (common on Windows).
            if isinstance(exc, (ConnectionAbortedError, ConnectionResetError, BrokenPipeError)):
                return
            if isinstance(exc, OSError) and exc.errno in (_errno.ECONNRESET, _errno.EPIPE):
                return
            traceback.print_exc()

    server = _QuietServer(("", args.port), _Handler)
    print(f"dgml-app-sample listening on http://localhost:{args.port}")
    if _workspace_root:
        print(f"Workspace: {_workspace_root}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
        server.shutdown()


if __name__ == "__main__":
    main()
