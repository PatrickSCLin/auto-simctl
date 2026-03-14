"""
auto-simctl MCP Server
======================
Exposes the orchestrator as MCP tools so any MCP-compatible vibe coding tool
(Cursor, Claude Desktop, …) can drive iOS/Android simulators directly.

Vibe coding loop
----------------
1. get_screen_state()                       ← see what's on screen right now
2. act("tap Watch app")                     ← tap something (one-shot, no HOME reset)
3. get_screen_state()                       ← observe the result
4. act("scroll down") / act("back")         ← gesture (no Qwen needed)
5. act("tap address bar")                   ← tap a text field  → keyboard opens
6. act("input_text https://example.com")    ← keyboard open → types directly
                                              (Enter is NOT pressed automatically)
7. act("press enter")                       ← submit / navigate (if needed)
8. get_screen_state()                       ← confirm
9. run_task("…complex multi-step goal…")    ← hand off to full AI agent

One-shot action rules (act)
----------------------------
- tap / swipe / scroll / pan  → executes once, returns immediately (no verification)
- input_text X                → if keyboard is already open: types X directly (no Qwen)
                                 if keyboard is NOT open: Qwen taps the right field first,
                                 then types X in the same call
- type X on <field>           → compound: tap field + type X (one act call)
- gesture keywords (swipe right, scroll down, back, …) → fast-path, Qwen not needed at all

Tools
-----
  start_servers()               Start Qwen (:8080) + UI-UG (:8081) inference servers
  stop_servers()                Stop both inference servers (free GPU/memory)
  list_devices()                List all booted simulators / connected devices
  get_screen_state()            Current screen: JSON summary + screenshot image
  act(task)                     ONE atomic action from current state (no reset)

Typical session
---------------
  1. start_servers()            ← start AI inference servers (once per session)
  2. list_devices()             ← find your device UDID
  3. get_screen_state()         ← see current screen
  4. act("tap Settings")        ← take one action
  5. get_screen_state()         ← observe result, decide next action
  6. act("scroll down")         ← take next action
     … repeat 4-5 until goal reached …
  7. stop_servers()             ← free GPU/memory when done

Setup (Cursor / Claude Desktop)
--------------------------------
Add to your MCP config:

  {
    "mcpServers": {
      "auto-simctl": {
        "command": "auto-simctl-mcp"
      }
    }
  }

  Or if installed from source:
  {
    "mcpServers": {
      "auto-simctl": {
        "command": "python3",
        "args": ["/path/to/auto-simctl/mcp_server/server.py"]
      }
    }
  }
"""
from __future__ import annotations

import json
import os
import sys
import time
from typing import Any

# Make the project root importable when running directly (python mcp_server/server.py)
# or via the auto-simctl-mcp entry point.  When installed as a package the path
# insert is a no-op because the packages are already on sys.path.
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

try:
    from fastmcp import FastMCP
    from fastmcp.utilities.types import Image
except ImportError:
    print(
        "fastmcp not installed. Run: pip install 'fastmcp>=2.0'",
        file=sys.stderr,
    )
    sys.exit(1)

from logger import get_logger
from mdb.bridge import DeviceBridge

log = get_logger("mcp_server")

mcp = FastMCP(
    "auto-simctl",
    instructions=(
        "auto-simctl controls iOS Simulators and Android devices from natural language. "
        "IMPORTANT: Call start_servers() first to start the AI inference servers before "
        "using act() with non-gesture commands. Gesture commands "
        "(swipe, scroll, back) work without servers. "
        "The caller (you) drives the decision loop — use get_screen_state() to observe, "
        "then act() to execute one atomic action, then get_screen_state() again. "
        "Typical session: start_servers() → list_devices() → get_screen_state() → "
        "act() → get_screen_state() → act() → … → stop_servers()."
    ),
)

# ── Shared lazy singletons ────────────────────────────────────────────────────
_bridge: DeviceBridge | None = None


def _get_bridge() -> DeviceBridge:
    global _bridge
    if _bridge is None:
        _bridge = DeviceBridge()
    return _bridge


def _resolve_device(device_udid: str) -> tuple[str, str]:
    """Return (udid, error_json_or_empty).  error is non-empty if not found."""
    bridge = _get_bridge()
    if device_udid == "auto":
        dev = bridge.first_device()
        if dev is None:
            return "", json.dumps({
                "error": "No devices available. "
                         "Start an iOS Simulator or connect an Android device.",
            })
        return dev.udid, ""
    return device_udid, ""


def _get_orchestrator():
    """Lazily build the Orchestrator (imports heavy deps only when needed)."""
    from agents.qwen_agent import QwenAgent
    from agents.ui_agent import UIAgent
    from orchestrator.loop import Orchestrator

    bridge = _get_bridge()
    qwen = QwenAgent()
    ui = UIAgent()
    return Orchestrator(mdb=bridge, qwen=qwen, ui_agent=ui, max_steps=5)


def _check_servers() -> str | None:
    """Return an error message if required inference servers are not running."""
    from agents.qwen_agent import QwenAgent
    from agents.ui_agent import UIAgent

    missing = []
    if not QwenAgent().server_running():
        missing.append("Qwen reasoning server (port 8080)")
    if not UIAgent().server_running():
        missing.append("UI-UG vision server (port 8081)")
    if missing:
        return (
            "Required inference servers are not running: "
            + ", ".join(missing)
            + ". Call the start_servers() MCP tool to start them, then retry."
        )
    return None


# ── Server lifecycle helpers ──────────────────────────────────────────────────

# All runtime files are stored in ~/.auto-simctl/ so every MCP instance
# (Cursor spawns 3+) shares the same PID / lock / log files.
_RUNTIME_DIR   = os.path.join(os.path.expanduser("~"), ".auto-simctl")
os.makedirs(_RUNTIME_DIR, exist_ok=True)

_QWEN_PID_FILE  = os.path.join(_RUNTIME_DIR, "qwen_server.pid")
_UIUG_PID_FILE  = os.path.join(_RUNTIME_DIR, "uiug_server.pid")
_QWEN_LOG_FILE  = os.path.join(_RUNTIME_DIR, "qwen_server.log")
_UIUG_LOG_FILE  = os.path.join(_RUNTIME_DIR, "uiug_server.log")
_START_LOCK_FILE = os.path.join(_RUNTIME_DIR, "start_servers.lock")


def _read_pid(path: str) -> int | None:
    try:
        return int(open(path).read().strip())
    except Exception:
        return None


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except (ProcessLookupError, PermissionError):
        return False


def _already_started(pid_file: str) -> bool:
    """Return True if a live process is already recorded in pid_file."""
    pid = _read_pid(pid_file)
    return pid is not None and _pid_alive(pid)


def _find_binary(name: str) -> str:
    """
    Locate a binary by name using multiple strategies:
    1. shutil.which (respects PATH)
    2. Same directory as the real Python interpreter (resolving symlinks)
    3. Same directory as the auto-simctl-mcp entry point
    Raises FileNotFoundError with install instructions if nothing is found.
    """
    import shutil
    import sys

    # 1. Standard PATH lookup
    found = shutil.which(name)
    if found:
        return found

    # 2. Resolve the real Python bin directory (handles /usr/local/bin symlinks)
    real_python = os.path.realpath(sys.executable)
    candidate = os.path.join(os.path.dirname(real_python), name)
    if os.path.isfile(candidate):
        return candidate

    # 3. Same directory as the auto-simctl-mcp script itself
    mcp_bin = shutil.which("auto-simctl-mcp")
    if mcp_bin:
        candidate = os.path.join(os.path.dirname(os.path.realpath(mcp_bin)), name)
        if os.path.isfile(candidate):
            return candidate

    raise FileNotFoundError(
        f"'{name}' not found. Install it with: pip install mlx-openai-server"
    )


def _find_ui_server_script() -> str:
    """
    Locate ui_server.py whether installed as a package or run from source.
    Prefers the installed auto-simctl-ui-server entry point, then importlib.
    """
    import shutil
    import importlib.util

    # Prefer the installed entry-point binary
    ep = shutil.which("auto-simctl-ui-server")
    if ep:
        return ep  # caller invokes it directly as a script

    # Fall back to the module file (works for editable + regular installs)
    spec = importlib.util.find_spec("ui_server")
    if spec and spec.origin:
        return spec.origin  # caller uses: python3 <path>

    raise FileNotFoundError(
        "ui_server not found. Re-install auto-simctl: pip install --upgrade auto-simctl"
    )


def _start_background(cmd: list[str], log_path: str, pid_path: str) -> int:
    """Launch cmd in background, redirect output to log_path, write PID to pid_path."""
    import subprocess
    with open(log_path, "a") as lf:
        proc = subprocess.Popen(
            cmd,
            stdout=lf,
            stderr=lf,
            start_new_session=True,
        )
    open(pid_path, "w").write(str(proc.pid))
    return proc.pid


def _kill_pid_file(label: str, pid_path: str) -> str:
    import signal
    pid = _read_pid(pid_path)
    if pid and _pid_alive(pid):
        try:
            os.kill(pid, signal.SIGTERM)
            return f"{label} stopped (PID {pid})"
        except Exception as e:
            return f"{label}: failed to stop — {e}"
    return f"{label} was not running"


# ── Tool: start_servers ───────────────────────────────────────────────────────

@mcp.tool()
def start_servers(
    qwen_model: str = str(
        os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub",
                     "qwen3.5-9b-mlx-4bit")
    ),
    uiug_model: str = str(
        os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub",
                     "ui-ug-7b-2601-4bit")
    ),
    qwen_port: int = 8080,
    uiug_port: int = 8081,
    timeout: int = 120,
) -> str:
    """
    Start the Qwen reasoning server and the UI-UG vision server in the background.

    CALL THIS FIRST before using act() or run_task() with non-gesture commands.
    Gesture-only commands (swipe, scroll, back) work without servers.

    Both servers are long-running background processes — they persist after this
    call returns.  You only need to call start_servers() once per session; the
    servers stay alive until stop_servers() is called or the machine reboots.

    Args:
        qwen_model: Path to the Qwen MLX model directory (default: auto-detected).
        uiug_model: Path to the UI-UG model directory (default: auto-detected).
        qwen_port:  Port for the Qwen reasoning server (default 8080).
        uiug_port:  Port for the UI-UG vision server (default 8081).
        timeout:    Seconds to wait for each server to become ready (default 120).

    Returns:
        JSON with status of both servers: { "qwen": "...", "uiug": "...", "ready": bool }
    """
    import fcntl
    import sys

    from agents.qwen_agent import QwenAgent
    from agents.ui_agent import UIAgent

    # ── Exclusive file lock — prevents multiple MCP instances from starting
    #    duplicate servers concurrently (Cursor spawns 3+ extension-host procs).
    lock_fd = open(_START_LOCK_FILE, "w")
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_EX)
        return _start_servers_locked(
            qwen_model=qwen_model,
            uiug_model=uiug_model,
            qwen_port=qwen_port,
            uiug_port=uiug_port,
            timeout=timeout,
        )
    finally:
        fcntl.flock(lock_fd, fcntl.LOCK_UN)
        lock_fd.close()


def _start_servers_locked(
    qwen_model: str,
    uiug_model: str,
    qwen_port: int,
    uiug_port: int,
    timeout: int,
) -> str:
    """Actual start logic — must only be called while holding _START_LOCK_FILE."""
    import sys

    from agents.qwen_agent import QwenAgent
    from agents.ui_agent import UIAgent

    results: dict[str, Any] = {}

    # ── Validate model paths before attempting to start ───────────────────────
    if not os.path.isfile(os.path.join(qwen_model, "config.json")):
        results["qwen"] = (
            f"ERROR: Qwen model not found at '{qwen_model}'. "
            "Download it with: "
            "python3 -c \"from huggingface_hub import snapshot_download; "
            "snapshot_download('mlx-community/Qwen3.5-9B-4bit', "
            "local_dir='~/.cache/huggingface/hub/qwen3.5-9b-mlx-4bit')\""
        )
    if not os.path.isfile(os.path.join(uiug_model, "config.json")):
        results["uiug"] = (
            f"ERROR: UI-UG model not found at '{uiug_model}'. "
            "Download it with: "
            "python3 -c \"from huggingface_hub import snapshot_download; "
            "snapshot_download('neovateai/UI-UG-7B-2601', "
            "local_dir='~/.cache/huggingface/hub/ui-ug-7b-2601')\""
        )
    if results:
        results["ready"] = False
        return json.dumps(results, ensure_ascii=False)

    # ── Qwen ─────────────────────────────────────────────────────────────────
    qwen = QwenAgent(model_path=qwen_model)
    # Double-check: port listening OR live PID already recorded
    if qwen.server_running() or _already_started(_QWEN_PID_FILE):
        results["qwen"] = f"already running on port {qwen_port}"
    else:
        try:
            mlx_bin = _find_binary("mlx-openai-server")
        except FileNotFoundError as e:
            results["qwen"] = f"ERROR: {e}"
            results["uiug"] = "skipped (Qwen failed)"
            results["ready"] = False
            return json.dumps(results, ensure_ascii=False)

        _start_background(
            [mlx_bin, "launch",
             "--model-path", qwen_model,
             "--model-type", "multimodal",
             "--port", str(qwen_port),
             "--host", "127.0.0.1",
             "--max-tokens", "4096"],
            log_path=_QWEN_LOG_FILE,
            pid_path=_QWEN_PID_FILE,
        )
        deadline = time.time() + timeout
        while time.time() < deadline:
            if qwen.server_running():
                break
            time.sleep(2)
        if qwen.server_running():
            results["qwen"] = f"started and ready on port {qwen_port}"
        else:
            results["qwen"] = (
                f"ERROR: did not become ready within {timeout}s. "
                f"Check logs: {_QWEN_LOG_FILE}"
            )

    # ── UI-UG ─────────────────────────────────────────────────────────────────
    ui = UIAgent(server_url=f"http://127.0.0.1:{uiug_port}")
    if ui.server_running() or _already_started(_UIUG_PID_FILE):
        results["uiug"] = f"already running on port {uiug_port}"
    else:
        try:
            ui_script = _find_ui_server_script()
        except FileNotFoundError as e:
            results["uiug"] = f"ERROR: {e}"
            results["ready"] = False
            return json.dumps(results, ensure_ascii=False)

        if ui_script.endswith(".py"):
            ui_cmd = [sys.executable, ui_script,
                      "--port", str(uiug_port), "--model-path", uiug_model]
        else:
            ui_cmd = [ui_script, "--port", str(uiug_port), "--model-path", uiug_model]

        _start_background(ui_cmd, log_path=_UIUG_LOG_FILE, pid_path=_UIUG_PID_FILE)
        deadline = time.time() + timeout
        while time.time() < deadline:
            if ui.server_running():
                break
            time.sleep(2)
        if ui.server_running():
            results["uiug"] = f"started and ready on port {uiug_port}"
        else:
            results["uiug"] = (
                f"ERROR: did not become ready within {timeout}s. "
                f"Check logs: {_UIUG_LOG_FILE}"
            )

    results["ready"] = all("ERROR" not in v for v in results.values())
    return json.dumps(results, ensure_ascii=False)


# ── Tool: stop_servers ────────────────────────────────────────────────────────

@mcp.tool()
def stop_servers() -> str:
    """
    Stop the Qwen reasoning server and UI-UG vision server.

    Use this to free GPU/memory when you are done with act() / run_task() tasks.
    Gesture commands (swipe, scroll, back) will continue to work after stopping.

    Returns:
        JSON with stop status for each server.
    """
    return json.dumps({
        "qwen": _kill_pid_file("Qwen", _QWEN_PID_FILE),
        "uiug": _kill_pid_file("UI-UG", _UIUG_PID_FILE),
    }, ensure_ascii=False)


# ── Tool: list_devices ────────────────────────────────────────────────────────

@mcp.tool()
def list_devices() -> str:
    """
    List all available devices: booted iOS Simulators and connected Android devices.

    Returns a JSON array.  Use the 'udid' field with other tools to target a
    specific device.  Pass device_udid="auto" (the default) to always pick the
    first booted device automatically.
    """
    bridge = _get_bridge()
    try:
        devices = bridge.list_devices()
    except Exception as e:
        return json.dumps({"error": str(e)})

    return json.dumps(
        [
            {
                "udid": d.udid,
                "name": d.name,
                "type": d.device_type.value,
                "state": d.state.value,
                "os_version": d.os_version,
            }
            for d in devices
        ],
        ensure_ascii=False,
        indent=2,
    )


# ── Tool: get_screen_state ────────────────────────────────────────────────────

@mcp.tool()
def get_screen_state(
    device_udid: str = "auto",
    include_screenshot: bool = True,
) -> list:
    """
    Return the current screen state of the device.

    Provides everything needed to decide what to do next:
    - JSON summary: foreground app, visible UI elements with tap coordinates,
      off-screen elements (hints for scrolling), scroll boundaries, keyboard state.
    - A screenshot image so you can see the actual screen (when include_screenshot=True).

    Use this BEFORE calling act() to understand the current state, and AFTER
    calling act() to observe what changed.

    Args:
        device_udid:        Device UDID or "auto" (default) for the first booted device.
        include_screenshot: Include the screenshot image in the response (default True).
                            Set False for faster/lighter polling loops.

    Returns:
        A list with [JSON text, screenshot image] so both text and visual
        context are available to the caller.
    """
    udid, err = _resolve_device(device_udid)
    if err:
        return [err]

    bridge = _get_bridge()
    try:
        shot = bridge.screenshot(udid)
    except Exception as e:
        return [json.dumps({"error": f"Screenshot failed: {e}"})]

    try:
        elements = bridge.list_elements(udid)
    except Exception:
        elements = []

    try:
        scroll = bridge.get_scroll_info(udid) if elements else {}
    except Exception:
        scroll = {}

    try:
        fg = bridge.get_foreground_app(udid)
    except Exception:
        fg = None

    keyboard_open = any(
        e.get("type") == "Button" and len(e.get("label", "").strip()) == 1
        for e in elements
    )
    visible = [e for e in elements if e.get("visible", True)]
    offscreen = [e for e in elements if not e.get("visible", True)]

    # Build a compact element description for the LLM caller.
    def _el(e: dict) -> dict[str, Any]:
        out: dict[str, Any] = {"label": e.get("label", ""), "type": e.get("type", "")}
        if "cx" in e:
            out["cx"] = e["cx"]
        if "cy" in e:
            out["cy"] = e["cy"]
        return out

    state: dict[str, Any] = {
        "foreground_app": fg,
        "keyboard_open": keyboard_open,
        "scroll": {
            "has_content_above": scroll.get("has_content_above", False),
            "has_content_below": scroll.get("has_content_below", False),
            "has_content_left":  scroll.get("has_content_left",  False),
            "has_content_right": scroll.get("has_content_right", False),
        },
        "visible_elements":   [_el(e) for e in visible],
        "offscreen_elements": [{"label": e.get("label", ""), "type": e.get("type", "")}
                                for e in offscreen],
        # Human-readable summary (makes it easier for the caller to parse)
        "summary": _build_screen_summary(fg, keyboard_open, scroll, visible, offscreen),
    }

    json_text = json.dumps(state, ensure_ascii=False, indent=2)

    if include_screenshot:
        img = Image(data=shot.png_bytes, format="png")
        return [json_text, img]
    return [json_text]


def _build_screen_summary(
    fg: dict | None,
    keyboard_open: bool,
    scroll: dict,
    visible: list[dict],
    offscreen: list[dict],
) -> str:
    parts: list[str] = []

    if fg:
        parts.append(f"App: {fg.get('name', '?')} ({fg.get('bundle_id', '?')})")
    else:
        parts.append("App: unknown (SpringBoard / home screen)")

    if keyboard_open:
        parts.append("Keyboard: open")

    vis_labels = [e.get("label", "") for e in visible if e.get("label")]
    if vis_labels:
        parts.append(f"Visible: {', '.join(vis_labels[:12])}"
                     + (" …" if len(vis_labels) > 12 else ""))

    if offscreen:
        off_labels = [e.get("label", "") for e in offscreen if e.get("label")]
        if off_labels:
            parts.append(f"Off-screen: {', '.join(off_labels[:6])}"
                         + (" …" if len(off_labels) > 6 else ""))

    scroll_hints = []
    if scroll.get("has_content_above"):
        scroll_hints.append("↑ content above")
    if scroll.get("has_content_below"):
        scroll_hints.append("↓ content below")
    if scroll.get("has_content_left"):
        scroll_hints.append("← content left")
    if scroll.get("has_content_right"):
        scroll_hints.append("→ content right")
    if scroll_hints:
        parts.append("Scroll: " + ", ".join(scroll_hints))

    return " | ".join(parts)


# ── Tool: act ─────────────────────────────────────────────────────────────────

@mcp.tool()
def act(
    task: str,
    device_udid: str = "auto",
) -> str:
    """
    Execute ONE atomic action on the device, starting from the CURRENT screen.

    This tool does NOT press HOME or reset to the launcher first — it acts on
    whatever is currently visible.  It is designed for the vibe-coding loop:

        get_screen_state → decide → act → get_screen_state → decide → act → …

    One-shot semantics
    ------------------
    tap / swipe / scroll / pan always return immediately after one execution —
    there is no post-action verification loop.  The caller decides what to do
    next by calling get_screen_state() again.

    Supported commands (natural language):

    Gestures (fast-path, Qwen not needed):
      swipe right / swipe left / swipe up / swipe down
      scroll up / scroll down
      back / go back

    Tap:
      tap <name>
      e.g. "tap Watch app", "tap Settings", "tap OK"

    Text input (two-step pattern):
      Step 1 — open the keyboard:
        act("tap address bar")          → taps the field, keyboard appears
      Step 2 — type (fast-path when keyboard is open):
        act("input_text https://…")    → detects open keyboard, types directly
      Step 3 — submit (if needed):
        act("press enter") / act("press go")
          → input_text does NOT press Enter/Return automatically;
            if the action requires submission (URL navigation, search, form
            submit), you must issue a separate act() to press the key.

    Text input (compound, one call):
      act("type https://… on address textfield")
        → taps the field, waits for keyboard, types — all in one act() call
        → does NOT press Enter; follow up with act("press enter") if needed

    Args:
        task:        What to do — e.g. "scroll down", "tap Watch app",
                     "input_text hello world", "type https://google.com on address bar"
        device_udid: Device UDID or "auto" (default) for the first booted device.

    Returns:
        JSON: { "success": bool, "action": str, "conclusion": str, "steps": int }
    """
    udid, err = _resolve_device(device_udid)
    if err:
        return err

    # Gesture fast-path: no Qwen server needed
    from orchestrator.loop import _detect_gesture
    gesture = _detect_gesture(task)
    if gesture is None:
        # Non-gesture act requires Qwen
        server_err = _check_servers()
        if server_err:
            return json.dumps({"success": False, "error": server_err})

    orch = _get_orchestrator()
    try:
        result = orch.run(task=task, device_udid=udid, reset=False)
    except Exception as e:
        log.error(f"act failed: {e}")
        return json.dumps({"success": False, "error": str(e)})

    # Return a compact result (avoid flooding the caller with step logs)
    action_str = str(result.logs[0].action) if result.logs else "unknown"
    return json.dumps(
        {
            "success": result.success,
            "action": action_str,
            "conclusion": result.conclusion,
            "steps": result.steps_taken,
        },
        ensure_ascii=False,
    )



# ── Entry point ───────────────────────────────────────────────────────────────


def main() -> None:
    """Entry point for the ``auto-simctl-mcp`` console script."""
    mcp.run()


if __name__ == "__main__":
    main()
