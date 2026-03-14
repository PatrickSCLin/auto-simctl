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
  list_devices()               List all booted simulators / connected devices
  get_screen_state()           Current screen: JSON summary + screenshot image
  act(task)                    ONE atomic action from current state (no reset)
  run_task(task)               Full autonomous multi-step AI task (with HOME reset)

Setup (Cursor / Claude Desktop)
--------------------------------
Add to your MCP config:

  {
    "mcpServers": {
      "auto-simctl": {
        "command": "python3",
        "args": ["/path/to/auto-simctl/mcp_server/server.py"]
      }
    }
  }

Prerequisites
-------------
  python3 cli.py server start   # start Qwen (:8080) + UI-UG (:8081) servers
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
        "Use get_screen_state to observe the current screen (returns JSON + screenshot), "
        "then act() to issue a single atomic command, or run_task() for a full multi-step goal. "
        "Typical vibe-coding loop: get_screen_state → act → get_screen_state → act → …"
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


def _get_orchestrator(max_steps: int = 20):
    """Lazily build the Orchestrator (imports heavy deps only when needed)."""
    from agents.qwen_agent import QwenAgent
    from agents.ui_agent import UIAgent
    from orchestrator.loop import Orchestrator

    bridge = _get_bridge()
    qwen = QwenAgent()
    ui = UIAgent()
    return Orchestrator(mdb=bridge, qwen=qwen, ui_agent=ui, max_steps=max_steps)


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
            + ". Start them with: python3 cli.py server start"
        )
    return None


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


# ── Tool: run_task ────────────────────────────────────────────────────────────

@mcp.tool()
def run_task(
    task: str,
    device_udid: str = "auto",
    max_steps: int = 20,
) -> str:
    """
    Run a full multi-step AI-driven task on a device, starting from the HOME screen.

    The agent will autonomously:
    1. Navigate to the home screen (pre-flight reset)
    2. Take screenshots and read accessibility elements each step
    3. Decide and execute actions using the Qwen reasoning model
    4. Repeat until the task is complete or max_steps is reached

    Use this for complex goals like "Open Settings → Wi-Fi → toggle off".
    For single-step commands, prefer act() instead (faster, no reset).

    Args:
        task:        Natural language goal, e.g. "Open Files app and find recent documents"
        device_udid: Device UDID or "auto" (default) for the first booted device.
        max_steps:   Maximum number of actions before giving up (default 20).

    Returns:
        JSON with: success, conclusion, steps_taken, blocked_reason, and per-step logs.
    """
    udid, err = _resolve_device(device_udid)
    if err:
        return err

    server_err = _check_servers()
    if server_err:
        return json.dumps({"success": False, "error": server_err})

    orch = _get_orchestrator(max_steps=max_steps)
    try:
        result = orch.run(task=task, device_udid=udid, reset=True)
    except Exception as e:
        log.error(f"run_task failed: {e}")
        return json.dumps({"success": False, "error": str(e)})

    # Include a concise evidence list (no raw screenshots — too large for MCP)
    evidence = [
        {
            "step": log_entry.step,
            "action": str(log_entry.action),
            "error": log_entry.error,
        }
        for log_entry in result.logs
    ]
    return json.dumps(
        {
            "success": result.success,
            "conclusion": result.conclusion,
            "steps_taken": result.steps_taken,
            "blocked_reason": result.blocked_reason,
            "evidence": evidence,
        },
        ensure_ascii=False,
        indent=2,
    )


# ── Entry point ───────────────────────────────────────────────────────────────


def main() -> None:
    """Entry point for the ``auto-simctl-mcp`` console script."""
    mcp.run()


if __name__ == "__main__":
    main()
