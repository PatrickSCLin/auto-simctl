"""
auto-simctl MCP Server — exposes the orchestrator as an MCP tool.

Allows Cursor, Claude Desktop, and other MCP-compatible vibe coding tools
to call `run_task` and `list_devices` directly from the coding loop.

Usage (once fastmcp is installed):
    python mcp_server/server.py

Then add to Cursor/Claude MCP config:
    {
      "mcpServers": {
        "auto-simctl": {
          "command": "python",
          "args": ["/path/to/auto-simctl/mcp_server/server.py"]
        }
      }
    }
"""
from __future__ import annotations

import json
import sys
import os

# Make project root importable
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

try:
    from fastmcp import FastMCP
except ImportError:
    print("fastmcp not installed. Run: pip install fastmcp", file=sys.stderr)
    sys.exit(1)

from agents.qwen_agent import QwenAgent
from agents.ui_agent import UIAgent
from mdb.bridge import DeviceBridge
from orchestrator.loop import Orchestrator

mcp = FastMCP("auto-simctl")

# Lazily initialized singletons
_mdb: DeviceBridge | None = None
_qwen: QwenAgent | None = None
_ui: UIAgent | None = None
_orchestrator: Orchestrator | None = None


def _get_orchestrator() -> Orchestrator:
    global _mdb, _qwen, _ui, _orchestrator
    if _orchestrator is None:
        _mdb = DeviceBridge()
        _qwen = QwenAgent()
        _ui = UIAgent()
        _orchestrator = Orchestrator(mdb=_mdb, qwen=_qwen, ui_agent=_ui, max_steps=20)
    return _orchestrator


@mcp.tool()
def list_devices() -> str:
    """
    List all connected Android devices and available iOS Simulators.
    Returns a JSON array of device info objects.
    """
    bridge = DeviceBridge()
    devices = bridge.list_devices()
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


@mcp.tool()
def run_task(task: str, device_udid: str = "auto") -> str:
    """
    Run an AI-driven task on a mobile device.

    The agent will:
    1. Take a screenshot of the device
    2. Identify UI elements using UI-UG-7B-2601
    3. Decide the next action using Qwen3.5-2B
    4. Execute the action on the device
    5. Repeat until the task is complete or max steps reached

    Args:
        task: Natural language task description, e.g. "Open Settings and enable Dark Mode"
        device_udid: Device UDID from list_devices(), or "auto" to pick the first available

    Returns:
        JSON result with success, steps_taken, conclusion, and evidence
    """
    orch = _get_orchestrator()

    if device_udid == "auto":
        device = orch.mdb.first_device()
        if device is None:
            return json.dumps({
                "success": False,
                "conclusion": "No devices available. Connect a device or start an iOS Simulator.",
                "steps_taken": 0,
            })
        device_udid = device.udid

    # Ensure Qwen server is running
    if not orch.qwen.server_running():
        try:
            orch.qwen.start_server()
        except Exception as e:
            return json.dumps({
                "success": False,
                "conclusion": f"Failed to start Qwen inference server: {e}",
                "steps_taken": 0,
            })

    result = orch.run(task=task, device_udid=device_udid)
    return result.to_json()


@mcp.tool()
def screenshot(device_udid: str = "auto") -> str:
    """
    Take a screenshot from the specified device.
    Returns the image as a base64-encoded PNG data URL.

    Args:
        device_udid: Device UDID or "auto" for first available device
    """
    bridge = DeviceBridge()

    if device_udid == "auto":
        dev = bridge.first_device()
        if dev is None:
            return json.dumps({"error": "No devices available"})
        device_udid = dev.udid

    shot = bridge.screenshot(device_udid)
    return json.dumps({"data_url": shot.data_url, "device_udid": device_udid})


if __name__ == "__main__":
    mcp.run()
